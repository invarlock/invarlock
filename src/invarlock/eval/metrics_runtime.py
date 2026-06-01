"""Runtime perplexity, latency, and memory metrics."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, cast

import torch
import torch.nn as nn

from invarlock.core.exceptions import MetricsError, ValidationError
from invarlock.eval.data_support import EvaluationWindow
from invarlock.eval.metrics_runtime_resources import (
    cleanup_memory_measurement_failure as _cleanup_memory_measurement_failure,
)
from invarlock.eval.metrics_runtime_resources import (
    current_memory_mb as _current_memory_mb,
)
from invarlock.eval.metrics_runtime_resources import (
    latency_validation_error as _latency_validation_error,
)
from invarlock.eval.metrics_runtime_resources import (
    maybe_cuda_synchronize as _maybe_cuda_synchronize,
)
from invarlock.eval.metrics_runtime_resources import (
    memory_measurement_baseline as _memory_measurement_baseline,
)
from invarlock.eval.metrics_runtime_resources import (
    memory_validation_error as _memory_validation_error,
)

logger = logging.getLogger(__name__)
_METRICS_RUNTIME_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_MODEL_OUTPUT_FALLBACK_ERRORS = (AttributeError, TypeError, ValueError, RuntimeError)


def call_model(model: nn.Module, /, *args: Any, **kwargs: Any) -> Any:
    return cast(Any, model)(*args, **kwargs)


def forward_logits_causal(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    try:
        outputs = call_model(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, tuple | list) and outputs:
            logits = outputs[0]
    except _MODEL_OUTPUT_FALLBACK_ERRORS:
        outputs = call_model(model, input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, tuple | list):
            logits = outputs[0] if outputs else None
        else:
            logits = getattr(outputs, "logits", None)
            if logits is None:
                logits = outputs

    if logits is None:
        raise MetricsError(
            code="E401",
            message="METRICS-COMPUTE-FAILED: model returned neither loss nor logits",
        )
    if not isinstance(logits, torch.Tensor):
        raise MetricsError(
            code="E401",
            message="METRICS-COMPUTE-FAILED: model logits must be a tensor",
        )
    return logits


def forward_loss_causal(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor | None]:
    import torch.nn.functional as F

    try:
        outputs = call_model(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return float(outputs.loss.detach().cpu()), getattr(outputs, "logits", None)
        logits = getattr(outputs, "logits", None)
    except (TypeError, AttributeError):
        outputs = call_model(
            model, input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        if isinstance(outputs, tuple | list):
            if (
                labels is not None
                and len(outputs) >= 2
                and torch.is_tensor(outputs[0])
                and outputs[0].ndim == 0
            ):
                return float(outputs[0].detach().cpu()), outputs[1] if len(
                    outputs
                ) > 1 else None
            logits = outputs[0] if len(outputs) > 0 else None
        else:
            maybe_loss = getattr(outputs, "loss", None)
            maybe_logits = getattr(outputs, "logits", None)
            if maybe_loss is not None:
                return float(maybe_loss.detach().cpu()), maybe_logits
            logits = maybe_logits

    if logits is None:
        raise MetricsError(
            code="E401",
            message="METRICS-COMPUTE-FAILED: model returned neither loss nor logits",
        )

    if labels is None:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={"reason": "labels are required to compute perplexity loss"},
        )

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return float(loss.detach().cpu()), logits


class PerplexityStatus:
    """Quality status levels for ppl-like primary metrics (perplexity)."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"

    @classmethod
    def from_value(cls, ppl: float, vocab_size: int | None = None) -> str:
        del vocab_size
        if ppl < 50:
            return cls.EXCELLENT
        if ppl < 100:
            return cls.GOOD
        if ppl < 200:
            return cls.ACCEPTABLE
        if ppl < 500:
            return cls.POOR
        return cls.UNUSABLE


def validate_perplexity(
    ppl: float,
    vocab_size: int | None = None,
    context: str = "evaluation",
    warn_threshold: float = 200.0,
    error_threshold: float = 2000.0,
    allow_high: bool = False,
) -> tuple[bool, str, str]:
    if math.isnan(ppl) or math.isinf(ppl):
        return False, "invalid", f"Perplexity is {ppl}"
    if ppl < 1.0:
        return False, "invalid", f"Perplexity {ppl:.2f} is less than 1.0"

    status = PerplexityStatus.from_value(ppl, vocab_size)

    if vocab_size is not None:
        warn_threshold = max(warn_threshold, vocab_size * 0.5)
        error_threshold = max(error_threshold, vocab_size * 2.0)

    if ppl > error_threshold and not allow_high:
        message = (
            f"Perplexity {ppl:.1f} exceeds error threshold {error_threshold:.0f} "
            f"in {context}. Model appears to be untrained or corrupted."
        )
        return False, status, message

    if ppl > warn_threshold:
        message = (
            f"Perplexity {ppl:.1f} exceeds warning threshold {warn_threshold:.0f} "
            f"in {context}. Model may be severely degraded."
        )
        if not allow_high:
            logger.warning(message)
        return True, status, message

    if status == PerplexityStatus.POOR:
        message = f"Perplexity {ppl:.1f} indicates poor model quality in {context}."
        logger.info(message)
        return True, status, message

    if status == PerplexityStatus.ACCEPTABLE:
        message = f"Perplexity {ppl:.1f} is acceptable for {context}."
        return True, status, message

    message = f"Perplexity {ppl:.1f} is {status} for {context}."
    return True, status, message


def _resolve_eval_device(
    model: nn.Module, device: str | torch.device | None
) -> torch.device:
    if device is None:
        try:
            resolved = next(model.parameters()).device
        except StopIteration:
            resolved = torch.device("cpu")
    else:
        resolved = torch.device(device) if isinstance(device, str) else device

    if isinstance(resolved, torch.device) and resolved.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        is_available = bool(
            mps_backend is not None
            and hasattr(mps_backend, "is_available")
            and mps_backend.is_available()
        )
        if not is_available:
            logger.warning(
                "Requested device 'mps' for metrics evaluation but MPS backend "
                "is not available; falling back to CPU."
            )
            resolved = torch.device("cpu")

    return resolved


def _infer_model_vocab_size(model: nn.Module) -> int | None:
    get_emb = getattr(model, "get_input_embeddings", None)
    if callable(get_emb):
        emb = get_emb()
        weight = getattr(emb, "weight", None)
        if weight is not None and hasattr(weight, "shape"):
            size = int(weight.shape[0])
            if size > 0:
                return size

    max_embeddings = 0
    modules_iter = getattr(model, "modules", None)
    if callable(modules_iter):
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                max_embeddings = max(max_embeddings, int(module.num_embeddings))
    if max_embeddings > 0:
        return max_embeddings

    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size
    return None


def _resolve_pad_token_id(model: nn.Module, vocab_size: int | None) -> int:
    config = getattr(model, "config", None)
    pad_token_id = getattr(config, "pad_token_id", None)
    if isinstance(pad_token_id, int) and pad_token_id >= 0:
        if vocab_size is None or pad_token_id < vocab_size:
            return pad_token_id
    return 0


def _sanitize_token_ids_for_model(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor | None,
    *,
    vocab_size: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if vocab_size <= 0:
        return input_ids, attention_mask, labels

    invalid_inputs = (input_ids < 0) | (input_ids >= vocab_size)
    if invalid_inputs.any():
        input_ids = input_ids.masked_fill(invalid_inputs, pad_token_id)
        if attention_mask is not None:
            attention_mask = attention_mask.masked_fill(invalid_inputs, 0)
        if labels is not None:
            labels = labels.masked_fill(invalid_inputs, -100)

    if labels is not None:
        invalid_labels = (labels != -100) & ((labels < 0) | (labels >= vocab_size))
        if invalid_labels.any():
            labels = labels.masked_fill(invalid_labels, -100)

    return input_ids, attention_mask, labels


@torch.no_grad()
def compute_perplexity_strict(
    model: nn.Module, dataloader, device: str | torch.device | None = None
) -> float:
    device = _resolve_eval_device(model, device)
    model.eval()
    model_vocab_size = _infer_model_vocab_size(model)
    pad_token_id = _resolve_pad_token_id(model, model_vocab_size)
    nll_sum = 0.0
    tok_count = 0

    for batch in dataloader:
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("inputs", None))
            labels = batch.get("labels", None)
            attention_mask = batch.get("attention_mask", None)
            token_type_ids = batch.get("token_type_ids", None)
        elif isinstance(batch, tuple | list):
            input_ids = batch[0] if len(batch) > 0 else None
            labels = batch[1] if len(batch) > 1 else None
            attention_mask = batch[2] if len(batch) > 2 else None
            token_type_ids = batch[3] if len(batch) > 3 else None
        else:
            input_ids = batch
            labels = None
            attention_mask = None
            token_type_ids = None

        if input_ids is None or not isinstance(input_ids, torch.Tensor):
            continue

        input_ids = input_ids.to(device)
        attn = attention_mask.to(device) if attention_mask is not None else None
        token_type_ids_t = (
            token_type_ids.to(device) if token_type_ids is not None else None
        )

        if labels is None:
            labels = input_ids.clone()
            if attn is not None:
                labels[attn == 0] = -100
        else:
            labels = labels.to(device)

        if model_vocab_size is not None:
            input_ids, attn, labels = _sanitize_token_ids_for_model(
                input_ids,
                attn,
                labels,
                vocab_size=model_vocab_size,
                pad_token_id=pad_token_id,
            )

        if input_ids.size(1) < 2:
            continue

        is_masked_lm = hasattr(model, "config") and getattr(
            model.config, "model_type", ""
        ) in {"bert", "roberta", "distilbert", "albert"}

        if is_masked_lm:
            masked_labels = labels.clone()
            if attn is not None:
                masked_labels = masked_labels.masked_fill(attn == 0, -100)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn,
                token_type_ids=token_type_ids_t,
                labels=masked_labels,
                return_dict=True,
            )
            loss = outputs.loss
            if loss is None:
                continue
            valid_tokens = int((masked_labels != -100).sum().item())
            if valid_tokens == 0:
                continue
            nll_sum += float(loss.item()) * valid_tokens
            tok_count += valid_tokens
            continue

        logits = forward_logits_causal(model, input_ids=input_ids, attention_mask=attn)

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = attn[:, 1:] if attn is not None else None

        valid = shift_labels != -100
        if shift_mask is not None:
            valid = valid & shift_mask.bool()
        if not valid.any():
            continue

        log_probs = shift_logits.log_softmax(dim=-1)
        vocab_size = int(shift_logits.size(-1))
        valid = valid & (shift_labels >= 0) & (shift_labels < vocab_size)
        if not valid.any():
            continue
        tgt = shift_labels.clamp(min=0, max=vocab_size - 1).unsqueeze(-1)
        nll = -log_probs.gather(-1, tgt).squeeze(-1)

        nll_sum += nll[valid].sum().item()
        tok_count += int(valid.sum().item())

    if tok_count == 0:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={
                "reason": "No valid tokens for perplexity (all masked or seq_len<=1)."
            },
        )

    return float(torch.exp(torch.tensor(nll_sum / tok_count)))


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    dataloader,
    max_samples: int = 100,
    device: str | torch.device | None = None,
) -> float:
    device = _resolve_eval_device(model, device)
    model.eval()
    model_vocab_size = _infer_model_vocab_size(model)
    pad_token_id = _resolve_pad_token_id(model, model_vocab_size)
    nll_sum = 0.0
    tok_count = 0
    batch_count = 0

    for i, batch in enumerate(dataloader):
        if max_samples is not None and i >= max_samples:
            break

        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("inputs", None))
            labels = batch.get("labels", None)
            attention_mask = batch.get("attention_mask", None)
        elif isinstance(batch, tuple | list):
            input_ids = batch[0] if len(batch) > 0 else None
            labels = batch[1] if len(batch) > 1 else None
            attention_mask = batch[2] if len(batch) > 2 else None
        else:
            input_ids = batch
            labels = None
            attention_mask = None

        if input_ids is None or not isinstance(input_ids, torch.Tensor):
            continue

        input_ids = input_ids.to(device)
        attn = attention_mask.to(device) if attention_mask is not None else None

        if labels is None:
            labels = input_ids.clone()
            if attn is not None:
                labels[attn == 0] = -100
        else:
            labels = labels.to(device)

        if model_vocab_size is not None:
            input_ids, attn, labels = _sanitize_token_ids_for_model(
                input_ids,
                attn,
                labels,
                vocab_size=model_vocab_size,
                pad_token_id=pad_token_id,
            )

        if input_ids.size(1) < 2:
            continue

        logits = forward_logits_causal(model, input_ids=input_ids, attention_mask=attn)

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = attn[:, 1:] if attn is not None else None

        valid = shift_labels != -100
        if shift_mask is not None:
            valid = valid & shift_mask.bool()
        if not valid.any():
            continue

        log_probs = shift_logits.log_softmax(dim=-1)
        vocab_size = int(shift_logits.size(-1))
        valid = valid & (shift_labels >= 0) & (shift_labels < vocab_size)
        if not valid.any():
            continue
        tgt = shift_labels.clamp(min=0, max=vocab_size - 1).unsqueeze(-1)

        if str(device).startswith("mps"):
            log_probs_cpu = log_probs.cpu()
            tgt_cpu = tgt.cpu()
            nll_cpu = -log_probs_cpu.gather(-1, tgt_cpu).squeeze(-1)
            nll = nll_cpu.to(device)
        else:
            nll = -log_probs.gather(-1, tgt).squeeze(-1)

        nll_sum += nll[valid].sum().item()
        tok_count += int(valid.sum().item())
        batch_count += 1

    if tok_count == 0:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={
                "reason": (
                    f"No valid tokens for perplexity computation after {batch_count} batches. "
                    "All tokens were either padding or sequences were too short (<=1 token). "
                    "Ensure your data contains sequences of at least 2 tokens."
                )
            },
        )

    avg_nll = nll_sum / tok_count
    ppl = float(math.exp(avg_nll))
    if ppl < 1.0:
        logger.warning(
            f"Computed perplexity {ppl:.2f} is less than 1.0, setting to 1.0"
        )
        ppl = 1.0
    elif not math.isfinite(ppl):
        logger.warning(f"Computed perplexity is not finite: {ppl}")
        ppl = float("inf")

    return ppl


@torch.no_grad()
def compute_ppl(
    model: nn.Module,
    window: EvaluationWindow,
    device: str | torch.device | None = None,
) -> float:
    device = _resolve_eval_device(model, device)
    model.eval()
    model_vocab_size = _infer_model_vocab_size(model)
    pad_token_id = _resolve_pad_token_id(model, model_vocab_size)
    nll_sum = 0.0
    tok_count = 0

    for input_ids, attention_mask in zip(
        window.input_ids, window.attention_masks, strict=False
    ):
        if not input_ids:
            continue

        input_ids_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        attention_mask_tensor = torch.LongTensor(attention_mask).unsqueeze(0).to(device)

        if model_vocab_size is not None:
            input_ids_tensor, attention_mask_tensor, _ = _sanitize_token_ids_for_model(
                input_ids_tensor,
                attention_mask_tensor,
                labels=None,
                vocab_size=model_vocab_size,
                pad_token_id=pad_token_id,
            )

        if input_ids_tensor.size(1) < 2:
            continue

        logits = forward_logits_causal(
            model,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
        )

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids_tensor[:, 1:]
        shift_mask = attention_mask_tensor[:, 1:]

        valid = (shift_labels != -100) & shift_mask.bool()
        if not valid.any():
            continue

        log_probs = shift_logits.log_softmax(dim=-1)
        vocab_size = int(shift_logits.size(-1))
        valid = valid & (shift_labels >= 0) & (shift_labels < vocab_size)
        if not valid.any():
            continue
        tgt = shift_labels.clamp(min=0, max=vocab_size - 1).unsqueeze(-1)

        if str(device).startswith("mps"):
            log_probs_cpu = log_probs.cpu()
            tgt_cpu = tgt.cpu()
            nll_cpu = -log_probs_cpu.gather(-1, tgt_cpu).squeeze(-1)
            nll = nll_cpu.to(device)
        else:
            nll = -log_probs.gather(-1, tgt).squeeze(-1)

        nll_sum += nll[valid].sum().item()
        tok_count += int(valid.sum().item())

    if tok_count == 0:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={
                "reason": "No valid tokens for perplexity computation in evaluation window",
            },
        )

    avg_nll = nll_sum / tok_count
    ppl = float(math.exp(avg_nll))
    if ppl < 1.0:
        logger.warning(
            f"Computed perplexity {ppl:.2f} is less than 1.0, setting to 1.0"
        )
        ppl = 1.0
    elif not math.isfinite(ppl):
        logger.warning(f"Computed perplexity is not finite: {ppl}")
        ppl = float("inf")

    return ppl


def measure_latency(
    model: nn.Module,
    window: EvaluationWindow,
    device: str | torch.device | None = None,
    warmup_steps: int = 3,
    measurement_steps: int = 10,
) -> float:
    device_t = _resolve_eval_device(model, device)
    model.eval()

    if not window.input_ids:
        raise _latency_validation_error(
            "latency measurement requires a non-empty evaluation window",
            {"window_size": 0},
        )

    sample_input_ids = None
    sample_attention_mask = None

    for input_ids, attention_mask in zip(
        window.input_ids, window.attention_masks, strict=False
    ):
        if len(input_ids) > 10:
            sample_input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device_t)
            sample_attention_mask = (
                torch.LongTensor(attention_mask).unsqueeze(0).to(device_t)
            )
            break

    if sample_input_ids is None or sample_attention_mask is None:
        raise _latency_validation_error(
            "latency measurement requires at least one sequence longer than 10 tokens",
            {"window_size": len(window.input_ids)},
        )

    with torch.inference_mode():
        for _ in range(warmup_steps):
            try:
                _ = call_model(
                    model,
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                )
            except _METRICS_RUNTIME_ERRORS as exc:
                raise RuntimeError("Latency warmup failed.") from exc

    _maybe_cuda_synchronize(device_t)

    start_time = time.perf_counter()
    with torch.inference_mode():
        for _ in range(measurement_steps):
            try:
                _ = call_model(
                    model,
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                )
            except _METRICS_RUNTIME_ERRORS as exc:
                raise RuntimeError("Latency measurement failed.") from exc

    _maybe_cuda_synchronize(device_t)

    total_time_ms = (time.perf_counter() - start_time) * 1000
    total_tokens = int(sample_attention_mask.sum().item()) * measurement_steps
    if total_tokens == 0:
        raise _latency_validation_error(
            "latency measurement requires at least one attended token",
            {"measurement_steps": measurement_steps},
        )

    latency_ms_per_token = total_time_ms / total_tokens
    logger.debug(
        f"Measured latency: {latency_ms_per_token:.3f} ms/token over {measurement_steps} steps"
    )
    return latency_ms_per_token


def measure_memory(
    model: nn.Module,
    window: EvaluationWindow,
    device: str | torch.device | None = None,
) -> float:
    device_t = _resolve_eval_device(model, device)
    model.eval()

    baseline_memory, process = _memory_measurement_baseline(device_t)

    max_memory = baseline_memory
    measured_samples = 0

    with torch.inference_mode():
        for i, (input_ids, attention_mask) in enumerate(
            zip(window.input_ids, window.attention_masks, strict=False)
        ):
            if i >= 5:
                break
            if not input_ids:
                continue

            try:
                input_ids_tensor = (
                    torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device_t)
                )
                attention_mask_tensor = (
                    torch.tensor(attention_mask, dtype=torch.long)
                    .unsqueeze(0)
                    .to(device_t)
                )

                _ = model(
                    input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
                )

                current_memory = _current_memory_mb(device_t, process)
                max_memory = max(max_memory, current_memory)
                measured_samples += 1
            except _METRICS_RUNTIME_ERRORS as exc:
                _cleanup_memory_measurement_failure(device_t)
                raise RuntimeError(
                    f"Memory measurement failed for sample {i}."
                ) from exc

    if measured_samples == 0:
        raise _memory_validation_error(
            "memory measurement requires at least one non-empty sample",
            {"window_size": len(window.input_ids)},
        )

    logger.debug(f"Peak memory usage: {max_memory:.1f} MB")
    return max_memory


__all__ = [
    "PerplexityStatus",
    "call_model",
    "compute_perplexity",
    "compute_perplexity_strict",
    "compute_ppl",
    "forward_logits_causal",
    "forward_loss_causal",
    "measure_latency",
    "measure_memory",
    "validate_perplexity",
]
