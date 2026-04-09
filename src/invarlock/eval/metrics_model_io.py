"""Model-output normalization for eval metrics."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from invarlock.core.exceptions import MetricsError, ValidationError

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


__all__ = ["call_model", "forward_logits_causal", "forward_loss_causal"]
