"""Deterministic training batch preparation for evidence producers."""

from __future__ import annotations

import math
import platform
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from typing import Any


class TrainingBatchError(ValueError):
    """Raised when a tokenizer violates an immutable training profile."""


def reload_forward_smoke(
    model: Any,
    batch: Mapping[str, Any],
    *,
    torch: Any,
    device: Any,
    state_sha256: Callable[..., str],
    tensor_bytes: Callable[[Any, Any], bytes],
) -> dict[str, Any]:
    """Run two finite, byte-identical inference passes."""

    model.eval()
    replay_inputs = {name: tensor for name, tensor in batch.items() if name != "labels"}
    input_hash = state_sha256(replay_inputs, torch=torch)
    inputs = {name: tensor.to(device) for name, tensor in replay_inputs.items()}
    observations: list[tuple[str, list[int]]] = []
    for _ in range(2):
        with torch.inference_mode():
            output = model(**inputs)
        logits = (
            output.get("logits")
            if isinstance(output, Mapping)
            else getattr(output, "logits", None)
        )
        if logits is None or not torch.is_tensor(logits) or logits.numel() < 1:
            raise TrainingBatchError(
                "reloaded subject inference returned no logits tensor"
            )
        observed = logits.detach().float().cpu().contiguous()
        if not bool(torch.isfinite(observed).all()):
            raise TrainingBatchError(
                "reloaded subject inference returned non-finite logits"
            )
        observations.append(
            (
                f"sha256:{sha256(tensor_bytes(observed, torch)).hexdigest()}",
                list(observed.shape),
            )
        )
    if observations[0] != observations[1]:
        raise TrainingBatchError(
            "reloaded subject inference was not repeat-deterministic"
        )
    logits_hash, logits_shape = observations[0]
    return {
        "inference_performed": True,
        "all_logits_finite": True,
        "repeat_runs": len(observations),
        "input_sha256": input_hash,
        "logits_sha256": logits_hash,
        "logits_shape": logits_shape,
        "device": str(device),
    }


def prepare_batches(
    tokenizer: Any,
    rows: Sequence[str],
    profile: Any,
    *,
    torch: Any,
    state_sha256: Callable[..., str],
) -> tuple[list[dict[str, Any]], int, str]:
    microbatches = profile.steps * profile.gradient_accumulation_steps
    batches: list[dict[str, Any]] = []
    preprocessing_state: dict[str, Any] = {}
    token_count = 0
    cursor = 0
    for index in range(microbatches):
        texts = [
            rows[(cursor + offset) % len(rows)]
            for offset in range(profile.micro_batch_size)
        ]
        cursor += profile.micro_batch_size
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=profile.max_sequence_length,
            return_tensors="pt",
        )
        if "input_ids" not in encoded:
            raise TrainingBatchError("tokenizer did not return input_ids")
        input_ids = encoded["input_ids"]
        attention = encoded.get("attention_mask", torch.ones_like(input_ids))
        expected_shape = (profile.micro_batch_size, profile.max_sequence_length)
        if (
            tuple(input_ids.shape) != expected_shape
            or tuple(attention.shape) != expected_shape
        ):
            raise TrainingBatchError(
                "tokenizer did not produce the exact profile batch shape"
            )
        labels = input_ids.clone()
        labels.masked_fill_(attention == 0, -100)
        batches.append(
            {"input_ids": input_ids, "attention_mask": attention, "labels": labels}
        )
        preprocessing_state[f"{index:06d}.input_ids"] = input_ids
        preprocessing_state[f"{index:06d}.attention_mask"] = attention
        token_count += int(attention.sum().item())
    return batches, token_count, state_sha256(preprocessing_state, torch=torch)


def train(
    model: Any,
    parameters: Sequence[Any],
    batches: Sequence[Mapping[str, Any]],
    profile: Any,
    *,
    optimizer_cls: Any,
    device: Any,
) -> list[float]:
    optimizer = optimizer_cls(
        parameters,
        lr=profile.optimizer.learning_rate,
        betas=profile.optimizer.betas,
        eps=profile.optimizer.eps,
        weight_decay=profile.optimizer.weight_decay,
    )
    model.train()
    losses: list[float] = []
    batch_index = 0
    for _step in range(profile.steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(profile.gradient_accumulation_steps):
            batch = {
                name: tensor.to(device) for name, tensor in batches[batch_index].items()
            }
            batch_index += 1
            output = model(**batch)
            loss = output["loss"] if isinstance(output, Mapping) else output.loss
            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                raise TrainingBatchError("training produced a non-finite loss")
            (loss / profile.gradient_accumulation_steps).backward()
            accumulated += loss_value
        optimizer.step()
        losses.append(accumulated / profile.gradient_accumulation_steps)
    if len(losses) != profile.steps:
        raise TrainingBatchError("optimizer did not complete every requested step")
    return losses


def toolchain(
    torch: Any, transformers_version: str, peft_version: str | None
) -> dict[str, str]:
    value = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers_version,
    }
    if peft_version is not None:
        value["peft"] = peft_version
    return value


__all__ = [
    "TrainingBatchError",
    "prepare_batches",
    "reload_forward_smoke",
    "toolchain",
    "train",
]
