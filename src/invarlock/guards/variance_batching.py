from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from invarlock.utils.bootstrap import (
    bootstrap_mean_statistics,
    percentile_interval_from_statistics,
)

from .variance_types import CalibrationBatchContext

_VARIANCE_BATCHING_ERRORS = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
)


def safe_mean(
    samples: list[float] | np.ndarray, default: float | None = None
) -> float | None:
    """Compute mean of samples, returning default if empty."""
    if samples is None:
        return default
    arr = np.asarray(samples)
    if arr.size == 0:
        return default
    return float(np.nanmean(arr))


def materialize_batch(guard: Any, batch: Any) -> Any:
    """Detach tensors from device and clone calibration batches for reuse."""
    if isinstance(batch, dict):
        return {key: materialize_batch(guard, value) for key, value in batch.items()}
    if isinstance(batch, list | tuple):
        return type(batch)(materialize_batch(guard, value) for value in batch)
    if isinstance(batch, torch.Tensor):
        return batch.detach().cpu()
    try:
        return copy.deepcopy(batch)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return batch


def ensure_tensor_value(value: Any) -> Any:
    """Convert common calibration value types to torch tensors."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value)
    if isinstance(value, list | tuple):
        try:
            return torch.as_tensor(value)
        except (RuntimeError, TypeError, ValueError):
            return value
    if isinstance(value, int | float):
        return torch.tensor(value)
    return value


def tensorize_calibration_batches(guard: Any, batches: list[Any]) -> list[Any]:
    """Ensure calibration batches contain tensor payloads for model execution."""
    tensor_batches: list[Any] = []
    for batch in batches:
        if isinstance(batch, dict):
            converted: dict[str, Any] = {}
            for key, value in batch.items():
                if key in {"input_ids", "inputs", "attention_mask", "labels"}:
                    converted[key] = ensure_tensor_value(value)
                else:
                    converted[key] = value
            tensor_batches.append(converted)
        elif isinstance(batch, list | tuple):
            converted_list = [ensure_tensor_value(value) for value in batch]
            tensor_batches.append(type(batch)(converted_list))
        else:
            tensor_batches.append(ensure_tensor_value(batch))
    return tensor_batches


def extract_window_ids(guard: Any, batches: list[Any]) -> list[str]:
    """Extract window identifiers from calibration batches when present."""
    window_ids: list[str] = []
    for batch in batches:
        candidate: Any | None = None
        if isinstance(batch, dict):
            if "window_id" in batch:
                candidate = batch["window_id"]
            elif "window_ids" in batch:
                candidate = batch["window_ids"]
            elif isinstance(batch.get("metadata"), dict):
                meta = batch["metadata"]
                candidate = meta.get("window_id") or meta.get("window_ids")

        if candidate is None:
            continue
        if isinstance(candidate, list | tuple):
            window_ids.extend(str(item) for item in candidate)
        else:
            window_ids.append(str(candidate))
    if not window_ids and batches:
        window_ids = [str(index) for index in range(len(batches))]
    return window_ids


def store_calibration_batches(guard: Any, batches: list[Any]) -> None:
    """Persist calibration batches for deterministic post-edit evaluation."""
    materialized = [materialize_batch(guard, batch) for batch in batches]
    guard._calibration_batches = tensorize_calibration_batches(guard, materialized)
    guard._calibration_window_ids = extract_window_ids(
        guard, guard._calibration_batches
    )
    observed_ids = list(guard._calibration_window_ids)
    observed_digest = (
        hashlib.blake2s(
            "||".join(observed_ids).encode("utf-8"), digest_size=16
        ).hexdigest()
        if observed_ids
        else None
    )
    expected_ids = guard._expected_window_ids()
    context = CalibrationBatchContext(
        window_ids=list(observed_ids),
        count=len(guard._calibration_batches),
        observed_digest=observed_digest,
        expected_digest=guard._pairing_digest if expected_ids else None,
    )
    guard._calibration_context = {
        "window_ids": context.window_ids,
        "count": context.count,
        "observed_digest": context.observed_digest,
    }
    if context.expected_digest is not None:
        guard._calibration_context["expected_digest"] = context.expected_digest
        expected_subset = expected_ids[: len(observed_ids)] if observed_ids else []
        if observed_ids != expected_subset:
            mismatch = {
                "expected_count": len(expected_ids),
                "observed_count": len(observed_ids),
                "expected_sample": expected_subset[:5]
                if expected_subset
                else expected_ids[:5],
                "observed_sample": observed_ids[:5],
            }
            guard._log_event(
                "pairing_mismatch",
                level="ERROR",
                message="Variance guard calibration windows do not match baseline pairing",
                **mismatch,
            )
            guard._prepared = False
            raise RuntimeError(
                "Variance guard pairing mismatch: calibration windows diverge from baseline schedule"
            )
    guard._stats.setdefault("calibration", {})
    guard._stats["calibration"].update(guard._calibration_context)


def collect_calibration_batches(guard: Any, dataloader, windows: int) -> list[Any]:
    """Collect a deterministic slice of calibration batches."""
    batches: list[Any] = []
    iterator = iter(dataloader)
    for _ in range(max(windows, 0)):
        try:
            batches.append(next(iterator))
        except StopIteration:
            break
    return batches


def prepare_batch_tensors(
    guard: Any, batch: Any, device: torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Normalize batch inputs to tensors on the target device."""
    if isinstance(batch, dict):
        input_ids = batch.get("input_ids", batch.get("inputs"))
        attention_mask = batch.get("attention_mask")
    elif isinstance(batch, tuple | list) and batch:
        input_ids = batch[0]
        attention_mask = batch[1] if len(batch) > 1 else None
    else:
        input_ids = batch
        attention_mask = None

    if input_ids is None:
        return None, None

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    try:
        input_ids = input_ids.to(device)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        input_ids = input_ids.clone()

    labels = input_ids.clone()
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.as_tensor(attention_mask)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        try:
            attention_mask = attention_mask.to(device)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            attention_mask = attention_mask.clone()
        labels = labels.masked_fill(attention_mask == 0, -100)

    return input_ids, labels


def compute_ppl_for_batches(
    guard: Any,
    model: nn.Module,
    batches: list[Any],
    device: torch.device,
    *,
    return_counts: bool = False,
) -> tuple[list[float], list[float]] | tuple[list[float], list[float], list[int]]:
    """Compute per-batch perplexity and log-loss values for deterministic calibration."""
    ppl_values: list[float] = []
    loss_values: list[float] = []
    token_counts: list[int] = []
    if not batches:
        return (
            (ppl_values, loss_values, token_counts)
            if return_counts
            else (ppl_values, loss_values)
        )

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            try:
                inputs, labels = guard._prepare_batch_tensors(batch, device)
                if inputs is None or labels is None:
                    continue

                try:
                    outputs = model(inputs, labels=labels)
                except TypeError:
                    outputs = model(inputs)
                loss_val = None
                if hasattr(outputs, "loss") and hasattr(outputs.loss, "item"):
                    loss_val = outputs.loss.item()
                if loss_val is None and isinstance(outputs, torch.Tensor):
                    try:
                        if labels is not None and outputs.shape == labels.shape:
                            loss_val = torch.nn.functional.mse_loss(
                                outputs.float(), labels.float()
                            ).item()
                        else:
                            loss_val = outputs.float().pow(2).mean().item()
                    except _VARIANCE_BATCHING_ERRORS:
                        loss_val = None

                if loss_val is None or not math.isfinite(loss_val):
                    continue

                loss = float(loss_val)
                try:
                    ppl = math.exp(loss)
                except OverflowError:
                    continue
                ppl_values.append(ppl)
                loss_values.append(loss)
                if return_counts:
                    count = None
                    try:
                        if labels is not None and isinstance(labels, torch.Tensor):
                            count = int((labels != -100).sum().item())
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        count = None
                    if count is None:
                        try:
                            count = int(inputs.numel())
                        except (AttributeError, RuntimeError, TypeError, ValueError):
                            count = 0
                    token_counts.append(int(max(count, 0)))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue

    if model_was_training:
        model.train()

    if return_counts:
        return ppl_values, loss_values, token_counts
    return ppl_values, loss_values


def bootstrap_mean_ci(
    guard: Any,
    samples: list[float],
    alpha: float,
    n_bootstrap: int = 500,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the sample mean."""
    if not samples:
        raise ValueError("Cannot compute CI on empty samples")
    data = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    stats = bootstrap_mean_statistics(
        data,
        n_bootstrap=int(n_bootstrap),
        random_state=rng,
    )
    return percentile_interval_from_statistics(stats, alpha=alpha)


__all__ = [
    "bootstrap_mean_ci",
    "collect_calibration_batches",
    "compute_ppl_for_batches",
    "ensure_tensor_value",
    "extract_window_ids",
    "materialize_batch",
    "prepare_batch_tensors",
    "safe_mean",
    "store_calibration_batches",
    "tensorize_calibration_batches",
]
