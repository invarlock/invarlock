from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Any

from ..types import LogLevel

_CALIBRATION_ACCESS_ERRORS = (
    AttributeError,
    IndexError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)
_LABEL_SNAPSHOT_ERRORS = (
    AttributeError,
    IndexError,
    RuntimeError,
    TypeError,
    ValueError,
)


def slice_calibration(
    calibration_data: Any,
    *,
    start: int,
    count: int,
    allow_materialize: bool,
) -> tuple[list[Any], Any]:
    end = start + count
    try:
        sliced = calibration_data[start:end]
        return (sliced if isinstance(sliced, list) else list(sliced), calibration_data)
    except _CALIBRATION_ACCESS_ERRORS as error:
        if hasattr(calibration_data, "__getitem__") and hasattr(
            calibration_data, "__len__"
        ):
            try:
                return (
                    [calibration_data[i] for i in range(start, end)],
                    calibration_data,
                )
            except _CALIBRATION_ACCESS_ERRORS:
                pass
        if allow_materialize and hasattr(calibration_data, "__iter__"):
            materialized = (
                calibration_data
                if isinstance(calibration_data, list)
                else list(calibration_data)
            )
            return (materialized[start:end], materialized)
        raise TypeError(
            "Calibration data must support slicing or random access. "
            "Provide a list/sequence or enable materialization."
        ) from error


def resolve_limit(batches: Sequence[Any], requested: int) -> int:
    if not batches:
        return 0
    if requested <= 0:
        return len(batches)
    return min(len(batches), requested)


def compute_slice_summary(
    runner: Any,
    model: Any,
    batches: Sequence[Any],
    *,
    max_batches: int,
    start_idx: int,
    device: Any,
    resolved_loss_mode: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    import torch

    total_tokens_local = 0
    actual_tokens_local = 0
    weighted_log_loss = 0.0
    log_losses: list[float] = []
    window_ids: list[int] = []
    collected_tokens: list[list[int]] = []
    collected_attn: list[list[int]] = []
    collected_labels: list[list[int]] = []
    token_counts: list[int] = []
    masked_token_counts: list[int] = []
    actual_token_counts: list[int] = []
    count = 0
    zero_mask_batches = 0
    non_finite_losses = 0
    any_labels_seen = False
    eval_error: dict[str, Any] | None = None
    store_windows = os.environ.get(
        "INVARLOCK_STORE_EVAL_WINDOWS", "1"
    ).strip().lower() not in {
        "0",
        "false",
        "no",
    }

    if not batches:
        return (
            {
                "ppl": float("nan"),
                "total_tokens": 0,
                "num_batches": 0,
                "log_losses": [],
                "window_ids": [],
                "tokens": [],
                "attention_masks": [],
                "weighted_log_loss": 0.0,
                "window_token_counts": [],
            },
            None,
        )

    limit = resolve_limit(batches, max_batches)
    alignment_logged = False

    for batch in batches[:limit]:
        if max_batches > 0 and count >= max_batches:  # pragma: no cover
            break

        labels = None
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("inputs"))
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels")
        else:
            input_ids = batch
            attention_mask = None

        if input_ids is None:
            continue

        if isinstance(input_ids, torch.Tensor):
            input_ids_t = input_ids.to(device=device, dtype=torch.long)
        else:
            input_ids_t = torch.as_tensor(input_ids, device=device, dtype=torch.long)

        if input_ids_t.dim() == 1:
            input_ids_t = input_ids_t.unsqueeze(0)

        attn_t = None
        if attention_mask is not None:
            if isinstance(attention_mask, torch.Tensor):
                attn_t = attention_mask.to(device=device, dtype=torch.long)
            else:
                attn_t = torch.as_tensor(
                    attention_mask, device=device, dtype=torch.long
                )
            if attn_t.dim() == 1:
                attn_t = attn_t.unsqueeze(0)

        if labels is not None:
            any_labels_seen = True
            if isinstance(labels, torch.Tensor):
                labels_t = labels.to(device=device, dtype=torch.long)
            else:
                labels_t = torch.as_tensor(labels, device=device, dtype=torch.long)
            if labels_t.dim() == 1:
                labels_t = labels_t.unsqueeze(0)
        else:
            labels_t = input_ids_t.clone()
            if attn_t is not None:
                labels_t = labels_t.masked_fill(attn_t == 0, -100)

        snapshot = input_ids_t.detach().cpu()
        attn_snapshot = attn_t.detach().cpu() if attn_t is not None else None

        with torch.no_grad():
            if attn_t is not None:
                outputs = model(input_ids_t, attention_mask=attn_t, labels=labels_t)
            else:
                outputs = model(input_ids_t, labels=labels_t)

        loss_val = (
            outputs.loss.item()
            if hasattr(outputs, "loss") and hasattr(outputs.loss, "item")
            else None
        )
        if loss_val is None:
            if os.environ.get("INVARLOCK_DEBUG_TRACE"):
                runner._log_event(
                    "eval",
                    "missing_loss",
                    LogLevel.DEBUG,
                    {
                        "has_loss_attr": bool(hasattr(outputs, "loss")),
                        "labels_provided": bool(labels is not None),
                        "window_index": start_idx + count,
                    },
                )
            continue
        if not isinstance(loss_val, int | float) or not math.isfinite(float(loss_val)):
            non_finite_losses += 1
            runner._log_event(
                "eval",
                "non_finite_loss",
                LogLevel.WARNING,
                {"window_index": start_idx + count, "loss": loss_val},
            )
            continue

        if attn_snapshot is not None:
            tokens_in_batch = int(attn_snapshot.sum().item())
        else:
            tokens_in_batch = int(input_ids_t.numel())

        if tokens_in_batch <= 0:
            continue

        masked_tokens_batch = int((labels_t != -100).sum().item())
        effective_masked = masked_tokens_batch
        if labels is not None and masked_tokens_batch <= 0:
            zero_mask_batches += 1
            effective_masked = tokens_in_batch
            if os.environ.get("INVARLOCK_DEBUG_TRACE"):
                sample_labels = None
                try:
                    sample_labels = labels_t[0].detach().cpu().tolist()[:8]
                except _LABEL_SNAPSHOT_ERRORS:  # pragma: no cover - defensive
                    sample_labels = None
                runner._log_event(
                    "eval",
                    "zero_mask_batch",
                    LogLevel.WARNING,
                    {
                        "window_index": start_idx + count,
                        "tokens_in_batch": tokens_in_batch,
                        "masked_tokens": masked_tokens_batch,
                        "labels_sample": sample_labels,
                        "fallback_weight": effective_masked,
                    },
                )
        effective_weight = effective_masked if labels is not None else tokens_in_batch

        if store_windows:
            for row in snapshot:
                collected_tokens.append(row.tolist())
            if attn_snapshot is not None:
                for row in attn_snapshot:
                    collected_attn.append(row.tolist())
            else:
                for row in snapshot:
                    collected_attn.append([1] * len(row))
            collected_labels.extend(labels_t.detach().cpu().tolist())

        if not alignment_logged:
            runner._log_event(
                "eval",
                "label_alignment",
                LogLevel.INFO,
                {
                    "ignore_index": -100,
                    "used_attention_mask": bool(attn_snapshot is not None),
                    "tokens_in_batch": tokens_in_batch,
                    "masked_tokens": masked_tokens_batch,
                },
            )
            alignment_logged = True

        log_losses.append(float(loss_val))
        actual_tokens_local += tokens_in_batch
        total_tokens_local += effective_weight
        weighted_log_loss += float(loss_val) * effective_weight
        token_counts.append(effective_weight)
        masked_token_counts.append(masked_tokens_batch)
        if labels is not None and masked_tokens_batch <= 0:
            masked_token_counts[-1] = effective_masked
        actual_token_counts.append(tokens_in_batch)
        window_ids.append(start_idx + count)
        count += 1

    if count == 0:
        if non_finite_losses:
            runner._log_event(
                "eval",
                "non_finite_loss_total",
                LogLevel.ERROR,
                {"non_finite_losses": non_finite_losses, "requested": limit},
            )
            eval_error = {
                "error": "non_finite_loss",
                "detail": (
                    "Evaluation produced only non-finite loss values; "
                    "primary metric evidence is unavailable."
                ),
            }
        if resolved_loss_mode == "mlm":
            error_msg = (
                "MLM evaluation produced zero usable batches; "
                "ensure baseline pairing includes masked tokens."
            )
            if any_labels_seen:
                error_msg = (
                    "MLM evaluation saw labels but zero masked tokens were accumulated; "
                    "check calibration data integrity."
                )
            runner._log_event(
                "eval",
                "mlm_missing_masks",
                LogLevel.ERROR,
                {
                    "any_labels": bool(any_labels_seen),
                    "requested": limit,
                    "zero_mask_batches": zero_mask_batches,
                },
            )
            eval_error = {"error": "mlm_missing_masks", "detail": error_msg}
        return (
            {
                "ppl": float("nan"),
                "total_tokens": total_tokens_local,
                "actual_total_tokens": actual_tokens_local,
                "num_batches": 0,
                "log_losses": [],
                "window_ids": [],
                "tokens": [],
                "attention_masks": [],
                "weighted_log_loss": 0.0,
                "window_token_counts": [],
                "masked_token_counts": [],
                "actual_token_counts": [],
                "labels": [],
            },
            eval_error,
        )

    mean_loss = (
        weighted_log_loss / total_tokens_local
        if total_tokens_local > 0
        else sum(log_losses) / max(count, 1)
    )
    return (
        {
            "ppl": float(math.exp(mean_loss)),
            "total_tokens": total_tokens_local,
            "num_batches": count,
            "log_losses": log_losses,
            "window_ids": window_ids,
            "tokens": collected_tokens,
            "attention_masks": collected_attn,
            "weighted_log_loss": weighted_log_loss,
            "window_token_counts": token_counts,
            "masked_token_counts": masked_token_counts,
            "actual_token_counts": actual_token_counts,
            "labels": collected_labels,
            "actual_total_tokens": actual_tokens_local,
        },
        eval_error,
    )


__all__ = [
    "compute_slice_summary",
    "resolve_limit",
    "slice_calibration",
]
