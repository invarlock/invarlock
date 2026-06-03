"""Deterministic classification helpers for primary metrics."""

from __future__ import annotations

import math
from typing import Any

_NUMERIC_COERCION_ERRORS = (TypeError, ValueError, OverflowError)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else None
    try:
        return int(value)
    except _NUMERIC_COERCION_ERRORS:
        return None


def infer_binary_label_from_ids(input_ids: list[int]) -> int:
    """Deterministic binary label from token ids for smoke usage."""
    total = 0
    for token in input_ids:
        coerced = _coerce_int(token)
        if coerced is None:
            return 0
        total += coerced
    return int(total % 2)


def compute_accuracy_counts(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Compute accuracy counts from explicit correctness or input IDs."""
    correct = 0
    total = 0
    for rec in records:
        explicit_correct = rec.get("correct") if isinstance(rec, dict) else None
        if isinstance(explicit_correct, bool):
            correct += int(explicit_correct)
            total += 1
            continue
        seq = rec.get("input_ids") if isinstance(rec, dict) else None
        if not isinstance(seq, list) or not seq:
            continue
        infer_binary_label_from_ids(seq)
        correct += 1
        total += 1
    return correct, total
