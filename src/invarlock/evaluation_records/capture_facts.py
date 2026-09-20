"""Explicit per-case observations shared by native evaluator capture profiles."""

from __future__ import annotations

import math
from typing import Any

from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError


def merge_capture_scores(
    scores: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Retain named observations without inferring scorer or judge authority."""
    supplied = metadata.get("invarlock_scores", {})
    if not isinstance(supplied, dict):
        raise EvaluationRecordsError("invarlock_scores must be an object")
    merged = dict(scores)
    for name, value in supplied.items():
        if (
            not isinstance(name, str)
            or not name
            or type(value) not in (int, float)
            or (type(value) is float and not math.isfinite(value))
        ):
            raise EvaluationRecordsError(
                "invarlock_scores requires named finite numeric observations"
            )
        if name in merged and merged[name] != value:
            raise EvaluationRecordsError(
                f"explicit observation {name!r} conflicts with the native score"
            )
        merged[name] = value
    return merged
