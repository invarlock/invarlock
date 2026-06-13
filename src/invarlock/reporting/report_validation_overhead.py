"""Guard-overhead validation helpers."""

from __future__ import annotations

import math
from typing import Any

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _coerce_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    return coerced if math.isfinite(coerced) else None


def guard_overhead_has_error_diagnostic(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, list | tuple):
        return False
    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        if str(item.get("severity", "")).strip().lower() == "error":
            return True
    return False


def resolve_guard_overhead_pass(
    guard_overhead: dict[str, Any] | None,
    *,
    tiny_relax: bool,
) -> bool:
    if not (isinstance(guard_overhead, dict) and guard_overhead):
        return True
    if "passed" in guard_overhead:
        guard_overhead_pass = bool(guard_overhead.get("passed"))
        if tiny_relax and (
            not bool(guard_overhead.get("evaluated", True))
            or guard_overhead_has_error_diagnostic(guard_overhead)
        ):
            return True
        return guard_overhead_pass
    ratio_val = _coerce_finite_float(guard_overhead.get("overhead_ratio"))
    threshold_val = _coerce_finite_float(guard_overhead.get("overhead_threshold", 0.01))
    if threshold_val is None:
        threshold_val = 0.01
    if tiny_relax and threshold_val < 0.10:
        threshold_val = 0.10
    if ratio_val is None:
        return True
    return ratio_val <= (1.0 + max(0.0, threshold_val))
