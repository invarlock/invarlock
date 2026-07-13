"""Guard-metric-impact validation helpers."""

from __future__ import annotations

import math
from typing import Any

from invarlock.eval.guard_metric_impact import (
    degradation_within_limit,
    guard_metric_impact_payload_errors,
)

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


def guard_metric_impact_has_error_diagnostic(payload: Any) -> bool:
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


def resolve_guard_metric_impact_pass(
    guard_metric_impact: dict[str, Any] | None,
    *,
    tiny_relax: bool,
) -> bool:
    del tiny_relax
    if not (isinstance(guard_metric_impact, dict) and guard_metric_impact):
        return False
    if guard_metric_impact.get("evaluated") is not True:
        return False
    if guard_metric_impact_has_error_diagnostic(guard_metric_impact):
        return False
    if guard_metric_impact_payload_errors(
        guard_metric_impact,
        require_bare_report=True,
    ):
        return False
    passed = guard_metric_impact.get("passed")
    if not isinstance(passed, bool):
        return False
    degradation = _coerce_finite_float(guard_metric_impact.get("degradation"))
    limit = _coerce_finite_float(guard_metric_impact.get("degradation_limit"))
    return passed and degradation_within_limit(
        degradation=degradation,
        degradation_limit=limit,
    )
