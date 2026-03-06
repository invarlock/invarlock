"""Guard-overhead helpers extracted from report_builder."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Any

ValidateGuardOverheadFn = Callable[..., Any]
ComputePrimaryMetricFn = Callable[..., dict[str, Any]]
GetMetricFn = Callable[[str], Any]


def prepare_guard_overhead_section(
    raw: Any,
    *,
    validate_guard_overhead_fn: ValidateGuardOverheadFn | None = None,
) -> tuple[dict[str, Any], bool]:
    """Normalize guard overhead payload and determine whether it passes the gate."""

    if validate_guard_overhead_fn is None:
        from .validate import validate_guard_overhead

        validate_guard_overhead_impl: ValidateGuardOverheadFn = validate_guard_overhead
    else:
        validate_guard_overhead_impl = validate_guard_overhead_fn

    if not isinstance(raw, dict) or not raw:
        return {}, True

    payload = copy.deepcopy(raw)

    def _coerce_float(value: Any) -> float | None:
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        return coerced if math.isfinite(coerced) else None

    threshold = _coerce_float(payload.get("overhead_threshold"))
    if threshold is None:
        threshold = 0.01
    threshold = max(0.0, threshold)

    sanitized: dict[str, Any] = {
        "overhead_threshold": threshold,
        "threshold_percent": threshold * 100,
        "source": str(payload.get("source", "report")),
    }
    try:
        mode = payload.get("mode")
        if mode is None:
            mode = payload.get("guard_overhead_mode")
        if isinstance(mode, str) and mode.strip():
            sanitized["mode"] = mode.strip()
    except Exception:
        pass
    try:
        skipped = bool(payload.get("skipped", False))
        if skipped:
            sanitized["skipped"] = True
            reason = payload.get("skip_reason")
            if isinstance(reason, str) and reason.strip():
                sanitized["skip_reason"] = reason.strip()
    except Exception:
        pass

    # Prefer structured reports and reuse the validator when available
    bare_report = payload.pop("bare_report", None)
    guarded_report = payload.pop("guarded_report", None)
    if isinstance(bare_report, dict) and isinstance(guarded_report, dict):
        result = validate_guard_overhead_impl(
            bare_report, guarded_report, overhead_threshold=threshold
        )
        metrics = result.metrics or {}
        sanitized.update(
            {
                "overhead_ratio": metrics.get("overhead_ratio"),
                "overhead_percent": metrics.get("overhead_percent"),
                "bare_ppl": metrics.get("bare_ppl"),
                "guarded_ppl": metrics.get("guarded_ppl"),
                "messages": list(result.messages),
                "warnings": list(result.warnings),
                "errors": list(result.errors),
                "checks": dict(result.checks),
                "evaluated": True,
                "passed": bool(result.passed),
            }
        )
        return sanitized, bool(result.passed)

    # Fall back to direct ratio computation when reports are not provided
    bare_ppl = _coerce_float(payload.get("bare_ppl"))
    guarded_ppl = _coerce_float(payload.get("guarded_ppl"))
    ratio = _coerce_float(payload.get("overhead_ratio"))

    if ratio is None and bare_ppl is not None and guarded_ppl is not None:
        if bare_ppl > 0:
            ratio = guarded_ppl / bare_ppl
        else:
            ratio = None

    if bare_ppl is not None:
        sanitized["bare_ppl"] = bare_ppl
    if guarded_ppl is not None:
        sanitized["guarded_ppl"] = guarded_ppl

    sanitized["messages"] = (
        [str(m) for m in payload.get("messages", [])]
        if isinstance(payload.get("messages"), list)
        else []
    )
    sanitized["warnings"] = (
        [str(w) for w in payload.get("warnings", [])]
        if isinstance(payload.get("warnings"), list)
        else []
    )
    sanitized["errors"] = (
        [str(e) for e in payload.get("errors", [])]
        if isinstance(payload.get("errors"), list)
        else []
    )
    checks = payload.get("checks")
    sanitized["checks"] = dict(checks) if isinstance(checks, dict) else {}

    if ratio is not None:
        sanitized["overhead_ratio"] = ratio
        sanitized["overhead_percent"] = (ratio - 1.0) * 100
        passed = ratio <= (1.0 + threshold)
        sanitized["evaluated"] = True
        sanitized["passed"] = passed
        return sanitized, passed

    # Unable to compute ratio – treat as not evaluated and soft-pass
    # to align with CLI/run behavior and avoid spurious failures in tiny runs.
    if not sanitized["errors"]:
        sanitized["errors"] = ["Guard overhead ratio unavailable"]
    sanitized["evaluated"] = False
    sanitized["passed"] = True
    return sanitized, True


def compute_quality_overhead_from_guard(
    raw_guard: Any,
    pm_kind_hint: str | None = None,
    *,
    compute_primary_metric_from_report_fn: ComputePrimaryMetricFn | None = None,
    get_metric_fn: GetMetricFn | None = None,
) -> dict[str, Any] | None:
    """Compute PM-aware quality overhead from guard context when possible."""

    if compute_primary_metric_from_report_fn is None or get_metric_fn is None:
        from invarlock.eval.primary_metric import (
            compute_primary_metric_from_report,
            get_metric,
        )

        compute_primary_metric_from_report_impl: ComputePrimaryMetricFn = (
            compute_primary_metric_from_report
        )
        get_metric_impl: GetMetricFn = get_metric
    else:
        compute_primary_metric_from_report_impl = compute_primary_metric_from_report_fn
        get_metric_impl = get_metric_fn

    try:
        if not isinstance(raw_guard, dict):
            return None
        bare = raw_guard.get("bare_report")
        guarded = raw_guard.get("guarded_report")
        if not (isinstance(bare, dict) and isinstance(guarded, dict)):
            return None
        kind = (
            (pm_kind_hint or "").strip().lower()
            if isinstance(pm_kind_hint, str)
            else ""
        )
        if not kind:
            kind = "ppl_causal"
        pm_b = compute_primary_metric_from_report_impl(bare, kind=kind)
        pm_g = compute_primary_metric_from_report_impl(guarded, kind=kind)
        g_point = pm_g.get("final")
        b_point = pm_b.get("final")
        if not (
            isinstance(g_point, int | float)
            and isinstance(b_point, int | float)
            and math.isfinite(float(g_point))
            and math.isfinite(float(b_point))
        ):
            return None
        # Resolve direction from registry when possible
        try:
            direction = get_metric_impl(kind).direction
        except Exception:  # pragma: no cover
            direction = str(pm_g.get("direction", "")).lower()
        if direction == "lower":
            if float(b_point) <= 0:
                return None
            value = float(g_point) / float(b_point)
            basis = "ratio"
        else:
            value = 100.0 * (float(g_point) - float(b_point))
            basis = "delta_pp"
        return {"basis": basis, "value": value, "kind": kind}
    except Exception:  # pragma: no cover
        return None
