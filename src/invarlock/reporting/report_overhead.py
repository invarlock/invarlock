"""Guard-overhead normalization helpers for evaluation report assembly."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

ValidateGuardOverheadFn = Callable[..., Any]
ComputePrimaryMetricFn = Callable[..., dict[str, Any]]
GetMetricFn = Callable[[str], Any]

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class GuardOverheadSummary:
    evaluated: bool
    passed: bool
    overhead_percent: float | None
    overhead_ratio: float | None
    threshold_fraction: float


def _coerce_non_bool_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _append_guard_overhead_diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    severity: str,
    message: Any,
) -> None:
    diagnostics.append(
        {
            "kind": f"guard_overhead_{severity}",
            "severity": severity,
            "message": str(message),
            "details": {},
        }
    )


def _coerce_guard_overhead_diagnostics(raw: Any) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    if isinstance(raw, list | tuple):
        for item in raw:
            if not isinstance(item, dict):
                continue
            message = item.get("message")
            if not isinstance(message, str) or not message:
                continue
            severity = item.get("severity")
            if not isinstance(severity, str) or not severity:
                severity = "info"
            details = item.get("details")
            diagnostics.append(
                {
                    "kind": str(item.get("kind") or f"guard_overhead_{severity}"),
                    "severity": severity,
                    "message": message,
                    "details": dict(details) if isinstance(details, dict) else {},
                }
            )
    return diagnostics


def normalize_guard_overhead_result(
    payload: dict[str, object] | None,
) -> dict[str, object]:
    """Normalize guard-overhead payload for tiny or degenerate runs."""
    payload = dict(payload or {})
    try:
        ratio = payload.get("overhead_ratio")
        resolved = _coerce_non_bool_float(ratio)
        value = resolved if resolved is not None else float("nan")
    except (TypeError, ValueError):
        value = float("nan")
    if not (isinstance(value, float) and math.isfinite(value)):
        payload["evaluated"] = False
        payload["passed"] = True
    return payload


def finalize_guard_overhead_payload(
    payload: Mapping[str, Any] | None,
    result: Any,
) -> dict[str, Any]:
    """Normalize validator output into the persisted guard-overhead payload."""
    resolved = dict(payload or {})
    diagnostics = _coerce_guard_overhead_diagnostics(resolved.pop("diagnostics", ()))
    resolved.pop("messages", None)
    resolved.pop("warnings", None)
    resolved.pop("errors", None)
    diagnostics.extend(
        _coerce_guard_overhead_diagnostics(getattr(result, "diagnostics", ()))
    )
    try:
        checks = dict(getattr(result, "checks", {}))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        checks = {}

    metrics_obj = getattr(result, "metrics", {})
    if not isinstance(metrics_obj, dict):
        metrics_obj = {}

    overhead_ratio = metrics_obj.get("overhead_ratio")
    if overhead_ratio is None:
        overhead_ratio = getattr(result, "overhead_ratio", None)
    overhead_percent = metrics_obj.get("overhead_percent")
    if overhead_percent is None:
        overhead_percent = getattr(result, "overhead_percent", None)

    resolved.update(
        {
            "diagnostics": diagnostics,
            "checks": checks,
            "overhead_ratio": _coerce_non_bool_float(overhead_ratio),
            "overhead_percent": _coerce_non_bool_float(overhead_percent),
            "passed": bool(getattr(result, "passed", False)),
            "evaluated": True,
        }
    )
    normalized = normalize_guard_overhead_result(resolved)
    return dict(normalized)


def prepare_guard_overhead_report(
    guard_overhead_payload: Mapping[str, Any] | None,
    *,
    resolved_loss_type: str | None,
    core_report: Any,
    report: Mapping[str, Any] | None,
    default_threshold: float,
    extract_pm_snapshot_for_overhead_fn: Any,
    validate_guard_overhead_fn: Any,
) -> dict[str, Any]:
    """Build the persisted guard-overhead report payload."""
    payload = dict(guard_overhead_payload or {})
    if bool(payload.get("skipped", False)):
        return payload

    try:
        loss_kind = str(resolved_loss_type or "causal").lower()
        if loss_kind == "mlm":
            pm_kind_for_overhead = "ppl_mlm"
        elif loss_kind in {"seq2seq", "s2s", "t5"}:
            pm_kind_for_overhead = "ppl_seq2seq"
        else:
            pm_kind_for_overhead = "ppl_causal"

        pm_guarded = extract_pm_snapshot_for_overhead_fn(
            core_report, kind=pm_kind_for_overhead
        )
        if not isinstance(pm_guarded, dict) or not pm_guarded:
            pm_guarded = extract_pm_snapshot_for_overhead_fn(
                report, kind=pm_kind_for_overhead
            )

        payload["guarded_report"] = (
            {"metrics": {"primary_metric": pm_guarded}}
            if isinstance(pm_guarded, dict) and pm_guarded
            else None
        )
    except (AttributeError, TypeError, ValueError):
        payload["guarded_report"] = None

    bare_struct = payload.get("bare_report") or {}
    guarded_struct = payload.get("guarded_report") or {}
    result = validate_guard_overhead_fn(
        bare_struct,
        guarded_struct,
        overhead_threshold=payload.get("overhead_threshold", default_threshold),
    )
    return finalize_guard_overhead_payload(payload, result)


def build_guard_overhead_summary(
    guard_overhead_info: Mapping[str, Any] | None,
    *,
    default_threshold: float,
) -> GuardOverheadSummary:
    """Build a stable typed summary for guard-overhead results."""
    info = dict(guard_overhead_info or {})
    try:
        fallback_threshold = float(default_threshold)
        if fallback_threshold < 0.0 or not math.isfinite(fallback_threshold):
            fallback_threshold = 0.01
    except (TypeError, ValueError):
        fallback_threshold = 0.01

    evaluated = bool(info.get("evaluated", True))
    passed = bool(info.get("passed", True))

    resolved_overhead_percent: float | None = None
    overhead_percent = _coerce_non_bool_float(info.get("overhead_percent"))
    if overhead_percent is not None:
        resolved_overhead_percent = overhead_percent

    resolved_overhead_ratio: float | None = None
    ratio_value = _coerce_non_bool_float(info.get("overhead_ratio"))
    if ratio_value is not None:
        resolved_overhead_ratio = ratio_value

    threshold_value = info.get("overhead_threshold", fallback_threshold)
    threshold_fraction = _coerce_non_bool_float(threshold_value)
    if threshold_fraction is None:
        threshold_fraction = fallback_threshold
    return GuardOverheadSummary(
        evaluated=evaluated,
        passed=passed,
        overhead_percent=resolved_overhead_percent,
        overhead_ratio=resolved_overhead_ratio,
        threshold_fraction=threshold_fraction,
    )


def _append_diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    kind: str,
    severity: str,
    message: Any,
    details: Mapping[str, Any] | None = None,
) -> None:
    diagnostics.append(
        {
            "kind": kind,
            "severity": severity,
            "message": str(message),
            "details": dict(details or {}),
        }
    )


def _coerce_diagnostics(raw: Any) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, (list, tuple)) else ():
        if isinstance(item, Mapping):
            diagnostics.append(
                {
                    "kind": str(item.get("kind", "guard_overhead_diagnostic")),
                    "severity": str(item.get("severity", "info")),
                    "message": str(item.get("message", "")),
                    "details": {
                        str(key): value
                        for key, value in item.items()
                        if key not in {"kind", "severity", "message"}
                    },
                }
            )
        elif all(hasattr(item, attr) for attr in ("kind", "severity", "message")):
            details = getattr(item, "details", {})
            diagnostics.append(
                {
                    "kind": str(getattr(item, "kind", "guard_overhead_diagnostic")),
                    "severity": str(getattr(item, "severity", "info")),
                    "message": str(getattr(item, "message", "")),
                    "details": dict(details) if isinstance(details, Mapping) else {},
                }
            )
        else:
            diagnostics.append(
                {
                    "kind": "guard_overhead_diagnostic",
                    "severity": "info",
                    "message": str(item),
                    "details": {},
                }
            )
    return diagnostics


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
    except _NON_FATAL_EXCEPTIONS:
        pass
    try:
        skipped = bool(payload.get("skipped", False))
        if skipped:
            sanitized["skipped"] = True
            reason = payload.get("skip_reason")
            if isinstance(reason, str) and reason.strip():
                sanitized["skip_reason"] = reason.strip()
    except _NON_FATAL_EXCEPTIONS:
        pass

    # Prefer structured reports and reuse the validator when available
    bare_report = payload.pop("bare_report", None)
    guarded_report = payload.pop("guarded_report", None)
    if isinstance(bare_report, dict) and isinstance(guarded_report, dict):
        result = validate_guard_overhead_impl(
            bare_report, guarded_report, overhead_threshold=threshold
        )
        metrics = result.metrics or {}
        diagnostics = _coerce_diagnostics(getattr(result, "diagnostics", ()))
        sanitized.update(
            {
                "overhead_ratio": metrics.get("overhead_ratio"),
                "overhead_percent": metrics.get("overhead_percent"),
                "bare_ppl": metrics.get("bare_ppl"),
                "guarded_ppl": metrics.get("guarded_ppl"),
                "diagnostics": diagnostics,
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

    diagnostics = _coerce_diagnostics(payload.get("diagnostics"))
    sanitized["diagnostics"] = diagnostics
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
    if not diagnostics:
        sanitized["diagnostics"] = [
            {
                "kind": "guard_overhead_unavailable",
                "severity": "warning",
                "message": "Guard overhead ratio unavailable",
                "details": {},
            }
        ]
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
        except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
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
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        return None
