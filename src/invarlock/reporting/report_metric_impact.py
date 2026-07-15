"""Guard-metric-impact normalization helpers for evaluation report assembly."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from invarlock.eval.guard_metric_impact import (
    GUARD_METRIC_IMPACT_REPORT_FIELDS,
    arm_facts_match_measurements,
    build_guard_metric_bare_report,
    compute_guard_metric_impact,
    degradation_within_limit,
    extract_guard_metric_arm_facts,
    guard_metric_schedule_digest,
    metric_value_from_arm_facts,
)

_GUARD_METRIC_IMPACT_INPUT_FIELDS = GUARD_METRIC_IMPACT_REPORT_FIELDS | {
    "errors",
    "guard_metric_impact_mode",
    "guarded_report",
    "messages",
    "warnings",
}

ValidateGuardMetricImpactFn = Callable[..., Any]
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
class GuardMetricImpactSummary:
    evaluated: bool
    passed: bool
    degradation: float | None
    display_value: float | None
    display_unit: str | None
    degradation_limit: float


def _coerce_non_bool_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _append_guard_metric_impact_diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    severity: str,
    message: Any,
) -> None:
    diagnostics.append(
        {
            "kind": f"guard_metric_impact_{severity}",
            "severity": severity,
            "message": str(message),
            "details": {},
        }
    )


def _coerce_guard_metric_impact_diagnostics(raw: Any) -> list[dict[str, Any]]:
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
                    "kind": str(item.get("kind") or f"guard_metric_impact_{severity}"),
                    "severity": severity,
                    "message": message,
                    "details": dict(details) if isinstance(details, dict) else {},
                }
            )
    return diagnostics


def normalize_guard_metric_impact_result(
    payload: dict[str, object] | None,
) -> dict[str, object]:
    """Normalize guard-metric-impact payload for tiny or degenerate runs."""
    payload = dict(payload or {})
    try:
        degradation = payload.get("degradation")
        resolved = _coerce_non_bool_float(degradation)
        value = resolved if resolved is not None else float("nan")
    except (TypeError, ValueError):
        value = float("nan")
    if not (isinstance(value, float) and math.isfinite(value)):
        payload["evaluated"] = False
        payload["passed"] = False
    return payload


def finalize_guard_metric_impact_payload(
    payload: Mapping[str, Any] | None,
    result: Any,
) -> dict[str, Any]:
    """Normalize validator output into the persisted guard-metric-impact payload."""
    resolved = dict(payload or {})
    resolved.pop("guarded_report", None)
    diagnostics = _coerce_guard_metric_impact_diagnostics(
        resolved.pop("diagnostics", ())
    )
    resolved.pop("messages", None)
    resolved.pop("warnings", None)
    resolved.pop("errors", None)
    diagnostics.extend(
        _coerce_guard_metric_impact_diagnostics(getattr(result, "diagnostics", ()))
    )
    try:
        checks = dict(getattr(result, "checks", {}))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        checks = {}

    metrics_obj = getattr(result, "metrics", {})
    if not isinstance(metrics_obj, dict):
        metrics_obj = {}

    resolved.update(
        {
            "diagnostics": diagnostics,
            "checks": checks,
            "passed": bool(getattr(result, "passed", False)),
            "evaluated": True,
        }
    )
    for field in (
        "metric_kind",
        "direction",
        "degradation_basis",
        "display_unit",
    ):
        value = metrics_obj.get(field)
        if isinstance(value, str) and value:
            resolved[field] = value
    for field in ("bare_value", "guarded_value", "degradation", "display_value"):
        resolved[field] = _coerce_non_bool_float(metrics_obj.get(field))
    normalized = normalize_guard_metric_impact_result(resolved)
    return dict(normalized)


def prepare_guard_metric_impact_report(
    guard_metric_impact_payload: Mapping[str, Any] | None,
    *,
    resolved_loss_type: str | None,
    core_report: Any,
    report: Mapping[str, Any] | None,
    default_limit: float,
    extract_pm_snapshot_for_metric_impact_fn: Any,
    validate_guard_metric_impact_fn: Any,
) -> dict[str, Any]:
    """Build the persisted guard-metric-impact report payload."""
    payload = dict(guard_metric_impact_payload or {})
    if bool(payload.get("skipped", False)):
        payload["evaluated"] = False
        payload["passed"] = False
        payload.setdefault("mode", "skipped")
        return payload

    try:
        loss_kind = str(resolved_loss_type or "causal").lower()
        if loss_kind in {"classification", "accuracy"}:
            pm_kind_for_metric_impact = "accuracy"
        elif loss_kind == "mlm":
            pm_kind_for_metric_impact = "ppl_mlm"
        elif loss_kind in {"seq2seq", "s2s", "t5"}:
            pm_kind_for_metric_impact = "ppl_seq2seq"
        else:
            pm_kind_for_metric_impact = "ppl_causal"

        pm_guarded = extract_pm_snapshot_for_metric_impact_fn(
            core_report, kind=pm_kind_for_metric_impact
        )
        if not isinstance(pm_guarded, dict) or not pm_guarded:
            pm_guarded = extract_pm_snapshot_for_metric_impact_fn(
                report, kind=pm_kind_for_metric_impact
            )

        guarded_facts = extract_guard_metric_arm_facts(
            core_report,
            pm_kind_for_metric_impact,
        )
        if guarded_facts is not None:
            payload["guarded_facts"] = guarded_facts
            replayed = metric_value_from_arm_facts(
                pm_kind_for_metric_impact,
                guarded_facts,
            )
            if replayed is not None and isinstance(pm_guarded, dict):
                pm_guarded = dict(pm_guarded)
                pm_guarded["final"] = replayed
        payload["guarded_report"] = (
            {"metrics": {"primary_metric": pm_guarded}}
            if isinstance(pm_guarded, dict) and pm_guarded
            else None
        )
    except (AttributeError, TypeError, ValueError):
        payload["guarded_report"] = None

    bare_struct = payload.get("bare_report") or {}
    guarded_struct = payload.get("guarded_report") or {}
    result = validate_guard_metric_impact_fn(
        bare_struct,
        guarded_struct,
        degradation_limit=payload.get("degradation_limit", default_limit),
    )
    finalized = finalize_guard_metric_impact_payload(payload, result)
    finalized["degradation_limit"] = _coerce_non_bool_float(
        payload.get("degradation_limit", default_limit)
    )
    schedule_digest = guard_metric_schedule_digest(
        core_report,
        finalized.get("metric_kind"),
    )
    if schedule_digest is not None:
        finalized["schedule_digest"] = schedule_digest
    facts_replay = arm_facts_match_measurements(
        finalized.get("metric_kind"),
        finalized.get("bare_facts"),
        finalized.get("guarded_facts"),
        finalized.get("bare_value"),
        finalized.get("guarded_value"),
    )
    checks = finalized.setdefault("checks", {})
    if isinstance(checks, dict):
        checks["arm_facts_replay"] = facts_replay
    if not facts_replay:
        finalized["evaluated"] = False
        finalized["passed"] = False
        diagnostics = finalized.setdefault("diagnostics", [])
        if isinstance(diagnostics, list):
            _append_guard_metric_impact_diagnostic(
                diagnostics,
                severity="error",
                message=(
                    "Guard metric impact arm facts are missing or disagree with "
                    "the retained primary-metric values"
                ),
            )
    return finalized


def build_guard_metric_impact_summary(
    guard_metric_impact_info: Mapping[str, Any] | None,
    *,
    default_limit: float,
) -> GuardMetricImpactSummary:
    """Build a stable typed summary for guard-metric-impact results."""
    info = dict(guard_metric_impact_info or {})
    try:
        fallback_limit = float(default_limit)
        if fallback_limit < 0.0 or not math.isfinite(fallback_limit):
            fallback_limit = 0.01
    except (TypeError, ValueError):
        fallback_limit = 0.01

    evaluated = info.get("evaluated") is True
    passed = evaluated and info.get("passed") is True

    display_unit = info.get("display_unit")
    resolved_unit = display_unit if isinstance(display_unit, str) else None
    resolved_display_value = _coerce_non_bool_float(info.get("display_value"))
    resolved_degradation = _coerce_non_bool_float(info.get("degradation"))

    limit_value = info.get("degradation_limit", fallback_limit)
    degradation_limit = _coerce_non_bool_float(limit_value)
    if degradation_limit is None:
        degradation_limit = fallback_limit
    return GuardMetricImpactSummary(
        evaluated=evaluated,
        passed=passed,
        degradation=resolved_degradation,
        display_value=resolved_display_value,
        display_unit=resolved_unit,
        degradation_limit=degradation_limit,
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
                    "kind": str(item.get("kind", "guard_metric_impact_diagnostic")),
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
                    "kind": str(
                        getattr(item, "kind", "guard_metric_impact_diagnostic")
                    ),
                    "severity": str(getattr(item, "severity", "info")),
                    "message": str(getattr(item, "message", "")),
                    "details": dict(details) if isinstance(details, Mapping) else {},
                }
            )
        else:
            diagnostics.append(
                {
                    "kind": "guard_metric_impact_diagnostic",
                    "severity": "info",
                    "message": str(item),
                    "details": {},
                }
            )
    return diagnostics


def _prepare_structured_metric_impact_section(
    payload: dict[str, Any],
    sanitized: dict[str, Any],
    *,
    validate_guard_metric_impact_fn: ValidateGuardMetricImpactFn,
    degradation_limit: float,
    is_skipped: bool,
) -> tuple[dict[str, Any], bool] | None:
    bare_report = payload.get("bare_report")
    guarded_report = payload.get("guarded_report")
    if (
        is_skipped
        or not isinstance(bare_report, dict)
        or not isinstance(guarded_report, dict)
    ):
        return None
    try:
        result = validate_guard_metric_impact_fn(
            bare_report, guarded_report, degradation_limit=degradation_limit
        )
    except _NON_FATAL_EXCEPTIONS as exc:
        sanitized.update(
            {
                "evaluated": False,
                "passed": False,
                "diagnostics": [
                    {
                        "kind": "guard_metric_impact_validation_error",
                        "severity": "error",
                        "message": str(exc),
                        "details": {},
                    }
                ],
                "checks": {},
            }
        )
        return sanitized, False
    metrics = result.metrics or {}
    diagnostics = _coerce_diagnostics(getattr(result, "diagnostics", ()))
    degradation = _coerce_non_bool_float(metrics.get("degradation"))
    metric_kind = metrics.get("metric_kind")
    direction = metrics.get("direction")
    degradation_basis = metrics.get("degradation_basis")
    bare_value = _coerce_non_bool_float(metrics.get("bare_value"))
    guarded_value = _coerce_non_bool_float(metrics.get("guarded_value"))
    display_value = _coerce_non_bool_float(metrics.get("display_value"))
    display_unit = metrics.get("display_unit")
    if isinstance(metric_kind, str):
        bare_envelope = build_guard_metric_bare_report(bare_report, metric_kind)
        if bare_envelope is not None:
            sanitized["bare_report"] = bare_envelope
        if "bare_facts" not in sanitized:
            facts = extract_guard_metric_arm_facts(bare_report, metric_kind)
            if facts is not None:
                sanitized["bare_facts"] = facts
        if "guarded_facts" not in sanitized:
            facts = extract_guard_metric_arm_facts(guarded_report, metric_kind)
            if facts is not None:
                sanitized["guarded_facts"] = facts
    result_passed = bool(result.passed)
    check_value = getattr(result, "checks", {}).get("guard_metric_impact")
    evaluated = bool(
        isinstance(metric_kind, str)
        and isinstance(direction, str)
        and isinstance(degradation_basis, str)
        and bare_value is not None
        and guarded_value is not None
        and degradation is not None
        and display_value is not None
        and isinstance(display_unit, str)
    )
    facts_replay = arm_facts_match_measurements(
        metric_kind,
        sanitized.get("bare_facts"),
        sanitized.get("guarded_facts"),
        bare_value,
        guarded_value,
    )
    passed = bool(
        evaluated
        and result_passed
        and check_value is True
        and facts_replay
        and degradation_within_limit(
            degradation=degradation,
            degradation_limit=degradation_limit,
        )
    )
    result_checks = dict(result.checks)
    result_checks["arm_facts_replay"] = facts_replay
    sanitized.update(
        {
            "metric_kind": metric_kind,
            "direction": direction,
            "bare_value": bare_value,
            "guarded_value": guarded_value,
            "degradation_basis": degradation_basis,
            "degradation": degradation,
            "degradation_limit": degradation_limit,
            "display_value": display_value,
            "display_unit": display_unit,
            "diagnostics": diagnostics,
            "checks": result_checks,
            "evaluated": evaluated and facts_replay,
            "passed": passed,
        }
    )
    return sanitized, passed


def _prepare_retained_metric_impact_section(
    payload: dict[str, Any],
    sanitized: dict[str, Any],
    *,
    degradation_limit: float,
    is_skipped: bool,
) -> tuple[dict[str, Any], bool]:
    metric_kind = payload.get("metric_kind")
    direction = payload.get("direction")
    bare_value = _coerce_non_bool_float(payload.get("bare_value"))
    guarded_value = _coerce_non_bool_float(payload.get("guarded_value"))
    degradation_basis = payload.get("degradation_basis")
    degradation = _coerce_non_bool_float(payload.get("degradation"))
    display_value = _coerce_non_bool_float(payload.get("display_value"))
    display_unit = payload.get("display_unit")
    recomputed = compute_guard_metric_impact(metric_kind, bare_value, guarded_value)
    bare_report = payload.get("bare_report")
    if isinstance(metric_kind, str) and isinstance(bare_report, Mapping):
        bare_envelope = build_guard_metric_bare_report(bare_report, metric_kind)
        if bare_envelope is not None:
            sanitized["bare_report"] = bare_envelope

    for field, value in (
        ("metric_kind", metric_kind),
        ("direction", direction),
        ("degradation_basis", degradation_basis),
        ("display_unit", display_unit),
    ):
        if isinstance(value, str):
            sanitized[field] = value
    for field, value in (
        ("bare_value", bare_value),
        ("guarded_value", guarded_value),
        ("degradation", degradation),
        ("display_value", display_value),
    ):
        if value is not None:
            sanitized[field] = value
    sanitized["degradation_limit"] = degradation_limit
    diagnostics = _coerce_diagnostics(payload.get("diagnostics"))
    sanitized["diagnostics"] = diagnostics
    checks = payload.get("checks")
    sanitized["checks"] = dict(checks) if isinstance(checks, dict) else {}

    measurements_valid = bool(
        recomputed is not None
        and arm_facts_match_measurements(
            metric_kind,
            sanitized.get("bare_facts"),
            sanitized.get("guarded_facts"),
            bare_value,
            guarded_value,
        )
        and direction == recomputed.direction
        and degradation_basis == recomputed.degradation_basis
        and degradation is not None
        and display_value is not None
        and display_unit == recomputed.display_unit
        and math.isclose(
            degradation,
            recomputed.degradation,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and math.isclose(
            display_value,
            recomputed.display_value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )
    has_error = any(
        str(item.get("severity", "")).lower() == "error" for item in diagnostics
    )
    has_failed_check = any(value is not True for value in sanitized["checks"].values())
    if measurements_valid:
        assert degradation is not None
        if is_skipped:
            sanitized["evaluated"] = False
            sanitized["passed"] = False
            return sanitized, False
        passed = (
            not has_error
            and not has_failed_check
            and degradation_within_limit(
                degradation=degradation,
                degradation_limit=degradation_limit,
            )
        )
        sanitized["evaluated"] = True
        sanitized["passed"] = passed
        return sanitized, passed
    if not diagnostics:
        sanitized["diagnostics"] = [
            {
                "kind": "guard_metric_impact_unavailable",
                "severity": "warning",
                "message": "Guard metric impact measurements unavailable",
                "details": {},
            }
        ]
    sanitized["evaluated"] = False
    sanitized["passed"] = False
    return sanitized, False


def prepare_guard_metric_impact_section(
    raw: Any,
    *,
    validate_guard_metric_impact_fn: ValidateGuardMetricImpactFn | None = None,
) -> tuple[dict[str, Any], bool]:
    """Normalize guard metric impact payload and determine whether it passes the gate."""

    if validate_guard_metric_impact_fn is None:
        from .validate import validate_guard_metric_impact

        validate_guard_metric_impact_impl: ValidateGuardMetricImpactFn = (
            validate_guard_metric_impact
        )
    else:
        validate_guard_metric_impact_impl = validate_guard_metric_impact_fn

    if not isinstance(raw, dict) or not raw:
        return (
            {
                "degradation_limit": 0.01,
                "source": "report",
                "evaluated": False,
                "passed": False,
                "diagnostics": [
                    {
                        "kind": "guard_metric_impact_unavailable",
                        "severity": "warning",
                        "message": "Guard metric impact evidence is missing",
                        "details": {},
                    }
                ],
                "checks": {},
            },
            False,
        )

    payload = copy.deepcopy(raw)

    stale_fields = {
        "bare_ppl",
        "guarded_ppl",
        "impact_ratio",
        "impact_percent",
        "impact_threshold",
        "metric_direction",
        "impact_basis",
        "impact_value",
    }
    supplied_stale_fields = sorted(stale_fields & set(payload))
    if supplied_stale_fields:
        return (
            {
                "degradation_limit": 0.01,
                "source": str(payload.get("source", "report")),
                "evaluated": False,
                "passed": False,
                "diagnostics": [
                    {
                        "kind": "guard_metric_impact_stale_contract",
                        "severity": "error",
                        "message": (
                            "Guard metric impact payload uses removed fields: "
                            + ", ".join(supplied_stale_fields)
                        ),
                        "details": {},
                    }
                ],
                "checks": {},
            },
            False,
        )

    unsupported_fields = sorted(set(payload) - _GUARD_METRIC_IMPACT_INPUT_FIELDS)
    if unsupported_fields:
        return (
            {
                "degradation_limit": 0.01,
                "source": str(payload.get("source", "report")),
                "evaluated": False,
                "passed": False,
                "diagnostics": [
                    {
                        "kind": "guard_metric_impact_unsupported_fields",
                        "severity": "error",
                        "message": (
                            "Guard metric impact payload contains unsupported fields: "
                            + ", ".join(unsupported_fields)
                        ),
                        "details": {},
                    }
                ],
                "checks": {},
            },
            False,
        )

    def _coerce_float(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        coerced = float(value)
        return coerced if math.isfinite(coerced) else None

    degradation_limit_supplied = "degradation_limit" in payload
    degradation_limit_candidate = _coerce_float(payload.get("degradation_limit", 0.01))
    degradation_limit_valid = (
        degradation_limit_candidate is not None and degradation_limit_candidate >= 0.0
    )
    degradation_limit: float
    if degradation_limit_valid:
        assert degradation_limit_candidate is not None
        degradation_limit = degradation_limit_candidate
    else:
        degradation_limit = 0.01

    supplied_source = payload.get("source", "report")
    if not isinstance(supplied_source, str) or not supplied_source.strip():
        return (
            {
                "degradation_limit": degradation_limit,
                "source": "report",
                "evaluated": False,
                "passed": False,
                "diagnostics": [
                    {
                        "kind": "guard_metric_impact_invalid_source",
                        "severity": "error",
                        "message": "Guard metric impact source must be a non-empty string",
                        "details": {},
                    }
                ],
                "checks": {},
            },
            False,
        )
    sanitized: dict[str, Any] = {
        "degradation_limit": degradation_limit,
        "source": supplied_source.strip(),
    }
    for facts_field in ("bare_facts", "guarded_facts"):
        facts = payload.get(facts_field)
        if isinstance(facts, Mapping):
            sanitized[facts_field] = dict(facts)
    if degradation_limit_supplied and not degradation_limit_valid:
        sanitized.update(
            {
                "evaluated": False,
                "passed": False,
                "diagnostics": [
                    {
                        "kind": "guard_metric_impact_invalid_degradation_limit",
                        "severity": "error",
                        "message": "Guard metric degradation_limit must be a finite non-negative number",
                        "details": {},
                    }
                ],
                "checks": {},
            }
        )
        return sanitized, False
    try:
        mode = payload.get("mode")
        if mode is None:
            mode = payload.get("guard_metric_impact_mode")
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

    is_skipped = sanitized.get("skipped") is True

    structured = _prepare_structured_metric_impact_section(
        payload,
        sanitized,
        validate_guard_metric_impact_fn=validate_guard_metric_impact_impl,
        degradation_limit=degradation_limit,
        is_skipped=is_skipped,
    )
    if structured is not None:
        return structured
    return _prepare_retained_metric_impact_section(
        payload,
        sanitized,
        degradation_limit=degradation_limit,
        is_skipped=is_skipped,
    )


def compute_guard_metric_impact_from_guard(
    raw_guard: Any,
    pm_kind_hint: str | None = None,
    *,
    compute_primary_metric_from_report_fn: ComputePrimaryMetricFn | None = None,
    get_metric_fn: GetMetricFn | None = None,
) -> dict[str, Any] | None:
    """Compute PM-aware guard metric impact from guard context when possible."""

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
        # Resolve through the registry to reject unknown or mismatched metric kinds.
        try:
            get_metric_impl(kind)
        except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
            return None
        measurement = compute_guard_metric_impact(kind, b_point, g_point)
        return measurement.to_metrics() if measurement is not None else None
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        return None
