from __future__ import annotations

import json
import math
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from invarlock.core.metric_kind_contract import (
    MetricKindContractError,
    is_ppl_metric_kind,
    normalize_metric_kind,
)
from invarlock.guards.authority import (
    guard_is_enforced,
    resolved_guard_authority,
)
from invarlock.primary_metric_tail import (
    PrimaryMetricTailContractError,
    require_primary_metric_tail,
)
from invarlock.reporting import report_schema as _report_schema
from invarlock.reporting.report_policy import (
    resolve_pm_acceptance_range_from_report,
    resolve_pm_drift_band_from_report,
    resolve_tiny_relax_from_report,
)
from invarlock.reporting.report_schema import (
    REPORT_JSON_SCHEMA,
    REPORT_SCHEMA_VERSION,
    validate_report,
)
from invarlock.reporting.validation.report import compute_validation_flags
from invarlock.reporting.verify_system_overhead import validate_system_overhead

_VERIFY_PARSE_EXCEPTIONS = (
    AttributeError,
    json.JSONDecodeError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


@lru_cache(maxsize=1)
def _compiled_canonical_report_validator(schema_runtime_id: int) -> Any | None:
    del schema_runtime_id
    return _report_schema._compile_jsonschema_validator(REPORT_JSON_SCHEMA)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _load_evaluation_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("evaluation report must decode to a JSON object")
    return payload


def _validate_report_schema_strict(
    report: Any,
    *,
    schema_version: str = REPORT_SCHEMA_VERSION,
    report_json_schema: dict[str, Any] = REPORT_JSON_SCHEMA,
    report_schema_module: Any = _report_schema,
) -> bool:
    if not isinstance(report, dict):
        return False
    if report.get("schema_version") != schema_version:
        return False

    schema_lib = getattr(report_schema_module, "jsonschema", None)
    if schema_lib is None:
        return False
    schema_failures = getattr(report_schema_module, "_JSONSCHEMA_FAILURES", ())
    schema_failures = tuple(
        exc
        for exc in schema_failures
        if isinstance(exc, type) and issubclass(exc, BaseException)
    )
    schema_validation_exceptions = (
        TypeError,
        ValueError,
        KeyError,
        RuntimeError,
    ) + schema_failures

    try:
        validator = (
            _compiled_canonical_report_validator(id(schema_lib))
            if report_json_schema is REPORT_JSON_SCHEMA
            else None
        )
        if validator is not None:
            validator.validate(report)
        else:
            schema_lib.validate(instance=report, schema=report_json_schema)
    except schema_validation_exceptions:
        return False
    return True


def _validate_logspace_ci_identity(
    report: dict[str, Any], *, profile: str | None
) -> list[str]:
    errors: list[str] = []
    pm = report.get("primary_metric", {}) or {}
    if not isinstance(pm, dict):
        return errors

    kind = str(pm.get("kind", "")).lower()
    if not kind.startswith("ppl"):
        return errors

    dataset = report.get("dataset", {})
    dataset_windows = dataset.get("windows", {}) if isinstance(dataset, dict) else {}
    stats = (
        dataset_windows.get("stats", {}) if isinstance(dataset_windows, dict) else {}
    )
    if not isinstance(stats, dict):
        return errors

    pairing_reason = stats.get("window_pairing_reason")
    paired_windows = _coerce_int(stats.get("paired_windows"))
    match_fraction = _coerce_float(stats.get("window_match_fraction"))
    overlap_fraction = _coerce_float(stats.get("window_overlap_fraction"))
    paired = bool(
        pairing_reason is None
        and paired_windows is not None
        and paired_windows > 0
        and isinstance(match_fraction, float)
        and match_fraction >= 0.999999
        and isinstance(overlap_fraction, float)
        and overlap_fraction <= 1e-9
    )
    if not paired:
        return errors

    baseline_ref = report.get("baseline_ref", {}) or {}
    baseline_pm = (
        baseline_ref.get("primary_metric") if isinstance(baseline_ref, dict) else None
    )
    baseline_final = baseline_pm.get("final") if isinstance(baseline_pm, dict) else None
    if not (_coerce_float(baseline_final) is not None):
        return errors

    def _finite_bounds(bounds: Any) -> bool:
        return (
            isinstance(bounds, (tuple, list))
            and len(bounds) == 2
            and all(_coerce_float(v) is not None for v in bounds)
        )

    def _coerce_bounds(bounds: Any) -> tuple[float, float] | None:
        if not _finite_bounds(bounds):
            return None
        return float(bounds[0]), float(bounds[1])

    prof = (profile or "").strip().lower() if isinstance(profile, str) else "dev"
    ci = pm.get("ci")
    display_ci = pm.get("display_ci")

    if prof in {"ci", "release"}:
        if not _finite_bounds(ci):
            errors.append(
                "primary_metric.ci missing for ppl-like metric under paired baseline in CI/Release."
            )
        if not _finite_bounds(display_ci):
            errors.append(
                "primary_metric.display_ci missing for ppl-like metric under paired baseline in CI/Release."
            )

    ci_bounds = _coerce_bounds(ci)
    display_bounds = _coerce_bounds(display_ci)
    if ci_bounds is None or display_bounds is None:
        return errors

    try:
        expected = (math.exp(ci_bounds[0]), math.exp(ci_bounds[1]))
    except OverflowError:
        errors.append(
            "primary_metric.ci exponentiation overflows finite display range."
        )
        return errors
    if any(not math.isfinite(value) or value <= 0.0 for value in expected):
        errors.append(
            "primary_metric.ci exponentiation is outside the finite positive "
            "display range."
        )
        return errors
    observed = display_bounds
    if any(value <= 0.0 for value in observed):
        errors.append("primary_metric.display_ci bounds must be positive for PPL.")
        return errors
    for obs, exp_val in zip(observed, expected, strict=False):
        tolerance = 5e-4 * max(1.0, abs(exp_val))
        if abs(obs - exp_val) > tolerance:
            errors.append(
                "primary_metric.display_ci mismatch: bounds do not match exp(ci)."
            )
            break
    return errors


def _validate_ppl_metric(
    report: dict[str, Any], pm: dict[str, Any], kind: str
) -> list[str]:
    errors: list[str] = []
    preview_value = _coerce_float(pm.get("preview"))
    ratio_value = _coerce_float(pm.get("ratio_vs_baseline"))
    baseline_ref = report.get("baseline_ref", {}) or {}
    baseline_pm = (
        baseline_ref.get("primary_metric") if isinstance(baseline_ref, dict) else None
    )
    try:
        baseline_kind = normalize_metric_kind(
            baseline_pm.get("kind") if isinstance(baseline_pm, dict) else None
        )
    except (MetricKindContractError, ValueError):
        baseline_kind = None
    if baseline_kind != kind:
        errors.append("PPL verification requires a same-kind baseline primary metric.")
    baseline_final = (
        _coerce_float(baseline_pm.get("final"))
        if isinstance(baseline_pm, dict)
        else None
    )
    final_value = _coerce_float(pm.get("final"))
    if preview_value is None or preview_value < 1.0:
        errors.append("PPL primary_metric.preview must be finite and at least 1.0.")
    if final_value is None or final_value < 1.0:
        errors.append("PPL primary_metric.final must be finite and at least 1.0.")
    if baseline_final is None:
        errors.append("PPL verification requires a finite baseline final value.")
    elif baseline_final < 1.0:
        errors.append(
            f"PPL baseline final must be at least 1.0 (found {baseline_final})."
        )
    if ratio_value is None or ratio_value <= 0.0:
        errors.append(
            "report is missing a finite positive primary_metric.ratio_vs_baseline value."
        )
    if (
        final_value is not None
        and final_value >= 1.0
        and baseline_final is not None
        and baseline_final >= 1.0
        and ratio_value is not None
        and ratio_value > 0.0
    ):
        expected_ratio = final_value / baseline_final
        if not math.isclose(ratio_value, expected_ratio, rel_tol=1e-6, abs_tol=1e-6):
            errors.append(
                "Primary metric ratio mismatch: "
                f"recorded={ratio_value:.12f}, expected={expected_ratio:.12f}"
            )
    return errors


def _validate_accuracy_metric(report: dict[str, Any], pm: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if "ratio_vs_baseline" in pm:
        errors.append(
            "primary_metric.ratio_vs_baseline is not allowed for accuracy; "
            "use delta_vs_baseline_pp."
        )
    delta_pp = _coerce_float(pm.get("delta_vs_baseline_pp"))
    preview_value = _coerce_float(pm.get("preview"))
    final_value = _coerce_float(pm.get("final"))
    baseline_ref = report.get("baseline_ref")
    baseline_pm = (
        baseline_ref.get("primary_metric") if isinstance(baseline_ref, dict) else None
    )
    try:
        baseline_kind = normalize_metric_kind(
            baseline_pm.get("kind") if isinstance(baseline_pm, dict) else None
        )
    except (MetricKindContractError, ValueError):
        baseline_kind = None
    if baseline_kind != "accuracy":
        errors.append(
            "Accuracy verification requires an accuracy baseline primary metric."
        )
    baseline_final = (
        _coerce_float(baseline_pm.get("final"))
        if isinstance(baseline_pm, dict)
        else None
    )
    if delta_pp is None:
        errors.append(
            "primary_metric.delta_vs_baseline_pp must be finite for accuracy."
        )
    if final_value is None or not 0.0 <= final_value <= 1.0:
        errors.append("Accuracy primary_metric.final must be finite in [0, 1].")
    if preview_value is None or not 0.0 <= preview_value <= 1.0:
        errors.append("Accuracy primary_metric.preview must be finite in [0, 1].")
    if baseline_final is None or not 0.0 <= baseline_final <= 1.0:
        errors.append("Accuracy verification requires baseline final in [0, 1].")
    if delta_pp is not None and final_value is not None and baseline_final is not None:
        expected_delta_pp = 100.0 * (final_value - baseline_final)
        if not math.isclose(delta_pp, expected_delta_pp, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(
                "Accuracy baseline delta mismatch: "
                f"recorded={delta_pp:.12f} expected={expected_delta_pp:.12f}"
            )
    return errors


def _validate_primary_metric(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_system_overhead(report))
    status = report.get("status")
    if isinstance(status, str) and status.strip().lower() in {
        "failed",
        "error",
        "rollback",
        "cancelled",
    }:
        errors.append(
            f"report status {status!r} is not a successful evaluation outcome."
        )
    flags = report.get("flags")
    rollback_reason = flags.get("rollback_reason") if isinstance(flags, dict) else None
    if isinstance(rollback_reason, str) and rollback_reason.strip():
        errors.append(
            "report records a rollback and cannot be verified as a successful "
            "evaluation outcome."
        )
    pm = report.get("primary_metric", {}) or {}
    if not isinstance(pm, dict) or not pm:
        errors.append("report missing primary_metric block.")
        return errors

    def _is_finite_number(value: Any) -> bool:
        return _coerce_float(value) is not None

    def _declares_invalid_primary_metric(metric: dict[str, Any]) -> bool:
        if bool(metric.get("invalid")):
            return True
        reason = metric.get("degraded_reason")
        if isinstance(reason, str):
            r = reason.strip().lower()
            return r.startswith("non_finite") or r in {
                "primary_metric_invalid",
                "evaluation_error",
            }
        return False

    try:
        kind = normalize_metric_kind(pm.get("kind"))
    except (MetricKindContractError, ValueError):
        errors.append(
            f"report has unsupported primary_metric.kind: {pm.get('kind')!r}."
        )
        return errors
    if kind is None:
        errors.append("report missing primary_metric.kind.")
        return errors
    pm_invalid = _declares_invalid_primary_metric(pm)
    if pm_invalid:
        errors.append("report primary_metric is invalid or degraded.")
        return errors

    if is_ppl_metric_kind(kind):
        errors.extend(_validate_ppl_metric(report, pm, kind))
    elif kind == "accuracy":
        errors.extend(_validate_accuracy_metric(report, pm))

    return errors


def _recompute_validation_flags(
    report: dict[str, Any],
    *,
    compute_validation_flags_fn: Callable[
        ..., dict[str, bool]
    ] = compute_validation_flags,
    resolve_pm_acceptance_range_from_report_fn: Callable[
        [dict[str, Any]], dict[str, float]
    ] = resolve_pm_acceptance_range_from_report,
    resolve_pm_drift_band_from_report_fn: Callable[
        [dict[str, Any]], dict[str, float]
    ] = resolve_pm_drift_band_from_report,
    resolve_tiny_relax_from_report_fn: Callable[[dict[str, Any]], bool]
    | Callable[[dict[str, Any]], Any] = resolve_tiny_relax_from_report,
) -> dict[str, bool]:
    pm = report.get("primary_metric") or {}
    if not isinstance(pm, dict):
        pm = {}

    ppl: dict[str, Any] = {}
    ratio_vs_baseline = _coerce_float(pm.get("ratio_vs_baseline"))
    if ratio_vs_baseline is not None:
        ppl["ratio_vs_baseline"] = ratio_vs_baseline

    preview = _coerce_float(pm.get("preview"))
    final = _coerce_float(pm.get("final"))
    if preview is not None and final is not None and preview > 0.0:
        ppl["preview_final_ratio"] = final / preview

    ppl_metrics: dict[str, Any] = {}
    telemetry = report.get("telemetry")
    if isinstance(telemetry, dict):
        for key in ("preview_total_tokens", "final_total_tokens"):
            value = _coerce_int(telemetry.get(key))
            if value is not None:
                ppl_metrics[key] = value

    dataset_windows = report.get("dataset", {}).get("windows", {})
    stats = (
        dataset_windows.get("stats", {}) if isinstance(dataset_windows, dict) else {}
    )
    if isinstance(stats, dict):
        coverage = stats.get("coverage")
        bootstrap = stats.get("bootstrap")
        bootstrap_metrics = (
            dict(ppl_metrics.get("bootstrap", {}))
            if isinstance(ppl_metrics.get("bootstrap"), dict)
            else {}
        )
        coverage_obj = None
        if isinstance(coverage, dict) and coverage:
            coverage_obj = coverage
        elif isinstance(bootstrap, dict) and isinstance(
            bootstrap.get("coverage"), dict
        ):
            coverage_obj = bootstrap.get("coverage")
        if isinstance(coverage_obj, dict) and coverage_obj:
            bootstrap_metrics["coverage"] = coverage_obj
        if bootstrap_metrics:
            ppl_metrics["bootstrap"] = bootstrap_metrics

    auto = report.get("auto")
    if not isinstance(auto, dict):
        auto = {}
    tier = str(auto.get("tier") or "balanced").strip().lower() or "balanced"
    target_ratio = _coerce_float(auto.get("target_pm_ratio"))
    pm_acceptance_range = resolve_pm_acceptance_range_from_report_fn(report)
    pm_drift_band = resolve_pm_drift_band_from_report_fn(report)
    tiny_relax = resolve_tiny_relax_from_report_fn(report)

    metrics_policy = None
    resolved_policy = report.get("resolved_policy")
    if isinstance(resolved_policy, dict):
        candidate = resolved_policy.get("metrics")
        if isinstance(candidate, dict) and candidate:
            metrics_policy = candidate

    get_tier_policies_fn = None
    if isinstance(metrics_policy, dict):

        def _report_tier_policies() -> dict[str, Any]:
            return {tier: {"metrics": metrics_policy}}

        get_tier_policies_fn = _report_tier_policies

    return compute_validation_flags_fn(
        ppl=ppl,
        spectral=report.get("spectral")
        if isinstance(report.get("spectral"), dict)
        else {},
        rmt=report.get("rmt") if isinstance(report.get("rmt"), dict) else {},
        invariants=report.get("invariants")
        if isinstance(report.get("invariants"), dict)
        else {},
        tier=tier,
        _ppl_metrics=ppl_metrics,
        target_ratio=target_ratio,
        pm_acceptance_range=pm_acceptance_range,
        pm_drift_band=pm_drift_band,
        guard_metric_impact=report.get("guard_metric_impact")
        if isinstance(report.get("guard_metric_impact"), dict)
        else None,
        primary_metric=pm,
        moe=report.get("moe") if isinstance(report.get("moe"), dict) else None,
        pm_tail=report.get("primary_metric_tail")
        if isinstance(report.get("primary_metric_tail"), dict)
        else None,
        tiny_relax=tiny_relax,
        get_tier_policies_fn=get_tier_policies_fn,
    )


def _validate_primary_metric_policy(
    report: dict[str, Any],
    *,
    profile: str | None = None,
    recompute_validation_flags_fn: Callable[
        [dict[str, Any]], dict[str, bool]
    ] = _recompute_validation_flags,
) -> list[str]:
    prof = str(profile or "dev").strip().lower()
    if prof not in {"ci", "release"}:
        return []

    flags = recompute_validation_flags_fn(report)
    if flags.get("primary_metric_acceptable") is True:
        return []

    telemetry = report.get("telemetry")
    total_tokens = None
    if isinstance(telemetry, dict):
        preview_tokens = _coerce_int(telemetry.get("preview_total_tokens"))
        final_tokens = _coerce_int(telemetry.get("final_total_tokens"))
        if preview_tokens is not None and final_tokens is not None:
            total_tokens = preview_tokens + final_tokens

    auto = report.get("auto")
    tier = "balanced"
    if isinstance(auto, dict):
        tier = str(auto.get("tier") or "balanced").strip().lower() or "balanced"

    detail = f"tier={tier}"
    if total_tokens is not None:
        detail += f", total_tokens={total_tokens}"
    return [f"Primary metric policy gate failed ({detail})."]


def _validate_release_gate_outcomes(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    validation = report.get("validation")
    if not isinstance(validation, dict):
        return ["Release verification requires a validation block."]

    authority, authority_errors, _ = resolved_guard_authority(report)
    errors.extend(authority_errors)
    required_true = [
        "primary_metric_acceptable",
        "preview_final_drift_acceptable",
        "invariants_pass",
    ]
    if guard_is_enforced(authority, "spectral"):
        required_true.append("spectral_stable")
    if guard_is_enforced(authority, "rmt"):
        required_true.append("rmt_stable")
    for key in required_true:
        if validation.get(key) is not True:
            errors.append(
                f"Release verification requires validation.{key} == true "
                f"(found {validation.get(key)!r})."
            )

    tail_present = "primary_metric_tail" in report
    tail_flag_present = "primary_metric_tail_acceptable" in validation
    primary_metric = report.get("primary_metric")
    tail_required = False
    if isinstance(primary_metric, dict):
        try:
            tail_required = is_ppl_metric_kind(primary_metric.get("kind"))
        except MetricKindContractError:
            tail_required = False
    if tail_required and not tail_present:
        errors.append(
            "Release verification requires primary_metric_tail evidence for a "
            "perplexity primary metric."
        )
    if tail_present:
        try:
            tail_outcome = require_primary_metric_tail(report["primary_metric_tail"])
        except PrimaryMetricTailContractError as exc:
            errors.append(f"Release verification rejected primary_metric_tail: {exc}.")
        else:
            expected_tail_flag = tail_outcome.acceptable
            if (
                validation.get("primary_metric_tail_acceptable")
                is not expected_tail_flag
            ):
                errors.append(
                    "Release verification requires validation.primary_metric_tail_acceptable "
                    f"to equal the exact tail outcome ({expected_tail_flag!r}; found "
                    f"{validation.get('primary_metric_tail_acceptable')!r})."
                )
            if not expected_tail_flag:
                errors.append(
                    "Release verification rejected the primary metric tail gate."
                )
    elif tail_flag_present:
        errors.append(
            "Release verification rejects validation.primary_metric_tail_acceptable "
            "without primary_metric_tail evidence."
        )

    guard_metric_impact = report.get("guard_metric_impact")
    if (
        isinstance(guard_metric_impact, dict)
        and guard_metric_impact
        and validation.get("guard_metric_impact_acceptable") is not True
    ):
        errors.append(
            "Release verification requires validation.guard_metric_impact_acceptable == "
            f"true (found {validation.get('guard_metric_impact_acceptable')!r})."
        )

    return errors


__all__ = [
    "_VERIFY_PARSE_EXCEPTIONS",
    "_coerce_float",
    "_coerce_int",
    "_load_evaluation_report",
    "_recompute_validation_flags",
    "_validate_logspace_ci_identity",
    "_validate_primary_metric",
    "_validate_primary_metric_policy",
    "_validate_release_gate_outcomes",
    "_validate_report_schema_strict",
    "REPORT_JSON_SCHEMA",
    "REPORT_SCHEMA_VERSION",
    "_report_schema",
    "compute_validation_flags",
    "resolve_pm_acceptance_range_from_report",
    "resolve_pm_drift_band_from_report",
    "resolve_tiny_relax_from_report",
    "validate_report",
]
