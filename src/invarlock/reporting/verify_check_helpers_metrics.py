from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from invarlock.core.metric_kind_contract import (
    MetricKindContractError,
    is_ppl_metric_kind,
    normalize_metric_kind,
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
from invarlock.reporting.report_validation import compute_validation_flags

_VERIFY_PARSE_EXCEPTIONS = (
    AttributeError,
    json.JSONDecodeError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


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

    expected = (math.exp(ci_bounds[0]), math.exp(ci_bounds[1]))
    observed = display_bounds
    for obs, exp_val in zip(observed, expected, strict=False):
        tolerance = 5e-4 * max(1.0, abs(exp_val))
        if abs(obs - exp_val) > tolerance:
            errors.append(
                "primary_metric.display_ci mismatch: bounds do not match exp(ci)."
            )
            break
    return errors


def _validate_primary_metric(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
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
    ratio_vs_baseline = pm.get("ratio_vs_baseline")
    final = pm.get("final")
    pm_invalid = _declares_invalid_primary_metric(pm)

    if is_ppl_metric_kind(kind):
        baseline_ref = report.get("baseline_ref", {}) or {}
        baseline_pm = (
            baseline_ref.get("primary_metric")
            if isinstance(baseline_ref, dict)
            else None
        )
        baseline_final = None
        if isinstance(baseline_pm, dict):
            bv = baseline_pm.get("final")
            baseline_final_value = _coerce_float(bv)
            if baseline_final_value is not None:
                baseline_final = baseline_final_value
        final_value = _coerce_float(final)
        baseline_final_value = _coerce_float(baseline_final)
        if final_value is not None and baseline_final_value is not None:
            if baseline_final_value <= 0.0:
                errors.append(
                    f"Baseline final must be > 0.0 to compute ratio (found {baseline_final})."
                )
            else:
                expected_ratio = final_value / baseline_final_value
                ratio_value = _coerce_float(ratio_vs_baseline)
                if ratio_value is None:
                    errors.append(
                        "report is missing a finite primary_metric.ratio_vs_baseline value."
                    )
                elif not math.isclose(
                    ratio_value,
                    expected_ratio,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    errors.append(
                        "Primary metric ratio mismatch: "
                        f"recorded={ratio_value:.12f}, expected={expected_ratio:.12f}"
                    )
        else:
            if (isinstance(final, (int, float)) and not _is_finite_number(final)) and (
                not pm_invalid
            ):
                errors.append(
                    "Primary metric final is non-finite but primary_metric.invalid is not set."
                )
    else:
        if pm_invalid:
            return errors
        if ratio_vs_baseline is None or not isinstance(ratio_vs_baseline, (int, float)):
            errors.append(
                "report missing primary_metric.ratio_vs_baseline for non-ppl metric."
            )
        elif not _is_finite_number(ratio_vs_baseline):
            errors.append(
                "report is missing a finite primary_metric.ratio_vs_baseline value."
            )

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
        guard_overhead=report.get("guard_overhead")
        if isinstance(report.get("guard_overhead"), dict)
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
    if bool(flags.get("primary_metric_acceptable", True)):
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

    required_true = (
        "primary_metric_acceptable",
        "preview_final_drift_acceptable",
        "invariants_pass",
        "spectral_stable",
        "rmt_stable",
    )
    for key in required_true:
        if validation.get(key) is not True:
            errors.append(
                f"Release verification requires validation.{key} == true "
                f"(found {validation.get(key)!r})."
            )

    if (
        "primary_metric_tail" in report
        or "primary_metric_tail_acceptable" in validation
    ) and validation.get("primary_metric_tail_acceptable") is not True:
        errors.append(
            "Release verification requires validation.primary_metric_tail_acceptable == "
            f"true (found {validation.get('primary_metric_tail_acceptable')!r})."
        )

    guard_overhead = report.get("guard_overhead")
    skipped = isinstance(guard_overhead, dict) and (
        bool(guard_overhead.get("skipped", False))
        or str(guard_overhead.get("mode", "")).strip().lower() == "skipped"
    )
    if (
        isinstance(guard_overhead, dict)
        and guard_overhead
        and not skipped
        and validation.get("guard_overhead_acceptable") is not True
    ):
        errors.append(
            "Release verification requires validation.guard_overhead_acceptable == "
            f"true (found {validation.get('guard_overhead_acceptable')!r})."
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
