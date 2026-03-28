from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


def _coerce_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _coerce_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _load_evaluation_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_report_schema_strict(
    report: dict[str, Any],
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

    try:
        schema_lib.validate(instance=report, schema=report_json_schema)
    except Exception:
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

    stats = report.get("dataset", {}).get("windows", {}).get("stats", {}) or {}
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
    if not (
        isinstance(baseline_final, (int, float))
        and math.isfinite(float(baseline_final))
    ):
        return errors

    def _finite_bounds(bounds: Any) -> bool:
        return (
            isinstance(bounds, (tuple, list))
            and len(bounds) == 2
            and all(
                isinstance(v, (int, float)) and math.isfinite(float(v)) for v in bounds
            )
        )

    prof = (profile or "").strip().lower() if isinstance(profile, str | None) else "dev"
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

    if not (_finite_bounds(ci) and _finite_bounds(display_ci)):
        return errors

    expected = (math.exp(float(ci[0])), math.exp(float(ci[1])))
    observed = (float(display_ci[0]), float(display_ci[1]))
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
        return isinstance(value, (int, float)) and math.isfinite(float(value))

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

    kind = str(pm.get("kind", "")).lower()
    ratio_vs_baseline = pm.get("ratio_vs_baseline")
    final = pm.get("final")
    pm_invalid = _declares_invalid_primary_metric(pm)

    if kind.startswith("ppl"):
        baseline_ref = report.get("baseline_ref", {}) or {}
        baseline_pm = (
            baseline_ref.get("primary_metric")
            if isinstance(baseline_ref, dict)
            else None
        )
        baseline_final = None
        if isinstance(baseline_pm, dict):
            bv = baseline_pm.get("final")
            if isinstance(bv, (int, float)):
                baseline_final = float(bv)
        if _is_finite_number(final) and _is_finite_number(baseline_final):
            if float(baseline_final) <= 0.0:
                errors.append(
                    f"Baseline final must be > 0.0 to compute ratio (found {baseline_final})."
                )
            else:
                expected_ratio = float(final) / float(baseline_final)
                if not _is_finite_number(ratio_vs_baseline):
                    errors.append(
                        "report is missing a finite primary_metric.ratio_vs_baseline value."
                    )
                elif not math.isclose(
                    float(ratio_vs_baseline),
                    expected_ratio,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    errors.append(
                        "Primary metric ratio mismatch: "
                        f"recorded={float(ratio_vs_baseline):.12f}, expected={expected_ratio:.12f}"
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


def _validate_pairing(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    stats = report.get("dataset", {}).get("windows", {}).get("stats", {})

    match_fraction = stats.get("window_match_fraction")
    overlap_fraction = stats.get("window_overlap_fraction")
    pairing_reason = stats.get("window_pairing_reason")
    paired_windows = _coerce_int(stats.get("paired_windows"))

    if pairing_reason is not None:
        errors.append(
            "window_pairing_reason must be null/None for paired reports "
            f"(found {pairing_reason!r})."
        )
    if paired_windows is None:
        errors.append("report missing paired_windows metric.")
    elif paired_windows == 0:
        errors.append("paired_windows must be > 0 for paired reports (found 0).")

    if match_fraction is None:
        errors.append("report missing window_match_fraction metric.")
    elif match_fraction < 0.999999:
        errors.append(
            f"window_match_fraction must be 1.0 for paired runs (found {match_fraction:.6f})."
        )

    if overlap_fraction is None:
        errors.append("report missing window_overlap_fraction metric.")
    elif overlap_fraction > 1e-9:
        errors.append(
            f"window_overlap_fraction must be 0.0 (found {overlap_fraction:.6f})."
        )

    return errors


def _validate_counts(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    dataset = report.get("dataset", {})
    dataset_windows = dataset.get("windows", {})
    expected_preview = dataset_windows.get("preview")
    expected_final = dataset_windows.get("final")

    stats = dataset_windows.get("stats", {})
    coverage = stats.get("coverage", {})

    preview_used = coverage.get("preview", {}).get("used") if coverage else None
    final_used = coverage.get("final", {}).get("used") if coverage else None
    paired_windows = stats.get("paired_windows")

    if expected_preview is not None:
        if preview_used is None:
            errors.append("report missing coverage.preview.used for preview windows.")
        elif int(preview_used) != int(expected_preview):
            errors.append(
                f"Preview window count mismatch: expected {expected_preview}, observed {preview_used}."
            )

    if expected_final is not None:
        if final_used is None:
            errors.append("report missing coverage.final.used for final windows.")
        elif int(final_used) != int(expected_final):
            errors.append(
                f"Final window count mismatch: expected {expected_final}, observed {final_used}."
            )

    if (
        paired_windows is not None
        and expected_preview is not None
        and int(paired_windows) != int(expected_preview)
    ):
        errors.append(
            f"Paired window count mismatch: expected {expected_preview}, observed {paired_windows}."
        )

    return errors


def _validate_drift_band(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if resolve_tiny_relax_from_report(report):
        return errors
    pm = report.get("primary_metric", {}) or {}
    if not isinstance(pm, dict) or not pm:
        errors.append("report missing primary_metric block.")
        return errors
    if bool(pm.get("invalid")):
        return errors
    drift_ratio = None
    try:
        prev = pm.get("preview")
        fin = pm.get("final")
        if (
            isinstance(prev, (int, float))
            and isinstance(fin, (int, float))
            and math.isfinite(float(prev))
            and math.isfinite(float(fin))
            and prev > 0
        ):
            drift_ratio = float(fin) / float(prev)
    except Exception:
        drift_ratio = None

    if not isinstance(drift_ratio, (int, float)):
        errors.append("report missing preview/final to compute drift ratio.")
        return errors

    drift_min = 0.95
    drift_max = 1.05
    band = pm.get("drift_band")
    try:
        if isinstance(band, dict):
            lo = band.get("min")
            hi = band.get("max")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                lo_f = float(lo)
                hi_f = float(hi)
                if math.isfinite(lo_f) and math.isfinite(hi_f) and 0 < lo_f < hi_f:
                    drift_min = lo_f
                    drift_max = hi_f
        elif isinstance(band, (list, tuple)) and len(band) == 2:
            lo_raw, hi_raw = band[0], band[1]
            if isinstance(lo_raw, (int, float)) and isinstance(hi_raw, (int, float)):
                lo_f = float(lo_raw)
                hi_f = float(hi_raw)
                if math.isfinite(lo_f) and math.isfinite(hi_f) and 0 < lo_f < hi_f:
                    drift_min = lo_f
                    drift_max = hi_f
    except Exception:
        pass

    if not drift_min <= float(drift_ratio) <= drift_max:
        errors.append(
            f"Preview→final drift ratio out of band ({drift_min:.2f}–{drift_max:.2f}): observed {drift_ratio:.6f}."
        )

    return errors


def _validate_tokenizer_hash(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    meta = report.get("meta", {}) or {}
    dataset = report.get("dataset", {}) or {}
    edited_hash = None
    try:
        edited_hash = meta.get("tokenizer_hash") or (
            (dataset.get("tokenizer") or {}).get("hash")
            if isinstance(dataset.get("tokenizer"), dict)
            else None
        )
    except Exception:
        edited_hash = None

    baseline_ref = report.get("baseline_ref", {}) or {}
    baseline_hash = baseline_ref.get("tokenizer_hash")

    if isinstance(edited_hash, str) and isinstance(baseline_hash, str):
        if edited_hash and baseline_hash and edited_hash != baseline_hash:
            errors.append("Tokenizer hash mismatch between baseline and edited runs.")
    return errors


def _resolve_path(payload: Any, path: str) -> Any:
    current = payload
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            return None
    return current


def _measurement_contract_digest(contract: Any) -> str | None:
    if not isinstance(contract, dict) or not contract:
        return None
    try:
        canonical = json.dumps(contract, sort_keys=True, default=str)
    except Exception:
        return None
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _validate_measurement_contracts(
    report: dict[str, Any], *, profile: str
) -> list[str]:
    errors: list[str] = []
    prof = (profile or "").strip().lower()
    resolved_policy = report.get("resolved_policy") or {}

    for guard_key in ("spectral", "rmt"):
        block = report.get(guard_key) or {}
        if not isinstance(block, dict):
            continue
        evaluated = bool(block.get("evaluated", True))
        if not evaluated:
            continue

        mc = block.get("measurement_contract")
        mc_hash = _measurement_contract_digest(mc)
        expected_hash = block.get("measurement_contract_hash")
        if not isinstance(mc, dict) or not mc:
            errors.append(f"report missing {guard_key}.measurement_contract.")
        elif isinstance(expected_hash, str) and expected_hash:
            if mc_hash and mc_hash != expected_hash:
                errors.append(
                    f"{guard_key}.measurement_contract_hash mismatch: expected={expected_hash}, computed={mc_hash}."
                )
        else:
            errors.append(f"report missing {guard_key}.measurement_contract_hash.")

        rp_guard = (
            resolved_policy.get(guard_key)
            if isinstance(resolved_policy, dict)
            else None
        )
        rp_mc = (
            rp_guard.get("measurement_contract") if isinstance(rp_guard, dict) else None
        )
        rp_hash = _measurement_contract_digest(rp_mc)
        if not isinstance(rp_mc, dict) or not rp_mc:
            errors.append(
                f"report missing resolved_policy.{guard_key}.measurement_contract."
            )
        elif mc_hash and rp_hash and mc_hash != rp_hash:
            errors.append(
                f"{guard_key} measurement_contract differs between analysis and resolved_policy "
                f"(analysis={mc_hash}, resolved_policy={rp_hash})."
            )

        if prof in {"ci", "release"}:
            match = block.get("measurement_contract_match")
            if match is not True:
                errors.append(
                    f"{guard_key} measurement contract must match baseline for {prof} profile."
                )

    return errors


def _apply_profile_lints(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    meta = report.get("meta", {})
    profile = meta.get("model_profile") if isinstance(meta, dict) else None
    if not isinstance(profile, dict):
        return errors

    lints = profile.get("cert_lints", [])
    if not isinstance(lints, list):
        return errors

    for lint in lints:
        if not isinstance(lint, dict):
            continue
        lint_type = str(lint.get("type", "")).lower()
        path = lint.get("path")
        expected = lint.get("value")
        message = lint.get("message") or "Model profile lint failed."
        actual = _resolve_path(report, path) if isinstance(path, str) else None

        if lint_type == "equals":
            if actual != expected:
                errors.append(
                    f"{message} Expected {path} == {expected!r}, observed {actual!r}."
                )
        elif lint_type == "gte":
            try:
                actual_val = float(actual)
                expected_val = float(expected)
            except (TypeError, ValueError):
                errors.append(
                    f"{message} Expected numeric comparison for {path}, observed {actual!r}."
                )
            else:
                if actual_val < expected_val:
                    errors.append(
                        f"{message} Expected {path} ≥ {expected_val}, observed {actual_val}."
                    )
        elif lint_type == "lte":
            try:
                actual_val = float(actual)
                expected_val = float(expected)
            except (TypeError, ValueError):
                errors.append(
                    f"{message} Expected numeric comparison for {path}, observed {actual!r}."
                )
            else:
                if actual_val > expected_val:
                    errors.append(
                        f"{message} Expected {path} ≤ {expected_val}, observed {actual_val}."
                    )

    return errors


def _validate_evaluation_report_payload(
    path: Path,
    *,
    profile: str | None = None,
    load_evaluation_report_fn: Callable[
        [Path], dict[str, Any]
    ] = _load_evaluation_report,
    validate_report_fn: Callable[[dict[str, Any]], bool] = validate_report,
    validate_report_schema_strict_fn: Callable[
        [dict[str, Any]], bool
    ] = _validate_report_schema_strict,
    validate_primary_metric_fn: Callable[
        [dict[str, Any]], list[str]
    ] = _validate_primary_metric,
    validate_pairing_fn: Callable[[dict[str, Any]], list[str]] = _validate_pairing,
    validate_counts_fn: Callable[[dict[str, Any]], list[str]] = _validate_counts,
    validate_logspace_ci_identity_fn: Callable[
        ..., list[str]
    ] = _validate_logspace_ci_identity,
    validate_drift_band_fn: Callable[
        [dict[str, Any]], list[str]
    ] = _validate_drift_band,
    validate_primary_metric_policy_fn: Callable[
        ..., list[str]
    ] = _validate_primary_metric_policy,
    apply_profile_lints_fn: Callable[
        [dict[str, Any]], list[str]
    ] = _apply_profile_lints,
    validate_tokenizer_hash_fn: Callable[
        [dict[str, Any]], list[str]
    ] = _validate_tokenizer_hash,
    validate_measurement_contracts_fn: Callable[
        ..., list[str]
    ] = _validate_measurement_contracts,
) -> list[str]:
    errors: list[str] = []
    report = load_evaluation_report_fn(path)
    try:
        prof = (
            (profile or "").strip().lower()
            if isinstance(profile, str | None)
            else "dev"
        )
    except Exception:
        prof = "dev"

    if prof in {"ci", "release"} and not validate_report_schema_strict_fn(report):
        errors.append("report schema validation failed.")
        return errors

    if not validate_report_fn(report):
        errors.append("report schema validation failed.")
        return errors

    errors.extend(validate_primary_metric_fn(report))
    errors.extend(validate_pairing_fn(report))
    errors.extend(validate_counts_fn(report))
    errors.extend(validate_logspace_ci_identity_fn(report, profile=profile))
    if prof in {"ci", "release"}:
        errors.extend(validate_drift_band_fn(report))
        errors.extend(validate_primary_metric_policy_fn(report, profile=prof))
    errors.extend(apply_profile_lints_fn(report))
    errors.extend(validate_tokenizer_hash_fn(report))
    if prof in {"ci", "release"}:
        errors.extend(validate_measurement_contracts_fn(report, profile=prof))

    if prof == "release":
        errors.extend(_validate_release_gate_outcomes(report))
        go = report.get("guard_overhead")
        if not isinstance(go, dict) or not go:
            errors.append(
                "Release verification requires guard_overhead (missing). "
                "Set context.run.skip_overhead_check=true in the run config to explicitly skip during evaluation."
            )
        else:
            skipped = bool(go.get("skipped", False)) or (
                str(go.get("mode", "")).strip().lower() == "skipped"
            )
            if not skipped:
                evaluated = go.get("evaluated")
                if evaluated is not True:
                    errors.append(
                        "Release verification requires evaluated guard_overhead (not evaluated). "
                        "Set context.run.skip_overhead_check=true in the run config to explicitly skip during evaluation."
                    )
                ratio = go.get("overhead_ratio")
                if ratio is None:
                    errors.append(
                        "Release verification requires guard_overhead.overhead_ratio (missing)."
                    )
    return errors


__all__ = [
    "_apply_profile_lints",
    "_coerce_float",
    "_coerce_int",
    "_load_evaluation_report",
    "_measurement_contract_digest",
    "_recompute_validation_flags",
    "_resolve_path",
    "_validate_counts",
    "_validate_drift_band",
    "_validate_evaluation_report_payload",
    "_validate_logspace_ci_identity",
    "_validate_measurement_contracts",
    "_validate_pairing",
    "_validate_primary_metric",
    "_validate_primary_metric_policy",
    "_validate_release_gate_outcomes",
    "_validate_report_schema_strict",
    "_validate_tokenizer_hash",
    "compute_validation_flags",
    "resolve_tiny_relax_from_report",
    "validate_report",
    "_report_schema",
]
