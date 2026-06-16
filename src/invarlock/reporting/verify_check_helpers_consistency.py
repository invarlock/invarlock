from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from invarlock.core.metric_kind_contract import (
    MetricKindContractError,
    is_ppl_metric_kind,
    normalize_metric_kind,
)

from .guards_common import _measurement_contract_digest
from .verify_check_helpers_metrics import (
    _VERIFY_PARSE_EXCEPTIONS,
    REPORT_JSON_SCHEMA,
    REPORT_SCHEMA_VERSION,
    _coerce_float,
    _coerce_int,
    _load_evaluation_report,
    _report_schema,
    _validate_logspace_ci_identity,
    _validate_primary_metric,
    _validate_primary_metric_policy,
    _validate_release_gate_outcomes,
    _validate_report_schema_strict,
    resolve_tiny_relax_from_report,
    validate_report,
)


def _validate_pairing(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    stats = report.get("dataset", {}).get("windows", {}).get("stats", {})

    match_fraction = _coerce_float(stats.get("window_match_fraction"))
    overlap_fraction = _coerce_float(stats.get("window_overlap_fraction"))
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
        expected_preview_count = _coerce_int(expected_preview)
        if expected_preview_count is None:
            errors.append("report has invalid dataset.windows.preview count.")
        elif preview_used is None:
            errors.append("report missing coverage.preview.used for preview windows.")
        else:
            preview_used_count = _coerce_int(preview_used)
            if preview_used_count is None:
                errors.append("report has invalid coverage.preview.used value.")
            elif preview_used_count != expected_preview_count:
                errors.append(
                    f"Preview window count mismatch: expected {expected_preview}, observed {preview_used}."
                )

    if expected_final is not None:
        expected_final_count = _coerce_int(expected_final)
        if expected_final_count is None:
            errors.append("report has invalid dataset.windows.final count.")
        elif final_used is None:
            errors.append("report missing coverage.final.used for final windows.")
        else:
            final_used_count = _coerce_int(final_used)
            if final_used_count is None:
                errors.append("report has invalid coverage.final.used value.")
            elif final_used_count != expected_final_count:
                errors.append(
                    f"Final window count mismatch: expected {expected_final}, observed {final_used}."
                )

    expected_preview_count = _coerce_int(expected_preview)
    paired_windows_count = _coerce_int(paired_windows)
    if paired_windows is not None and paired_windows_count is None:
        errors.append("report has invalid paired_windows metric.")
    elif (
        paired_windows_count is not None
        and expected_preview is not None
        and expected_preview_count is not None
        and paired_windows_count != expected_preview_count
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
    try:
        if bool(pm.get("invalid")):
            return errors
        pm_kind = normalize_metric_kind(pm.get("kind"))
    except (MetricKindContractError, RuntimeError, ValueError):
        pm_kind = None

    if pm_kind == "accuracy":
        prev = _coerce_float(pm.get("preview"))
        fin = _coerce_float(pm.get("final"))
        if prev is None or fin is None:
            errors.append("report missing preview/final to compute accuracy drift.")
            return errors

        accuracy_delta_limit = None
        try:
            resolved_policy = report.get("resolved_policy") or {}
            metrics_policy = (
                resolved_policy.get("metrics")
                if isinstance(resolved_policy, dict)
                else {}
            )
            accuracy_policy = (
                metrics_policy.get("accuracy")
                if isinstance(metrics_policy, dict)
                else {}
            )
            if isinstance(accuracy_policy, dict):
                accuracy_delta_limit = _coerce_float(
                    accuracy_policy.get("preview_final_delta_pp_max")
                )
                if accuracy_delta_limit is None:
                    accuracy_delta_limit = _coerce_float(
                        accuracy_policy.get("hysteresis_delta_pp")
                    )
        except _VERIFY_PARSE_EXCEPTIONS:
            accuracy_delta_limit = None
        if accuracy_delta_limit is None:
            accuracy_delta_limit = 0.1
        accuracy_delta_limit = max(0.0, accuracy_delta_limit)
        observed_delta = abs(fin - prev)
        if observed_delta > accuracy_delta_limit:
            errors.append(
                "Preview→final accuracy drift out of band "
                f"(≤ {accuracy_delta_limit:.6f}): observed {observed_delta:.6f}."
            )
        return errors

    if pm_kind is not None and not is_ppl_metric_kind(pm_kind):
        return errors

    drift_ratio = None
    try:
        prev = _coerce_float(pm.get("preview"))
        fin = _coerce_float(pm.get("final"))
        if prev is not None and fin is not None and prev > 0:
            drift_ratio = fin / prev
    except _VERIFY_PARSE_EXCEPTIONS:
        drift_ratio = None

    if drift_ratio is None:
        errors.append("report missing preview/final to compute drift ratio.")
        return errors

    drift_min = 0.95
    drift_max = 1.05
    band = pm.get("drift_band")
    try:
        if isinstance(band, dict):
            lo = _coerce_float(band.get("min"))
            hi = _coerce_float(band.get("max"))
            if lo is not None and hi is not None and 0 < lo < hi:
                drift_min = lo
                drift_max = hi
        elif isinstance(band, (list, tuple)) and len(band) == 2:
            lo_f = _coerce_float(band[0])
            hi_f = _coerce_float(band[1])
            if lo_f is not None and hi_f is not None and 0 < lo_f < hi_f:
                drift_min = lo_f
                drift_max = hi_f
    except _VERIFY_PARSE_EXCEPTIONS:
        pass

    if not drift_min <= drift_ratio <= drift_max:
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
    except _VERIFY_PARSE_EXCEPTIONS:
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


def _collect_provenance_window_ids(node: Any) -> list[Any]:
    if isinstance(node, dict):
        window_ids = node.get("window_ids")
        if isinstance(window_ids, list):
            return list(window_ids)
        collected: list[Any] = []
        for value in node.values():
            collected.extend(_collect_provenance_window_ids(value))
        return collected
    if isinstance(node, list):
        collected = []
        for value in node:
            collected.extend(_collect_provenance_window_ids(value))
        return collected
    return []


def _validate_variance_enablement(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    variance = report.get("variance") or {}
    if not isinstance(variance, dict) or not bool(variance.get("enabled", False)):
        return errors

    resolved_policy = report.get("resolved_policy") or {}
    variance_policy = (
        resolved_policy.get("variance") if isinstance(resolved_policy, dict) else {}
    )
    min_effect = 0.0
    if isinstance(variance_policy, dict):
        parsed_min_effect = _coerce_float(variance_policy.get("min_effect_lognll"))
        if parsed_min_effect is not None:
            min_effect = max(0.0, parsed_min_effect)
    improvement_threshold = -min_effect

    predictive_gate = variance.get("predictive_gate")
    if not isinstance(predictive_gate, dict) or not predictive_gate:
        errors.append(
            "variance.enabled=true requires variance.predictive_gate evidence."
        )
    else:
        if predictive_gate.get("passed") is not True:
            errors.append(
                "variance.enabled=true requires variance.predictive_gate.passed == true."
            )

        mean_delta = _coerce_float(predictive_gate.get("mean_delta"))
        if mean_delta is None:
            errors.append(
                "variance.enabled=true requires finite variance.predictive_gate.mean_delta."
            )
        elif mean_delta >= 0.0:
            errors.append(
                "variance.predictive_gate.mean_delta must be negative when VE is enabled."
            )
        elif mean_delta > improvement_threshold:
            errors.append(
                "variance.predictive_gate.mean_delta does not meet "
                f"-min_effect_lognll ({improvement_threshold:.6g})."
            )

        delta_ci = predictive_gate.get("delta_ci")
        if delta_ci is None:
            delta_ci = predictive_gate.get("ci")
        lower = upper = None
        if isinstance(delta_ci, tuple | list) and len(delta_ci) == 2:
            lower = _coerce_float(delta_ci[0])
            upper = _coerce_float(delta_ci[1])
        if lower is None or upper is None:
            errors.append(
                "variance.enabled=true requires finite variance.predictive_gate.delta_ci."
            )
        elif lower > upper:
            errors.append(
                "variance.predictive_gate.delta_ci lower bound exceeds upper bound."
            )
        elif upper >= 0.0:
            errors.append(
                "variance.predictive_gate.delta_ci must exclude zero when VE is enabled."
            )
        elif upper > improvement_threshold:
            errors.append(
                "variance.predictive_gate.delta_ci upper bound does not meet "
                f"-min_effect_lognll ({improvement_threshold:.6g})."
            )

    ab_test = variance.get("ab_test")
    if not isinstance(ab_test, dict) or not ab_test:
        errors.append("variance.enabled=true requires variance.ab_test evidence.")
        return errors

    provenance = ab_test.get("provenance")
    seed = ab_test.get("seed")
    if seed in (None, "") and isinstance(provenance, dict):
        seed = provenance.get("seed")
    if seed in (None, ""):
        errors.append("variance.enabled=true requires variance.ab_test.seed.")

    windows_used = _coerce_int(ab_test.get("windows_used"))
    if windows_used is None or windows_used <= 0:
        errors.append(
            "variance.enabled=true requires positive variance.ab_test.windows_used."
        )

    if not isinstance(provenance, dict) or not provenance:
        errors.append("variance.enabled=true requires variance.ab_test.provenance.")
    elif not _collect_provenance_window_ids(provenance):
        errors.append(
            "variance.enabled=true requires variance.ab_test.provenance.window_ids."
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
            actual_val = _coerce_float(actual)
            expected_val = _coerce_float(expected)
            if actual_val is None or expected_val is None:
                errors.append(
                    f"{message} Expected numeric comparison for {path}, observed {actual!r}."
                )
            else:
                if actual_val < expected_val:
                    errors.append(
                        f"{message} Expected {path} ≥ {expected_val}, observed {actual_val}."
                    )
        elif lint_type == "lte":
            actual_val = _coerce_float(actual)
            expected_val = _coerce_float(expected)
            if actual_val is None or expected_val is None:
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
    validate_variance_enablement_fn: Callable[
        [dict[str, Any]], list[str]
    ] = _validate_variance_enablement,
) -> list[str]:
    errors: list[str] = []
    report = load_evaluation_report_fn(path)
    try:
        prof = (
            (profile or "").strip().lower()
            if isinstance(profile, str | None)
            else "dev"
        )
    except _VERIFY_PARSE_EXCEPTIONS:
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
    errors.extend(validate_variance_enablement_fn(report))
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
    "_measurement_contract_digest",
    "_resolve_path",
    "_validate_counts",
    "_validate_drift_band",
    "_validate_evaluation_report_payload",
    "_validate_measurement_contracts",
    "_validate_pairing",
    "_validate_tokenizer_hash",
    "_validate_variance_enablement",
    "REPORT_JSON_SCHEMA",
    "REPORT_SCHEMA_VERSION",
    "_VERIFY_PARSE_EXCEPTIONS",
    "_coerce_float",
    "_coerce_int",
    "_load_evaluation_report",
    "_report_schema",
    "_validate_logspace_ci_identity",
    "_validate_primary_metric",
    "_validate_primary_metric_policy",
    "_validate_release_gate_outcomes",
    "_validate_report_schema_strict",
    "resolve_tiny_relax_from_report",
    "validate_report",
]
