from __future__ import annotations

import math
from typing import Any

from invarlock.core.auto_tuning import get_tier_policies

from .guards_common import _baseline_guard_payload, _measurement_contract_digest
from .policy_utils import _resolve_policy_tier
from .report_types import RunReport

_PARSE_EXCEPTIONS = (AttributeError, KeyError, OverflowError, TypeError, ValueError)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_float_or_none(value: Any) -> float | None:
    if value in (None, "", 0, 0.0):
        return None
    try:
        return float(value)
    except _PARSE_EXCEPTIONS:
        return None


def _to_int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except _PARSE_EXCEPTIONS:
        return default


def _compute_quantile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    fraction = position - lower
    return (
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def _summarize_from_z_scores(
    z_scores_map: Any, module_family_map: Any
) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]:
    from collections import defaultdict

    if not isinstance(z_scores_map, dict) or not z_scores_map:
        return {}, {}
    if not isinstance(module_family_map, dict) or not module_family_map:
        return {}, {}

    per_family_values: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for module_name, z_value in z_scores_map.items():
        family = module_family_map.get(module_name)
        if family is None:
            continue
        try:
            z_abs = abs(float(z_value))
        except (TypeError, ValueError):
            continue
        per_family_values[family].append((z_abs, module_name))

    family_quantiles_local: dict[str, dict[str, float]] = {}
    top_z_scores_local: dict[str, list[dict[str, Any]]] = {}
    for family, value_list in per_family_values.items():
        if not value_list:
            continue
        sorted_scores = sorted(z for z, _ in value_list)
        family_quantiles_local[family] = {
            "q95": _compute_quantile(sorted_scores, 0.95),
            "q99": _compute_quantile(sorted_scores, 0.99),
            "max": sorted_scores[-1],
            "count": len(sorted_scores),
        }
        top_entries = sorted(value_list, key=lambda entry: abs(entry[0]), reverse=True)[
            :3
        ]
        top_z_scores_local[family] = [
            {"module": name, "z": float(z)} for z, name in top_entries
        ]
    return family_quantiles_local, top_z_scores_local


def _resolve_spectral_defaults(
    report: RunReport,
) -> tuple[str, dict[str, Any], Any, Any, Any, Any]:
    tier = _resolve_policy_tier(report)
    tier_policies = get_tier_policies()
    tier_defaults = tier_policies.get(tier, tier_policies.get("balanced", {}))
    spectral_defaults = tier_defaults.get("spectral", {}) if tier_defaults else {}
    return (
        tier,
        spectral_defaults,
        spectral_defaults.get("sigma_quantile", 0.95),
        spectral_defaults.get("deadband", 0.1),
        spectral_defaults.get("family_caps", {}),
        spectral_defaults.get("max_caps", 5),
    )


def _find_spectral_guard(report: RunReport) -> dict[str, Any] | None:
    for guard in report.get("guards", []) or []:
        guard_entry = _as_dict(guard)
        if str(guard_entry.get("name", "")).lower() == "spectral":
            return guard_entry
    return None


def _resolve_guard_context(
    spectral_guard: dict[str, Any] | None,
    default_max_caps: Any,
) -> dict[str, Any]:
    guard_policy = _as_dict(spectral_guard.get("policy") if spectral_guard else {})
    guard_metrics = _as_dict(spectral_guard.get("metrics") if spectral_guard else {})
    if guard_metrics:
        raw = (
            guard_metrics.get("violations_detected")
            or guard_metrics.get("violations_found")
            or guard_metrics.get("caps_applied")
            or (1 if guard_metrics.get("correction_applied") else 0)
            or 0
        )
        caps_applied = _to_int_or_default(raw, 0)
    else:
        caps_applied = 0
    modules_checked = guard_metrics.get("modules_checked") if guard_metrics else None
    caps_exceeded = (
        bool(guard_metrics.get("caps_exceeded", False)) if guard_metrics else False
    )
    max_caps = guard_metrics.get("max_caps") if guard_metrics else None
    if max_caps is None and guard_policy:
        max_caps = guard_policy.get("max_caps")
    max_caps = _to_int_or_default(
        max_caps if max_caps is not None else default_max_caps,
        _to_int_or_default(default_max_caps, 5),
    )
    try:
        max_spectral_norm = float(
            guard_metrics.get("max_spectral_norm_final")
            or guard_metrics.get("max_spectral_norm")
            or 0.0
        )
    except _PARSE_EXCEPTIONS:
        max_spectral_norm = 0.0
    try:
        mean_spectral_norm = float(
            guard_metrics.get("mean_spectral_norm_final")
            or guard_metrics.get("mean_spectral_norm")
            or 0.0
        )
    except _PARSE_EXCEPTIONS:
        mean_spectral_norm = 0.0
    return {
        "guard_policy": guard_policy,
        "guard_metrics": guard_metrics,
        "caps_applied": caps_applied,
        "modules_checked": modules_checked,
        "caps_exceeded": caps_exceeded,
        "max_caps": max_caps,
        "max_spectral_norm": max_spectral_norm,
        "mean_spectral_norm": mean_spectral_norm,
    }


def _resolve_baseline_spectral(
    baseline: dict[str, Any], spectral_guard: dict[str, Any] | None
) -> tuple[dict[str, Any], float | None, float | None]:
    baseline_max = None
    baseline_mean = None
    baseline_spectral = _as_dict(_baseline_guard_payload(baseline, "spectral"))
    if baseline_spectral:
        baseline_max = baseline_spectral.get(
            "max_spectral_norm", baseline_spectral.get("max_spectral_norm_final")
        )
        baseline_mean = baseline_spectral.get(
            "mean_spectral_norm", baseline_spectral.get("mean_spectral_norm_final")
        )
    if baseline_max is None:
        baseline_metrics = (
            baseline.get("metrics", {}) if isinstance(baseline, dict) else {}
        )
        if isinstance(baseline_metrics, dict) and "spectral" in baseline_metrics:
            baseline_spectral_metrics = baseline_metrics["spectral"]
            if isinstance(baseline_spectral_metrics, dict):
                baseline_max = baseline_spectral_metrics.get("max_spectral_norm_final")
                baseline_mean = baseline_spectral_metrics.get(
                    "mean_spectral_norm_final"
                )
    guard_baseline_metrics: dict[str, Any] | None = None
    if spectral_guard:
        baseline_metrics = spectral_guard.get("baseline_metrics")
        if isinstance(baseline_metrics, dict):
            guard_baseline_metrics = baseline_metrics
    if baseline_max is None and guard_baseline_metrics:
        baseline_max = guard_baseline_metrics.get("max_spectral_norm")
        baseline_mean = guard_baseline_metrics.get("mean_spectral_norm")
    return (
        baseline_spectral,
        _to_float_or_none(baseline_max),
        _to_float_or_none(baseline_mean),
    )


def _build_guard_summary(
    guard_metrics: dict[str, Any],
    guard_policy: dict[str, Any],
    default_deadband: Any,
    default_sigma_quantile: Any,
    default_caps: Any,
    max_sigma_ratio: float,
    median_sigma_ratio: float,
    max_spectral_norm: float,
    mean_spectral_norm: float,
    baseline_max: float | None,
    baseline_mean: float | None,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, float]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, float]],
    dict[str, list[dict[str, Any]]],
    float | None,
]:
    summary: dict[str, Any] = {}
    family_quantiles: dict[str, dict[str, float]] = {}
    families: dict[str, dict[str, Any]] = {}
    family_caps: dict[str, dict[str, float]] = {}
    top_z_scores: dict[str, list[dict[str, Any]]] = {}
    deadband_used: float | None = None
    try:
        deadband_raw = guard_policy.get("deadband") if guard_policy else None
        if deadband_raw is None:
            deadband_raw = guard_metrics.get("deadband")
        if deadband_raw is None:
            deadband_raw = default_deadband
        if deadband_raw is not None:
            deadband_used = float(deadband_raw)
    except _PARSE_EXCEPTIONS:
        deadband_used = None

    sigma_q_used: float | None = None
    try:
        sigma_q_raw = guard_policy.get("sigma_quantile") if guard_policy else None
        if sigma_q_raw is None:
            sigma_q_raw = default_sigma_quantile
        if sigma_q_raw is not None:
            sigma_q_used = float(sigma_q_raw)
    except _PARSE_EXCEPTIONS:
        sigma_q_used = None

    summary = {
        "max_sigma_ratio": max_sigma_ratio,
        "median_sigma_ratio": median_sigma_ratio,
        "max_spectral_norm": max_spectral_norm,
        "mean_spectral_norm": mean_spectral_norm,
        "baseline_max_spectral_norm": baseline_max,
        "baseline_mean_spectral_norm": baseline_mean,
    }
    if sigma_q_used is not None:
        summary["sigma_quantile"] = sigma_q_used
    if deadband_used is not None:
        summary["deadband"] = deadband_used
    try:
        stability_score = _to_float_or_none(
            guard_metrics.get(
                "spectral_stability_score",
                guard_metrics.get("stability_score", 1.0),
            )
        )
        if stability_score is not None:
            summary["stability_score"] = stability_score
    except _PARSE_EXCEPTIONS:
        pass

    family_quantiles = _as_dict(guard_metrics.get("family_z_quantiles"))
    if not family_quantiles:
        family_quantiles = _as_dict(guard_metrics.get("family_z_summary"))
    families = _as_dict(guard_metrics.get("families"))
    if not families:
        family_summary_source = guard_metrics.get("family_z_summary")
        if not isinstance(family_summary_source, dict) or not family_summary_source:
            family_summary_source = guard_metrics.get("family_stats")
        if isinstance(family_summary_source, dict):
            for family_name, stats in family_summary_source.items():
                if not isinstance(stats, dict):
                    continue
                entry: dict[str, Any] = {}
                if "max" in stats:
                    try:
                        entry["max"] = float(stats["max"])
                    except _PARSE_EXCEPTIONS:
                        pass
                if "mean" in stats:
                    try:
                        entry["mean"] = float(stats["mean"])
                    except _PARSE_EXCEPTIONS:
                        pass
                if "count" in stats:
                    try:
                        entry["count"] = int(stats["count"])
                    except _PARSE_EXCEPTIONS:
                        pass
                if "violations" in stats:
                    try:
                        entry["violations"] = int(stats["violations"])
                    except _PARSE_EXCEPTIONS:
                        pass
                kappa = stats.get("kappa") if isinstance(stats, dict) else None
                if (
                    kappa is None
                    and family_caps.get(str(family_name), {}).get("kappa") is not None
                ):
                    kappa = family_caps[str(family_name)]["kappa"]
                try:
                    if kappa is not None:
                        entry["kappa"] = float(kappa)
                except _PARSE_EXCEPTIONS:
                    pass
                if entry:
                    families[str(family_name)] = entry

    family_caps = _as_dict(guard_metrics.get("family_caps"))
    if not family_caps and isinstance(guard_policy, dict):
        family_caps_from_policy = guard_policy.get("family_caps")
        if isinstance(family_caps_from_policy, dict):
            family_caps = family_caps_from_policy
    if not family_caps and isinstance(default_caps, dict):
        family_caps = default_caps

    raw_top = _as_dict(guard_metrics.get("top_z_scores"))
    top_z_scores = {}
    for family_name, entries in raw_top.items():
        if not isinstance(entries, list):
            continue
        cleaned: list[dict[str, Any]] = []
        for entry_raw in entries:
            entry = _as_dict(entry_raw)
            if not entry:
                continue
            zf = _to_float_or_none(entry.get("z"))
            if zf is None:
                continue
            cleaned.append({"module": entry.get("module"), "z": zf})
        if cleaned:
            cleaned.sort(key=lambda item: abs(item.get("z", 0.0)), reverse=True)
            top_z_scores[str(family_name)] = cleaned[:3]

    return summary, family_quantiles, families, family_caps, top_z_scores, deadband_used


def _derive_z_score_tables(
    spectral_guard: dict[str, Any] | None,
    guard_metrics: dict[str, Any],
    family_quantiles: dict[str, dict[str, float]],
    top_z_scores: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]:
    if not spectral_guard:
        return family_quantiles, top_z_scores
    z_map_candidate = spectral_guard.get("final_z_scores") or guard_metrics.get(
        "final_z_scores"
    )
    family_map_candidate = spectral_guard.get("module_family_map") or guard_metrics.get(
        "module_family_map"
    )
    derived_quantiles, derived_top = _summarize_from_z_scores(
        z_map_candidate, family_map_candidate
    )
    if derived_quantiles and not family_quantiles:
        family_quantiles = derived_quantiles
    if isinstance(derived_top, dict) and derived_top:
        if not isinstance(top_z_scores, dict) or not top_z_scores:
            top_z_scores = dict(derived_top)
        else:
            for family_name, entries in derived_top.items():
                current_entries = top_z_scores.get(family_name)
                if not isinstance(current_entries, list) or not current_entries:
                    top_z_scores[family_name] = entries
    return family_quantiles, top_z_scores


def _apply_metrics_only_ratio_fallback(
    report: RunReport, guard_metrics: dict[str, Any], summary: dict[str, Any]
) -> None:
    if guard_metrics:
        return
    spectral_data = (report.get("metrics", {}) or {}).get("spectral")
    if not isinstance(spectral_data, dict):
        return
    ratios = spectral_data.get("sigma_ratios")
    if not isinstance(ratios, list) or not ratios:
        return
    try:
        float_ratios = [float(value) for value in ratios]
        summary["max_sigma_ratio"] = max(float_ratios)
        summary["median_sigma_ratio"] = float(
            sorted(float_ratios)[len(float_ratios) // 2]
        )
    except _PARSE_EXCEPTIONS:
        pass


def _resolve_multiple_testing(*sources: Any) -> dict[str, Any] | None:
    for source in sources:
        if not isinstance(source, dict):
            continue
        candidate = source.get("multiple_testing")
        if isinstance(candidate, dict) and candidate:
            return candidate
    return None


def _build_policy_output(
    guard_policy: dict[str, Any],
    default_sigma_quantile: Any,
    multiple_testing: dict[str, Any] | None,
    tier: str,
) -> dict[str, Any] | None:
    if not isinstance(guard_policy, dict) or not guard_policy:
        return None
    policy_out = dict(guard_policy)
    if default_sigma_quantile is not None:
        sigma_quantile = policy_out.get("sigma_quantile")
        if sigma_quantile is not None:
            try:
                policy_out["sigma_quantile"] = float(sigma_quantile)
            except _PARSE_EXCEPTIONS:
                pass
    if tier == "balanced":
        policy_out["correction_enabled"] = False
        policy_out["max_spectral_norm"] = None
    if multiple_testing and "multiple_testing" not in policy_out:
        policy_out["multiple_testing"] = multiple_testing
    return policy_out


def _build_top_violations(
    spectral_guard: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    if not spectral_guard or not isinstance(spectral_guard.get("violations"), list):
        return None
    top_violations: list[dict[str, Any]] = []
    for violation in spectral_guard["violations"][:5]:
        if not isinstance(violation, dict):
            continue
        entry = {
            "module": violation.get("module"),
            "family": violation.get("family"),
            "kappa": violation.get("kappa"),
            "severity": violation.get("severity", "warn"),
        }
        z_score_value = _to_float_or_none(violation.get("z_score"))
        if z_score_value is not None:
            entry["z_score"] = z_score_value
        top_violations.append(entry)
    return top_violations or None


def _build_measurement_contract_fields(
    guard_metrics: dict[str, Any], baseline_spectral: dict[str, Any]
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    measurement_contract = _as_dict(guard_metrics.get("measurement_contract")) or None
    baseline_contract = _as_dict(baseline_spectral.get("measurement_contract")) or None
    measurement_contract_hash = _measurement_contract_digest(measurement_contract)
    baseline_hash = _measurement_contract_digest(baseline_contract)
    if measurement_contract is not None:
        fields["measurement_contract"] = measurement_contract
    if measurement_contract_hash:
        fields["measurement_contract_hash"] = measurement_contract_hash
    if baseline_hash:
        fields["baseline_measurement_contract_hash"] = baseline_hash
    if measurement_contract_hash and baseline_hash:
        fields["measurement_contract_match"] = bool(
            measurement_contract_hash == baseline_hash
        )
    return fields


def _build_spectral_result(
    tier: str,
    spectral_guard: dict[str, Any] | None,
    baseline_spectral: dict[str, Any],
    guard_policy: dict[str, Any],
    guard_metrics: dict[str, Any],
    caps_applied: int,
    summary: dict[str, Any],
    families: dict[str, dict[str, Any]],
    family_caps: dict[str, dict[str, float]],
    family_quantiles: dict[str, dict[str, float]],
    top_z_scores: dict[str, list[dict[str, Any]]],
    multiple_testing: dict[str, Any] | None,
    policy_out: dict[str, Any] | None,
    default_sigma_quantile: Any,
    deadband_used: float | None,
    max_caps: int,
    caps_exceeded: bool,
    modules_checked: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tier": tier,
        "caps_applied": caps_applied,
        "summary": summary,
        "families": families,
        "family_caps": family_caps,
    }
    try:
        summary["status"] = "stable" if int(caps_applied) == 0 else "capped"
    except _PARSE_EXCEPTIONS:
        summary["status"] = "stable" if not caps_applied else "capped"
    if policy_out:
        result["policy"] = policy_out
    if default_sigma_quantile is not None:
        result["sigma_quantile"] = default_sigma_quantile
    if deadband_used is not None:
        result["deadband"] = deadband_used
    result["max_caps"] = max_caps if isinstance(max_caps, (int, float)) else None
    try:
        summary["max_caps"] = result["max_caps"]
    except _PARSE_EXCEPTIONS:
        pass

    if multiple_testing:
        mt_copy = dict(multiple_testing)
        families_present = set((families or {}).keys()) or set(
            (family_caps or {}).keys()
        )
        try:
            mt_copy["m"] = int(mt_copy.get("m") or len(families_present))
        except _PARSE_EXCEPTIONS:
            mt_copy["m"] = len(families_present)
        result["multiple_testing"] = mt_copy
        result["bh_family_count"] = mt_copy["m"]

    if families:
        result["caps_applied_by_family"] = {
            family_name: int(details.get("violations", 0))
            for family_name, details in families.items()
            if isinstance(details, dict)
        }
    if top_z_scores:
        result["top_z_scores"] = top_z_scores
    top_violations = _build_top_violations(spectral_guard)
    if top_violations:
        result["top_violations"] = top_violations
    if family_quantiles:
        result["family_z_quantiles"] = family_quantiles
    result["evaluated"] = bool(spectral_guard)
    result.update(_build_measurement_contract_fields(guard_metrics, baseline_spectral))
    result["caps_exceeded"] = bool(caps_exceeded)
    try:
        summary["caps_exceeded"] = bool(caps_exceeded)
    except _PARSE_EXCEPTIONS:
        pass
    if modules_checked is not None:
        try:
            summary["modules_checked"] = int(modules_checked)
        except _PARSE_EXCEPTIONS:
            pass
    return result


def _extract_spectral_analysis(
    report: RunReport, baseline: dict[str, Any]
) -> dict[str, Any]:
    (
        tier,
        spectral_defaults,
        default_sigma_quantile,
        default_deadband,
        default_caps,
        default_max_caps,
    ) = _resolve_spectral_defaults(report)
    spectral_guard = _find_spectral_guard(report)
    guard_context = _resolve_guard_context(spectral_guard, default_max_caps)
    guard_policy = guard_context["guard_policy"]
    guard_metrics = guard_context["guard_metrics"]
    caps_applied = guard_context["caps_applied"]
    modules_checked = guard_context["modules_checked"]
    caps_exceeded = guard_context["caps_exceeded"]
    max_caps = guard_context["max_caps"]
    max_spectral_norm = guard_context["max_spectral_norm"]
    mean_spectral_norm = guard_context["mean_spectral_norm"]

    baseline_spectral, baseline_max, baseline_mean = _resolve_baseline_spectral(
        baseline, spectral_guard
    )
    max_sigma_ratio = (
        max_spectral_norm / baseline_max if baseline_max and baseline_max > 0 else 1.0
    )
    median_sigma_ratio = (
        mean_spectral_norm / baseline_mean
        if baseline_mean and baseline_mean > 0
        else 1.0
    )

    (
        summary,
        family_quantiles,
        families,
        family_caps,
        top_z_scores,
        deadband_used,
    ) = _build_guard_summary(
        guard_metrics,
        guard_policy,
        default_deadband,
        default_sigma_quantile,
        default_caps,
        max_sigma_ratio,
        median_sigma_ratio,
        max_spectral_norm,
        mean_spectral_norm,
        baseline_max,
        baseline_mean,
    )
    family_quantiles, top_z_scores = _derive_z_score_tables(
        spectral_guard, guard_metrics, family_quantiles, top_z_scores
    )
    _apply_metrics_only_ratio_fallback(report, guard_metrics, summary)

    multiple_testing = _resolve_multiple_testing(
        guard_metrics, guard_policy, spectral_defaults
    )
    policy_out = _build_policy_output(
        guard_policy, default_sigma_quantile, multiple_testing, tier
    )
    return _build_spectral_result(
        tier=tier,
        spectral_guard=spectral_guard,
        baseline_spectral=baseline_spectral,
        guard_policy=guard_policy,
        guard_metrics=guard_metrics,
        caps_applied=caps_applied,
        summary=summary,
        families=families,
        family_caps=family_caps,
        family_quantiles=family_quantiles,
        top_z_scores=top_z_scores,
        multiple_testing=multiple_testing,
        policy_out=policy_out,
        default_sigma_quantile=default_sigma_quantile,
        deadband_used=deadband_used,
        max_caps=max_caps,
        caps_exceeded=caps_exceeded,
        modules_checked=modules_checked,
    )
