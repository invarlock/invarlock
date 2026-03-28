from __future__ import annotations

import math
from typing import Any

from invarlock.core.auto_tuning import get_tier_policies

from .guards_common import _baseline_guard_payload, _measurement_contract_digest
from .policy_utils import _resolve_policy_tier
from .report_types import RunReport

_PARSE_EXCEPTIONS = (AttributeError, KeyError, OverflowError, TypeError, ValueError)


def _extract_spectral_analysis(
    report: RunReport, baseline: dict[str, Any]
) -> dict[str, Any]:
    tier = _resolve_policy_tier(report)
    tier_policies = get_tier_policies()
    tier_defaults = tier_policies.get(tier, tier_policies.get("balanced", {}))
    spectral_defaults = tier_defaults.get("spectral", {}) if tier_defaults else {}
    default_sigma_quantile = spectral_defaults.get("sigma_quantile", 0.95)
    default_deadband = spectral_defaults.get("deadband", 0.1)
    default_caps = spectral_defaults.get("family_caps", {})
    default_max_caps = spectral_defaults.get("max_caps", 5)

    spectral_guard = None
    for guard in report.get("guards", []) or []:
        if str(guard.get("name", "")).lower() == "spectral":
            spectral_guard = guard
            break

    guard_policy = spectral_guard.get("policy", {}) if spectral_guard else {}
    guard_metrics = spectral_guard.get("metrics", {}) if spectral_guard else {}
    if guard_metrics:
        raw = (
            guard_metrics.get("violations_detected")
            or guard_metrics.get("violations_found")
            or guard_metrics.get("caps_applied")
            or (1 if guard_metrics.get("correction_applied") else 0)
            or 0
        )
        try:
            caps_applied = int(raw)
        except _PARSE_EXCEPTIONS:
            caps_applied = 0
    else:
        caps_applied = 0
    modules_checked = guard_metrics.get("modules_checked") if guard_metrics else None
    caps_exceeded = (
        bool(guard_metrics.get("caps_exceeded", False)) if guard_metrics else False
    )
    max_caps = guard_metrics.get("max_caps") if guard_metrics else None
    if max_caps is None and guard_policy:
        max_caps = guard_policy.get("max_caps")
    if max_caps is None:
        max_caps = default_max_caps
    try:
        max_caps = int(max_caps)
    except _PARSE_EXCEPTIONS:
        max_caps = int(default_max_caps)

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

    baseline_max = None
    baseline_mean = None
    baseline_spectral = _baseline_guard_payload(baseline, "spectral")
    if isinstance(baseline_spectral, dict) and baseline_spectral:
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
    guard_baseline_metrics = None
    if spectral_guard and isinstance(spectral_guard.get("baseline_metrics"), dict):
        guard_baseline_metrics = spectral_guard.get("baseline_metrics")
    if baseline_max is None and guard_baseline_metrics:
        baseline_max = guard_baseline_metrics.get("max_spectral_norm")
        baseline_mean = guard_baseline_metrics.get("mean_spectral_norm")
    baseline_max = float(baseline_max) if baseline_max not in (None, 0, 0.0) else None
    baseline_mean = (
        float(baseline_mean) if baseline_mean not in (None, 0, 0.0) else None
    )

    max_sigma_ratio = (
        max_spectral_norm / baseline_max if baseline_max and baseline_max > 0 else 1.0
    )
    median_sigma_ratio = (
        mean_spectral_norm / baseline_mean
        if baseline_mean and baseline_mean > 0
        else 1.0
    )

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
            sorted_values[lower]
            + (sorted_values[upper] - sorted_values[lower]) * fraction
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
            top_entries = sorted(value_list, key=lambda t: abs(t[0]), reverse=True)[:3]
            top_z_scores_local[family] = [
                {"module": name, "z": float(z)} for z, name in top_entries
            ]

        return family_quantiles_local, top_z_scores_local

    summary: dict[str, Any] = {}
    family_quantiles: dict[str, dict[str, float]] = {}
    families: dict[str, dict[str, Any]] = {}
    family_caps: dict[str, dict[str, float]] = {}
    top_z_scores: dict[str, list[dict[str, Any]]] = {}
    deadband_used: float | None = None

    if isinstance(guard_metrics, dict):
        try:
            db_raw = guard_policy.get("deadband") if guard_policy else None
            if db_raw is None and isinstance(guard_metrics, dict):
                db_raw = guard_metrics.get("deadband")
            if db_raw is None:
                db_raw = default_deadband
            if db_raw is not None:
                deadband_used = float(db_raw)
        except _PARSE_EXCEPTIONS:
            deadband_used = None

        sigma_q_used: float | None = None
        try:
            pol_sq = None
            if isinstance(guard_policy, dict):
                pol_sq = guard_policy.get("sigma_quantile")
            if pol_sq is None:
                pol_sq = default_sigma_quantile
            if pol_sq is not None:
                sigma_q_used = float(pol_sq)
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
            summary["stability_score"] = float(
                guard_metrics.get(
                    "spectral_stability_score",
                    guard_metrics.get("stability_score", 1.0),
                )
            )
        except _PARSE_EXCEPTIONS:
            pass
        family_quantiles = (
            guard_metrics.get("family_z_quantiles")
            if isinstance(guard_metrics.get("family_z_quantiles"), dict)
            else {}
        )
        if not family_quantiles:
            family_quantiles = (
                guard_metrics.get("family_z_summary")
                if isinstance(guard_metrics.get("family_z_summary"), dict)
                else {}
            )
        families = (
            guard_metrics.get("families")
            if isinstance(guard_metrics.get("families"), dict)
            else {}
        )
        if not families:
            fzs = guard_metrics.get("family_z_summary")
            if not isinstance(fzs, dict) or not fzs:
                fzs = guard_metrics.get("family_stats")
            if isinstance(fzs, dict):
                for fam, stats in fzs.items():
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
                        and family_caps.get(str(fam), {}).get("kappa") is not None
                    ):
                        kappa = family_caps[str(fam)]["kappa"]
                    try:
                        if kappa is not None:
                            entry["kappa"] = float(kappa)
                    except _PARSE_EXCEPTIONS:
                        pass
                    if entry:
                        families[str(fam)] = entry
        family_caps = (
            guard_metrics.get("family_caps")
            if isinstance(guard_metrics.get("family_caps"), dict)
            else {}
        )
        if not family_caps and isinstance(guard_policy, dict):
            fam_caps_pol = guard_policy.get("family_caps")
            if isinstance(fam_caps_pol, dict):
                family_caps = fam_caps_pol
        if not family_caps and isinstance(default_caps, dict):
            family_caps = default_caps
        raw_top = (
            guard_metrics.get("top_z_scores")
            if isinstance(guard_metrics.get("top_z_scores"), dict)
            else {}
        )
        top_z_scores = {}
        if isinstance(raw_top, dict):
            for fam, entries in raw_top.items():
                if not isinstance(entries, list):
                    continue
                cleaned: list[dict[str, Any]] = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    mod = entry.get("module")
                    z = entry.get("z")
                    try:
                        zf = float(z)
                    except _PARSE_EXCEPTIONS:
                        continue
                    cleaned.append({"module": mod, "z": zf})
                if cleaned:
                    cleaned.sort(key=lambda d: abs(d.get("z", 0.0)), reverse=True)
                    top_z_scores[str(fam)] = cleaned[:3]

    if spectral_guard:
        z_map_candidate = spectral_guard.get("final_z_scores") or guard_metrics.get(
            "final_z_scores"
        )
        family_map_candidate = spectral_guard.get(
            "module_family_map"
        ) or guard_metrics.get("module_family_map")
        derived_quantiles, derived_top = _summarize_from_z_scores(
            z_map_candidate, family_map_candidate
        )
        if derived_quantiles and not family_quantiles:
            family_quantiles = derived_quantiles
        if isinstance(derived_top, dict) and derived_top:
            if not isinstance(top_z_scores, dict) or not top_z_scores:
                top_z_scores = dict(derived_top)
            else:
                for fam, entries in derived_top.items():
                    cur = top_z_scores.get(fam)
                    if not isinstance(cur, list) or not cur:
                        top_z_scores[fam] = entries

    if not guard_metrics:
        spectral_data = (report.get("metrics", {}) or {}).get("spectral", {})
        if isinstance(spectral_data, dict):
            ratios = spectral_data.get("sigma_ratios")
            if isinstance(ratios, list) and ratios:
                try:
                    float_ratios = [float(r) for r in ratios]
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

    multiple_testing = _resolve_multiple_testing(
        guard_metrics, guard_policy, spectral_defaults
    )

    policy_out: dict[str, Any] | None = None
    if isinstance(guard_policy, dict) and guard_policy:
        policy_out = dict(guard_policy)
        if default_sigma_quantile is not None:
            sq = policy_out.get("sigma_quantile")
            if sq is not None:
                try:
                    policy_out["sigma_quantile"] = float(sq)
                except _PARSE_EXCEPTIONS:
                    pass
        if tier == "balanced":
            policy_out["correction_enabled"] = False
            policy_out["max_spectral_norm"] = None
        if multiple_testing and "multiple_testing" not in policy_out:
            policy_out["multiple_testing"] = multiple_testing

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
    max_caps_val = int(max_caps) if isinstance(max_caps, int | float) else None
    result["max_caps"] = max_caps_val
    try:
        summary["max_caps"] = max_caps_val
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
        caps_by_family = {
            fam: int(details.get("violations", 0))
            for fam, details in families.items()
            if isinstance(details, dict)
        }
        result["caps_applied_by_family"] = caps_by_family
    if top_z_scores:
        result["top_z_scores"] = top_z_scores
    if spectral_guard and isinstance(spectral_guard.get("violations"), list):
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
            z_score = violation.get("z_score")
            try:
                entry["z_score"] = float(z_score)
            except _PARSE_EXCEPTIONS:
                pass
            top_violations.append(entry)
        if top_violations:
            result["top_violations"] = top_violations
    if family_quantiles:
        result["family_z_quantiles"] = family_quantiles
    result["evaluated"] = bool(spectral_guard)

    measurement_contract = None
    try:
        mc = (
            guard_metrics.get("measurement_contract")
            if isinstance(guard_metrics, dict)
            else None
        )
        if isinstance(mc, dict) and mc:
            measurement_contract = mc
    except _PARSE_EXCEPTIONS:
        measurement_contract = None
    baseline_contract = None
    try:
        bc = (
            baseline_spectral.get("measurement_contract")
            if isinstance(baseline_spectral, dict)
            else None
        )
        if isinstance(bc, dict) and bc:
            baseline_contract = bc
    except _PARSE_EXCEPTIONS:
        baseline_contract = None
    mc_hash = _measurement_contract_digest(measurement_contract)
    baseline_hash = _measurement_contract_digest(baseline_contract)
    if measurement_contract is not None:
        result["measurement_contract"] = measurement_contract
    if mc_hash:
        result["measurement_contract_hash"] = mc_hash
    if baseline_hash:
        result["baseline_measurement_contract_hash"] = baseline_hash
    if mc_hash and baseline_hash:
        result["measurement_contract_match"] = bool(mc_hash == baseline_hash)
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

    if families:
        caps_by_family = {
            family: int(details.get("violations", 0))
            for family, details in (families or {}).items()
            if isinstance(details, dict)
        }
        result["caps_applied_by_family"] = caps_by_family
    if top_z_scores:
        result["top_z_scores"] = top_z_scores
    if family_quantiles:
        result["family_z_quantiles"] = family_quantiles
    return result
