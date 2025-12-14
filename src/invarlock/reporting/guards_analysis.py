# mypy: ignore-errors
from __future__ import annotations

import math
from typing import Any, no_type_check

from invarlock.core.auto_tuning import get_tier_policies

from .policy_utils import _promote_legacy_multiple_testing_key, _resolve_policy_tier
from .report_types import RunReport


@no_type_check
def _extract_invariants(report: RunReport) -> dict[str, Any]:
    """Extract invariant check results (matches legacy shape used in tests)."""
    invariants_data = (report.get("metrics", {}) or {}).get("invariants", {})
    failures: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    # Collect failures from metrics.invariants
    if isinstance(invariants_data, dict) and invariants_data:
        for check_name, check_result in invariants_data.items():
            if isinstance(check_result, dict):
                if bool(check_result.get("passed", True)):
                    continue
                recorded_violation = False
                violations = check_result.get("violations")
                if isinstance(violations, list) and violations:
                    for violation in violations:
                        if not isinstance(violation, dict):
                            continue
                        entry: dict[str, Any] = {
                            "check": check_name,
                            "type": str(violation.get("type", "violation")),
                            "severity": violation.get("severity", "warning"),
                        }
                        detail = {k: v for k, v in violation.items() if k != "type"}
                        if detail:
                            entry["detail"] = detail
                        failures.append(entry)
                        recorded_violation = True
                if recorded_violation:
                    continue
                # No explicit violations list – treat as error
                failure_entry = {"check": check_name}
                failure_entry["type"] = str(check_result.get("type") or "failure")
                failure_entry["severity"] = "error"
                detail = {
                    k: v
                    for k, v in check_result.items()
                    if k not in {"passed", "violations", "type"}
                }
                if check_result.get("message"):
                    detail.setdefault("message", check_result["message"])
                if detail:
                    failure_entry["detail"] = detail
                failures.append(failure_entry)
            else:
                # Non-dict value: treat False as error severity
                if not bool(check_result):
                    failures.append(
                        {"check": check_name, "type": "failure", "severity": "error"}
                    )

    # Guard-level invariants info (counts + detailed violations)
    guard_entry = None
    for guard in report.get("guards", []) or []:
        if str(guard.get("name", "")).lower() == "invariants":
            guard_entry = guard
            break

    severity_status = "pass"
    if guard_entry:
        gm = guard_entry.get("metrics", {}) or {}
        summary = {
            "checks_performed": gm.get("checks_performed"),
            "violations_found": gm.get("violations_found"),
            "fatal_violations": gm.get("fatal_violations"),
            "warning_violations": gm.get("warning_violations"),
        }
        violations = guard_entry.get("violations", [])
        fatal_count = int(gm.get("fatal_violations", 0) or 0)
        warning_count = int(gm.get("warning_violations", 0) or 0)
        if violations:
            for violation in violations:
                if not isinstance(violation, dict):
                    continue
                row = {
                    "check": str(
                        violation.get("check") or violation.get("name") or "invariant"
                    ),
                    "type": str(violation.get("type") or "violation"),
                    "severity": str(violation.get("severity") or "warning"),
                }
                detail = {k: v for k, v in violation.items() if k not in row}
                if detail:
                    row["detail"] = detail
                failures.append(row)
        if fatal_count > 0:
            severity_status = "fail"
        elif warning_count > 0 or violations:
            severity_status = "warn"

    # If any error-severity entry exists among failures, escalate to fail
    if failures:
        has_error = any(str(f.get("severity", "warning")) == "error" for f in failures)
        if has_error:
            severity_status = "fail"
        elif severity_status == "pass":
            severity_status = "warn"

    status = severity_status
    if not summary:
        summary = {
            "checks_performed": 0,
            "violations_found": len(failures),
            "fatal_violations": 0,
            "warning_violations": len(failures),
        }

    return {
        "pre": "pass",
        "post": status,
        "status": status,
        "summary": summary,
        "details": invariants_data,
        "failures": failures,
    }


@no_type_check
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
        except Exception:
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
    except Exception:
        max_caps = int(default_max_caps)

    try:
        max_spectral_norm = float(
            guard_metrics.get("max_spectral_norm_final")
            or guard_metrics.get("max_spectral_norm")
            or 0.0
        )
    except Exception:
        max_spectral_norm = 0.0
    try:
        mean_spectral_norm = float(
            guard_metrics.get("mean_spectral_norm_final")
            or guard_metrics.get("mean_spectral_norm")
            or 0.0
        )
    except Exception:
        mean_spectral_norm = 0.0

    baseline_max = None
    baseline_mean = None
    baseline_spectral = (
        baseline.get("spectral", {}) if isinstance(baseline, dict) else {}
    )
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

    if isinstance(guard_metrics, dict):
        # Resolve deadband from policy/metrics/defaults
        deadband_used: float | None = None
        try:
            db_raw = guard_policy.get("deadband") if guard_policy else None
            if db_raw is None and isinstance(guard_metrics, dict):
                db_raw = guard_metrics.get("deadband")
            if db_raw is None:
                db_raw = default_deadband
            if db_raw is not None:
                deadband_used = float(db_raw)
        except Exception:
            deadband_used = None

        # Resolve sigma_quantile for summary (policy aliases supported)
        sigma_q_used: float | None = None
        try:
            pol_sq = None
            if isinstance(guard_policy, dict):
                pol_sq = (
                    guard_policy.get("sigma_quantile")
                    or guard_policy.get("contraction")
                    or guard_policy.get("kappa")
                )
            if pol_sq is None:
                pol_sq = default_sigma_quantile
            if pol_sq is not None:
                sigma_q_used = float(pol_sq)
        except Exception:
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
        except Exception:
            pass
        # Prefer explicit family_z_quantiles when present; otherwise accept summary
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
        # Build families table from available sources
        families = (
            guard_metrics.get("families")
            if isinstance(guard_metrics.get("families"), dict)
            else {}
        )
        if not families:
            # Prefer z-summary when available; accept legacy 'family_stats' too
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
                        except Exception:
                            pass
                    if "mean" in stats:
                        try:
                            entry["mean"] = float(stats["mean"])
                        except Exception:
                            pass
                    if "count" in stats:
                        try:
                            entry["count"] = int(stats["count"])
                        except Exception:
                            pass
                    if "violations" in stats:
                        try:
                            entry["violations"] = int(stats["violations"])
                        except Exception:
                            pass
                    # Propagate kappa from stats or family_caps
                    kappa = stats.get("kappa") if isinstance(stats, dict) else None
                    if (
                        kappa is None
                        and family_caps.get(str(fam), {}).get("kappa") is not None
                    ):
                        kappa = family_caps[str(fam)]["kappa"]
                    try:
                        if kappa is not None:
                            entry["kappa"] = float(kappa)
                    except Exception:
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
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    mod = e.get("module")
                    z = e.get("z")
                    try:
                        zf = float(z)
                    except Exception:
                        continue
                    cleaned.append({"module": mod, "z": zf})
                if cleaned:
                    cleaned.sort(key=lambda d: abs(d.get("z", 0.0)), reverse=True)
                    top_z_scores[str(fam)] = cleaned[:3]

    # Derive quantiles/top z from z-scores when available, and fill any gaps
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
        # Always backfill missing families in top_z_scores from derived_top
        if isinstance(derived_top, dict) and derived_top:
            if not isinstance(top_z_scores, dict) or not top_z_scores:
                top_z_scores = dict(derived_top)
            else:
                for fam, entries in derived_top.items():
                    cur = top_z_scores.get(fam)
                    if not isinstance(cur, list) or not cur:
                        top_z_scores[fam] = entries

    # Fallback: compute sigma ratios from raw ratios array when present
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
                except Exception:
                    pass

    # Multiple testing resolution
    def _resolve_multiple_testing(*sources: Any) -> dict[str, Any] | None:
        for source in sources:
            if not isinstance(source, dict):
                continue
            candidate = source.get("multiple_testing") or source.get("multipletesting")
            if isinstance(candidate, dict) and candidate:
                return candidate
        return None

    multiple_testing = _resolve_multiple_testing(
        guard_metrics, guard_policy, spectral_defaults
    )

    policy_out: dict[str, Any] | None = None
    if isinstance(guard_policy, dict) and guard_policy:
        policy_out = dict(guard_policy)
        _promote_legacy_multiple_testing_key(policy_out)
        if default_sigma_quantile is not None:
            sq = (
                policy_out.get("sigma_quantile")
                or policy_out.get("contraction")
                or policy_out.get("kappa")
            )
            if sq is not None:
                try:
                    policy_out["sigma_quantile"] = float(sq)
                except Exception:
                    pass
        policy_out.pop("contraction", None)
        policy_out.pop("kappa", None)
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
    # Attach status to summary for backward-compatibility in tests
    try:
        summary["status"] = "stable" if int(caps_applied) == 0 else "capped"
    except Exception:
        summary["status"] = "stable" if not caps_applied else "capped"
    if policy_out:
        result["policy"] = policy_out
    if default_sigma_quantile is not None:
        result["sigma_quantile"] = default_sigma_quantile
    if deadband_used is not None:
        result["deadband"] = deadband_used
    # Always include max_caps key for schema/tests parity
    max_caps_val = int(max_caps) if isinstance(max_caps, int | float) else None
    result["max_caps"] = max_caps_val
    try:
        summary["max_caps"] = max_caps_val
    except Exception:
        pass
    if multiple_testing:
        mt_copy = dict(multiple_testing)
        families_present = set((families or {}).keys()) or set(
            (family_caps or {}).keys()
        )
        try:
            mt_copy["m"] = int(mt_copy.get("m") or len(families_present))
        except Exception:
            mt_copy["m"] = len(families_present)
        result["multiple_testing"] = mt_copy
        result["bh_family_count"] = mt_copy["m"]

    # Additional derived fields for rendering/tests parity
    if families:
        caps_by_family = {
            fam: int(details.get("violations", 0))
            for fam, details in families.items()
            if isinstance(details, dict)
        }
        result["caps_applied_by_family"] = caps_by_family
    if top_z_scores:
        result["top_z_scores"] = top_z_scores
    # Top violations list from guard payload
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
            except Exception:
                pass
            top_violations.append(entry)
        if top_violations:
            result["top_violations"] = top_violations
    if family_quantiles:
        result["family_z_quantiles"] = family_quantiles
    result["caps_exceeded"] = bool(caps_exceeded)
    try:
        summary["caps_exceeded"] = bool(caps_exceeded)
    except Exception:
        pass
    # Propagate modules_checked when present
    if modules_checked is not None:
        try:
            summary["modules_checked"] = int(modules_checked)
        except Exception:
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


@no_type_check
def _extract_rmt_analysis(
    report: RunReport, baseline: dict[str, Any]
) -> dict[str, Any]:
    tier = _resolve_policy_tier(report)
    tier_policies = get_tier_policies()
    tier_defaults = tier_policies.get(tier, tier_policies.get("balanced", {}))
    default_epsilon_map = (
        tier_defaults.get("rmt", {}).get("epsilon_by_family")
        if isinstance(tier_defaults, dict)
        else {}
    )
    if not default_epsilon_map and isinstance(tier_defaults, dict):
        default_epsilon_map = (tier_defaults.get("rmt", {}) or {}).get("epsilon", {})
    default_epsilon_map = {
        str(family): float(value)
        for family, value in (default_epsilon_map or {}).items()
        if isinstance(value, int | float)
    }

    outliers_guarded = 0
    outliers_bare = 0
    epsilon_default = 0.1
    try:
        eps_def = (
            tier_defaults.get("rmt", {}).get("epsilon_default")
            if isinstance(tier_defaults, dict)
            else None
        )
        if isinstance(eps_def, int | float) and math.isfinite(float(eps_def)):
            epsilon_default = float(eps_def)
    except Exception:
        pass
    stable = True
    explicit_stability = False
    max_ratio = 0.0
    max_deviation_ratio = 1.0
    mean_deviation_ratio = 1.0
    epsilon_map: dict[str, float] = {}
    baseline_outliers_per_family: dict[str, int] = {}
    outliers_per_family: dict[str, int] = {}
    epsilon_violations: list[Any] = []
    margin_used = None
    deadband_used = None
    policy_out: dict[str, Any] | None = None

    for guard in report.get("guards", []) or []:
        if str(guard.get("name", "")).lower() == "rmt":
            guard_metrics = guard.get("metrics", {}) or {}
            guard_policy = guard.get("policy", {}) or {}
            if isinstance(guard_policy, dict) and guard_policy:
                policy_out = dict(guard_policy)
                if "epsilon_by_family" not in policy_out and isinstance(
                    policy_out.get("epsilon"), dict
                ):
                    policy_out["epsilon_by_family"] = dict(policy_out["epsilon"])
                if isinstance(policy_out.get("margin"), int | float) and math.isfinite(
                    float(policy_out.get("margin"))
                ):
                    margin_used = float(policy_out.get("margin"))
                if isinstance(
                    policy_out.get("deadband"), int | float
                ) and math.isfinite(float(policy_out.get("deadband"))):
                    deadband_used = float(policy_out.get("deadband"))
                if isinstance(
                    policy_out.get("epsilon_default"), int | float
                ) and math.isfinite(float(policy_out.get("epsilon_default"))):
                    epsilon_default = float(policy_out.get("epsilon_default"))
            if isinstance(
                guard_metrics.get("epsilon_default"), int | float
            ) and math.isfinite(float(guard_metrics.get("epsilon_default"))):
                epsilon_default = float(guard_metrics.get("epsilon_default"))
            outliers_guarded = guard_metrics.get(
                "rmt_outliers", guard_metrics.get("layers_flagged", outliers_guarded)
            )
            max_ratio = guard_metrics.get("max_ratio", 0.0)
            epsilon_map = guard_metrics.get("epsilon_by_family", {}) or epsilon_map
            if not epsilon_map and isinstance(guard_policy, dict):
                eps_src = guard_policy.get("epsilon_by_family") or guard_policy.get(
                    "epsilon"
                )
                if isinstance(eps_src, dict):
                    try:
                        epsilon_map = {
                            str(k): float(v)
                            for k, v in eps_src.items()
                            if isinstance(v, int | float) and math.isfinite(float(v))
                        }
                    except Exception:
                        pass
            baseline_outliers_per_family = (
                guard_metrics.get("baseline_outliers_per_family", {})
                or baseline_outliers_per_family
            )
            outliers_per_family = (
                guard_metrics.get("outliers_per_family", {}) or outliers_per_family
            )
            epsilon_violations = guard_metrics.get(
                "epsilon_violations", epsilon_violations
            )
            if outliers_per_family:
                outliers_guarded = sum(
                    int(v)
                    for v in outliers_per_family.values()
                    if isinstance(v, int | float)
                )
            if baseline_outliers_per_family:
                outliers_bare = sum(
                    int(v)
                    for v in baseline_outliers_per_family.values()
                    if isinstance(v, int | float)
                )
            flagged_rate = guard_metrics.get("flagged_rate", 0.0)
            stable = flagged_rate <= 0.5
            max_mp_ratio = guard_metrics.get("max_mp_ratio_final", 0.0)
            mean_mp_ratio = guard_metrics.get("mean_mp_ratio_final", 0.0)

            baseline_max = None
            baseline_mean = None
            baseline_rmt = baseline.get("rmt", {}) if isinstance(baseline, dict) else {}
            if baseline_rmt:
                baseline_max = baseline_rmt.get(
                    "max_mp_ratio", baseline_rmt.get("max_mp_ratio_final")
                )
                baseline_mean = baseline_rmt.get(
                    "mean_mp_ratio", baseline_rmt.get("mean_mp_ratio_final")
                )
                outliers_bare = baseline_rmt.get(
                    "outliers", baseline_rmt.get("rmt_outliers", 0)
                )
            if baseline_max is None:
                baseline_metrics = (
                    baseline.get("metrics", {}) if isinstance(baseline, dict) else {}
                )
                if "rmt" in baseline_metrics:
                    baseline_rmt_metrics = baseline_metrics["rmt"]
                    baseline_max = baseline_rmt_metrics.get("max_mp_ratio_final")
                    baseline_mean = baseline_rmt_metrics.get("mean_mp_ratio_final")
            if baseline_max is None and isinstance(guard.get("baseline_metrics"), dict):
                gb = guard.get("baseline_metrics")
                baseline_max = gb.get("max_mp_ratio")
                baseline_mean = gb.get("mean_mp_ratio")
            if baseline_max is not None and baseline_max > 0:
                max_deviation_ratio = max_mp_ratio / baseline_max
            else:
                max_deviation_ratio = 1.0
            if baseline_mean is not None and baseline_mean > 0:
                mean_deviation_ratio = mean_mp_ratio / baseline_mean
            else:
                mean_deviation_ratio = 1.0
            if isinstance(guard_metrics.get("stable"), bool):
                stable = bool(guard_metrics.get("stable"))
                explicit_stability = True
            break

    # Fallback: use metrics.rmt and/or top-level rmt section when guard is absent
    if outliers_guarded == 0:
        rmt_metrics = (report.get("metrics", {}) or {}).get("rmt", {})
        if isinstance(rmt_metrics, dict):
            try:
                outliers_guarded = int(rmt_metrics.get("outliers", 0) or 0)
            except Exception:
                outliers_guarded = 0
            if isinstance(rmt_metrics.get("stable"), bool):
                stable = bool(rmt_metrics.get("stable"))
                explicit_stability = True
        rmt_top = report.get("rmt", {}) if isinstance(report.get("rmt"), dict) else {}
        if isinstance(rmt_top, dict):
            fams = rmt_top.get("families", {})
            if isinstance(fams, dict) and fams:
                for fam, rec in fams.items():
                    if not isinstance(rec, dict):
                        continue
                    try:
                        outliers_per_family[str(fam)] = int(
                            rec.get("outliers_guarded", 0) or 0
                        )
                        baseline_outliers_per_family[str(fam)] = int(
                            rec.get("outliers_bare", 0) or 0
                        )
                        if rec.get("epsilon") is not None:
                            try:
                                epsilon_map[str(fam)] = float(rec.get("epsilon"))
                            except Exception:
                                pass
                    except Exception:
                        continue
            try:
                if outliers_bare == 0:
                    outliers_bare = int(rmt_top.get("outliers", 0) or 0)
            except Exception:
                pass

    # If stability not explicitly provided, derive from outlier behavior
    if not explicit_stability:
        try:
            if outliers_guarded == 0 and outliers_bare == 0:
                stable = True
            elif outliers_guarded <= outliers_bare:
                stable = True
            else:
                stable = (outliers_guarded - outliers_bare) / max(
                    outliers_bare, 1
                ) <= 0.5
        except Exception:
            pass

    delta_per_family = {
        k: int(outliers_per_family.get(k, 0))
        - int(baseline_outliers_per_family.get(k, 0))
        for k in set(outliers_per_family) | set(baseline_outliers_per_family)
    }
    delta_total = int(outliers_guarded) - int(outliers_bare)
    # Conservative baseline fallback when not available
    if outliers_bare == 0 and outliers_guarded > 0:
        # Assume baseline had fewer outliers to make acceptance harder
        outliers_bare = max(0, outliers_guarded - 1)

    # Recompute stability from epsilon rule when not explicitly provided
    if not explicit_stability:
        try:
            if outliers_per_family and baseline_outliers_per_family:
                families_union = set(outliers_per_family) | set(
                    baseline_outliers_per_family
                )
                checks: list[bool] = []
                for fam in families_union:
                    guarded = int(outliers_per_family.get(fam, 0) or 0)
                    bare = int(baseline_outliers_per_family.get(fam, 0) or 0)
                    eps_val = float(epsilon_map.get(fam, epsilon_default))
                    allowed = math.ceil(bare * (1.0 + eps_val))
                    checks.append(guarded <= allowed)
                if checks:
                    stable = all(checks)
            elif outliers_bare > 0:
                stable = outliers_guarded <= (
                    outliers_bare * (1.0 + float(epsilon_default))
                )
        except Exception:
            pass

    # Compute epsilon scalar (fallback) and detailed family breakdown
    if epsilon_map:
        epsilon_scalar = max(float(v) for v in epsilon_map.values())
    elif default_epsilon_map:
        try:
            epsilon_scalar = max(float(v) for v in default_epsilon_map.values())
        except Exception:
            epsilon_scalar = float(epsilon_default)
    else:
        epsilon_scalar = float(epsilon_default)
    try:
        epsilon_scalar = round(float(epsilon_scalar), 3)
    except Exception:
        epsilon_scalar = float(epsilon_default)

    def _to_int(v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    families = (
        set(outliers_per_family) | set(baseline_outliers_per_family) | set(epsilon_map)
    )
    family_breakdown = {
        family: {
            "bare": _to_int(baseline_outliers_per_family.get(family, 0)),
            "guarded": _to_int(outliers_per_family.get(family, 0)),
            "epsilon": float(epsilon_map.get(family, epsilon_scalar)),
        }
        for family in sorted(families)
    }

    # Stringify per-family dict keys for stability
    outliers_per_family = {str(k): _to_int(v) for k, v in outliers_per_family.items()}
    baseline_outliers_per_family = {
        str(k): _to_int(v) for k, v in baseline_outliers_per_family.items()
    }
    delta_per_family = {str(k): _to_int(v) for k, v in delta_per_family.items()}

    result = {
        "outliers_bare": outliers_bare,
        "outliers_guarded": outliers_guarded,
        "epsilon": epsilon_scalar,
        "epsilon_default": float(epsilon_default),
        "epsilon_by_family": epsilon_map,
        "outliers_per_family": outliers_per_family,
        "baseline_outliers_per_family": baseline_outliers_per_family,
        "delta_per_family": delta_per_family,
        "delta_total": delta_total,
        "epsilon_violations": epsilon_violations,
        "stable": stable,
        "status": "stable" if stable else "unstable",
        "max_ratio": max_ratio,
        "max_deviation_ratio": max_deviation_ratio,
        "mean_deviation_ratio": mean_deviation_ratio,
        "families": family_breakdown,
    }
    if margin_used is not None:
        result["margin"] = float(margin_used)
    if deadband_used is not None:
        result["deadband"] = float(deadband_used)
    if policy_out:
        result["policy"] = policy_out
    return result


@no_type_check
def _extract_variance_analysis(report: RunReport) -> dict[str, Any]:
    ve_enabled = False
    gain = None
    ppl_no_ve = None
    ppl_with_ve = None
    ratio_ci = None
    calibration = {}
    guard_metrics: dict[str, Any] = {}
    guard_policy: dict[str, Any] | None = None
    for guard in report.get("guards", []) or []:
        if "variance" in str(guard.get("name", "")).lower():
            metrics = guard.get("metrics", {}) or {}
            guard_metrics = metrics
            gp = guard.get("policy", {}) or {}
            if isinstance(gp, dict) and gp:
                guard_policy = dict(gp)
            ve_enabled = metrics.get("ve_enabled", bool(metrics))
            gain = metrics.get("ab_gain", metrics.get("gain", None))
            ppl_no_ve = metrics.get("ppl_no_ve", None)
            ppl_with_ve = metrics.get("ppl_with_ve", None)
            ratio_ci = metrics.get("ratio_ci", ratio_ci)
            calibration = metrics.get("calibration", calibration)
            break
    if gain is None:
        metrics_variance = (report.get("metrics", {}) or {}).get("variance", {})
        if isinstance(metrics_variance, dict):
            ve_enabled = metrics_variance.get("ve_enabled", ve_enabled)
            gain = metrics_variance.get("gain", gain)
            ppl_no_ve = metrics_variance.get("ppl_no_ve", ppl_no_ve)
            ppl_with_ve = metrics_variance.get("ppl_with_ve", ppl_with_ve)
            if not guard_metrics:
                guard_metrics = metrics_variance
    result = {"enabled": ve_enabled, "gain": gain}
    if ratio_ci:
        try:
            result["ratio_ci"] = (float(ratio_ci[0]), float(ratio_ci[1]))
        except Exception:
            pass
    if calibration:
        result["calibration"] = calibration
    if not ve_enabled and ppl_no_ve is not None and ppl_with_ve is not None:
        result["ppl_no_ve"] = ppl_no_ve
        result["ppl_with_ve"] = ppl_with_ve
    metadata_fields = [
        "tap",
        "target_modules",
        "target_module_names",
        "focus_modules",
        "scope",
        "proposed_scales",
        "proposed_scales_pre_edit",
        "proposed_scales_post_edit",
        "monitor_only",
        "max_calib_used",
        "mode",
        "min_rel_gain",
        "alpha",
    ]
    for field in metadata_fields:
        value = guard_metrics.get(field)
        if value not in (None, {}, []):
            result[field] = value
    predictive_gate = guard_metrics.get("predictive_gate")
    if predictive_gate:
        result["predictive_gate"] = predictive_gate
    ab_section: dict[str, Any] = {}
    if guard_metrics.get("ab_seed_used") is not None:
        ab_section["seed"] = guard_metrics["ab_seed_used"]
    if guard_metrics.get("ab_windows_used") is not None:
        ab_section["windows_used"] = guard_metrics["ab_windows_used"]
    if guard_metrics.get("ab_provenance"):
        prov = guard_metrics["ab_provenance"]
        if isinstance(prov, dict):
            prov_out = dict(prov)

            # Normalize a top-level `window_ids` list for docs + auditability.
            if "window_ids" not in prov_out:
                window_ids: set[int] = set()

                def _collect(node: Any) -> None:
                    if isinstance(node, dict):
                        ids = node.get("window_ids")
                        if isinstance(ids, list):
                            for wid in ids:
                                if isinstance(wid, int | float):
                                    window_ids.add(int(wid))
                        for v in node.values():
                            _collect(v)
                        return
                    if isinstance(node, list):
                        for v in node:
                            _collect(v)

                _collect(prov_out)
                if window_ids:
                    prov_out["window_ids"] = sorted(window_ids)

            ab_section["provenance"] = prov_out
        else:
            ab_section["provenance"] = prov
    if guard_metrics.get("ab_point_estimates"):
        ab_section["point_estimates"] = guard_metrics["ab_point_estimates"]
    if ab_section:
        result["ab_test"] = ab_section
    if guard_policy:
        result["policy"] = guard_policy
    return result


__all__ = [
    "_extract_invariants",
    "_extract_spectral_analysis",
    "_extract_rmt_analysis",
    "_extract_variance_analysis",
]
