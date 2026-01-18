# mypy: ignore-errors
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, no_type_check

from invarlock.core.auto_tuning import get_tier_policies

from .policy_utils import _resolve_policy_tier
from .report_types import RunReport


def _measurement_contract_digest(contract: Any) -> str | None:
    if not isinstance(contract, dict) or not contract:
        return None
    try:
        canonical = json.dumps(contract, sort_keys=True, default=str)
    except Exception:
        return None
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@no_type_check
def _extract_invariants(
    report: RunReport, baseline: RunReport | None = None
) -> dict[str, Any]:
    """Extract invariant check results (matches the shape used in tests)."""
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

    baseline_guard_entry = None
    if baseline is not None:
        for guard in baseline.get("guards", []) or []:
            if str(guard.get("name", "")).lower() == "invariants":
                baseline_guard_entry = guard
                break

    def _coerce_checks(value: Any) -> dict[str, Any] | None:
        return value if isinstance(value, dict) else None

    def _extract_guard_checks(
        entry: Any,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not isinstance(entry, dict):
            return None, None
        details = entry.get("details")
        if not isinstance(details, dict):
            return None, None
        return _coerce_checks(details.get("baseline_checks")), _coerce_checks(
            details.get("current_checks")
        )

    def _compare_invariants(
        baseline_checks: dict[str, Any],
        current_checks: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int, int]:
        violations: list[dict[str, Any]] = []

        # LayerNorm coverage check
        baseline_layer_norms = set(baseline_checks.get("layer_norm_paths", ()))
        current_layer_norms = set(current_checks.get("layer_norm_paths", ()))
        missing_layer_norms = sorted(baseline_layer_norms - current_layer_norms)
        if missing_layer_norms:
            violations.append(
                {
                    "type": "layer_norm_missing",
                    "missing": missing_layer_norms,
                    "message": "Expected LayerNorm modules are missing vs baseline",
                }
            )

        # Tokenizer / vocab alignment
        baseline_vocab_sizes = baseline_checks.get("embedding_vocab_sizes")
        current_vocab_sizes = current_checks.get("embedding_vocab_sizes")
        if isinstance(baseline_vocab_sizes, dict):
            for module_name, baseline_size in baseline_vocab_sizes.items():
                current_size = None
                if isinstance(current_vocab_sizes, dict):
                    current_size = current_vocab_sizes.get(module_name)
                if current_size is None or int(current_size) != int(baseline_size):
                    mismatch = {
                        "module": module_name,
                        "baseline": int(baseline_size),
                        "current": None if current_size is None else int(current_size),
                    }
                    violations.append(
                        {
                            "type": "tokenizer_mismatch",
                            "message": "Embedding vocabulary size changed vs baseline",
                            **mismatch,
                        }
                    )

        handled_keys = {
            "layer_norm_paths",
            "embedding_vocab_sizes",
            "config_vocab_size",
        }
        for check_name, baseline_value in baseline_checks.items():
            if check_name in handled_keys:
                continue
            current_value = current_checks.get(check_name)
            if current_value != baseline_value:
                violations.append(
                    {
                        "type": "invariant_violation",
                        "check": check_name,
                        "baseline": baseline_value,
                        "current": current_value,
                        "message": (
                            f"Invariant {check_name} changed from {baseline_value} to {current_value}"
                        ),
                    }
                )

        fatal_violation_types = {"tokenizer_mismatch"}
        fatal_count = 0
        warning_count = 0
        annotated: list[dict[str, Any]] = []
        for violation in violations:
            violation_type = str(violation.get("type") or "")
            severity = "fatal" if violation_type in fatal_violation_types else "warning"
            annotated_violation = dict(violation)
            annotated_violation.setdefault("severity", severity)
            annotated.append(annotated_violation)
            if severity == "fatal":
                fatal_count += 1
            else:
                warning_count += 1

        return annotated, fatal_count, warning_count

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
        base_fatal = 0
        base_warn = 0
        baseline_failures: list[dict[str, Any]] = []
        if baseline_guard_entry is not None:
            baseline_pre, baseline_post = _extract_guard_checks(baseline_guard_entry)
            current_pre, current_post = _extract_guard_checks(guard_entry)
            baseline_snapshot = baseline_pre or baseline_post
            current_snapshot = current_post or current_pre
            if isinstance(baseline_snapshot, dict) and isinstance(
                current_snapshot, dict
            ):
                baseline_failures, base_fatal, base_warn = _compare_invariants(
                    baseline_snapshot, current_snapshot
                )
                for violation in baseline_failures:
                    check_name = violation.get("check")
                    if not check_name:
                        check_name = (
                            violation.get("module")
                            or violation.get("type")
                            or "invariant"
                        )
                    row = {
                        "check": str(check_name),
                        "type": str(violation.get("type") or "violation"),
                        "severity": str(violation.get("severity") or "warning"),
                    }
                    detail = {k: v for k, v in violation.items() if k not in row}
                    if detail:
                        detail.setdefault("source", "baseline_compare")
                        row["detail"] = detail
                    failures.append(row)

        fatal_total = fatal_count + base_fatal
        warn_total = warning_count + base_warn
        try:
            summary["fatal_violations"] = fatal_total
            summary["warning_violations"] = warn_total
            summary["violations_found"] = fatal_total + warn_total
        except Exception:
            pass

        if fatal_total > 0:
            severity_status = "fail"
        elif warn_total > 0 or violations:
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

    details_out = invariants_data
    if not details_out and guard_entry and isinstance(guard_entry.get("details"), dict):
        details_out = guard_entry.get("details", {})

    return {
        "pre": "pass",
        "post": status,
        "status": status,
        "summary": summary,
        "details": details_out,
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
    deadband_used: float | None = None

    if isinstance(guard_metrics, dict):
        # Resolve deadband from policy/metrics/defaults
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

        # Resolve sigma_quantile for summary
        sigma_q_used: float | None = None
        try:
            pol_sq = None
            if isinstance(guard_policy, dict):
                pol_sq = guard_policy.get("sigma_quantile")
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
            # Prefer z-summary when available; accept 'family_stats' too
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
                except Exception:
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
    # Surface a stable/capped status on the summary for schema parity.
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
    except Exception:
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
    except Exception:
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
    """Extract RMT analysis using activation edge-risk ε-band semantics."""
    tier = _resolve_policy_tier(report)
    tier_policies = get_tier_policies()
    tier_defaults = tier_policies.get(tier, tier_policies.get("balanced", {}))

    default_epsilon_map = (
        tier_defaults.get("rmt", {}).get("epsilon_by_family")
        if isinstance(tier_defaults, dict)
        else {}
    )
    default_epsilon_map = {
        str(family): float(value)
        for family, value in (default_epsilon_map or {}).items()
        if isinstance(value, int | float) and math.isfinite(float(value))
    }

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

    baseline_rmt = baseline.get("rmt", {}) if isinstance(baseline, dict) else {}
    baseline_edge_by_family: dict[str, float] = {}
    baseline_contract = None
    if isinstance(baseline_rmt, dict) and baseline_rmt:
        bc = baseline_rmt.get("measurement_contract")
        if isinstance(bc, dict) and bc:
            baseline_contract = bc
        base = baseline_rmt.get("edge_risk_by_family") or baseline_rmt.get(
            "edge_risk_by_family_base"
        )
        if isinstance(base, dict):
            for k, v in base.items():
                if isinstance(v, int | float) and math.isfinite(float(v)):
                    baseline_edge_by_family[str(k)] = float(v)

    rmt_guard = None
    guard_metrics: dict[str, Any] = {}
    guard_policy: dict[str, Any] = {}
    for guard in report.get("guards", []) or []:
        if str(guard.get("name", "")).lower() == "rmt":
            rmt_guard = guard
            guard_metrics = guard.get("metrics", {}) or {}
            guard_policy = guard.get("policy", {}) or {}
            break

    policy_out: dict[str, Any] | None = None
    if isinstance(guard_policy, dict) and guard_policy:
        policy_out = dict(guard_policy)
        if isinstance(policy_out.get("epsilon_default"), int | float) and math.isfinite(
            float(policy_out.get("epsilon_default"))
        ):
            epsilon_default = float(policy_out.get("epsilon_default"))

    if isinstance(guard_metrics.get("epsilon_default"), int | float) and math.isfinite(
        float(guard_metrics.get("epsilon_default"))
    ):
        epsilon_default = float(guard_metrics.get("epsilon_default"))

    edge_base: dict[str, float] = {}
    edge_cur: dict[str, float] = {}
    if isinstance(guard_metrics, dict) and guard_metrics:
        base = guard_metrics.get("edge_risk_by_family_base") or {}
        cur = guard_metrics.get("edge_risk_by_family") or {}
        if isinstance(base, dict):
            for k, v in base.items():
                if isinstance(v, int | float) and math.isfinite(float(v)):
                    edge_base[str(k)] = float(v)
        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(v, int | float) and math.isfinite(float(v)):
                    edge_cur[str(k)] = float(v)
    if not edge_base and baseline_edge_by_family:
        edge_base = dict(baseline_edge_by_family)

    epsilon_map: dict[str, float] = {}
    eps_src = guard_metrics.get("epsilon_by_family") or {}
    if not eps_src and isinstance(guard_policy, dict):
        eps_src = guard_policy.get("epsilon_by_family") or {}
    if isinstance(eps_src, dict):
        for k, v in eps_src.items():
            if isinstance(v, int | float) and math.isfinite(float(v)):
                epsilon_map[str(k)] = float(v)

    epsilon_violations = guard_metrics.get("epsilon_violations") or []
    if not (isinstance(epsilon_violations, list) and epsilon_violations):
        epsilon_violations = []
        families = set(edge_cur) | set(edge_base)
        for family in families:
            base = float(edge_base.get(family, 0.0) or 0.0)
            cur = float(edge_cur.get(family, 0.0) or 0.0)
            if base <= 0.0:
                continue
            eps = float(
                epsilon_map.get(
                    family, default_epsilon_map.get(family, epsilon_default)
                )
            )
            allowed = (1.0 + eps) * base
            if cur > allowed:
                delta = (cur / base) - 1.0 if base > 0 else float("inf")
                epsilon_violations.append(
                    {
                        "family": family,
                        "edge_base": base,
                        "edge_cur": cur,
                        "delta": float(delta),
                        "allowed": allowed,
                        "epsilon": eps,
                    }
                )

    stable = bool(guard_metrics.get("stable", not epsilon_violations))

    families_all = sorted(
        set(edge_base) | set(edge_cur) | set(epsilon_map) | set(default_epsilon_map)
    )
    family_breakdown: dict[str, dict[str, Any]] = {}
    ratios: list[float] = []
    deltas: list[float] = []
    for family in families_all:
        base = float(edge_base.get(family, 0.0) or 0.0)
        cur = float(edge_cur.get(family, 0.0) or 0.0)
        eps = float(
            epsilon_map.get(family, default_epsilon_map.get(family, epsilon_default))
        )
        allowed = (1.0 + eps) * base if base > 0.0 else None
        ratio = (cur / base) if base > 0.0 else None
        delta = ((cur / base) - 1.0) if base > 0.0 else None
        if isinstance(ratio, float) and math.isfinite(ratio):
            ratios.append(ratio)
        if isinstance(delta, float) and math.isfinite(delta):
            deltas.append(delta)
        family_breakdown[family] = {
            "edge_base": base,
            "edge_cur": cur,
            "epsilon": eps,
            "allowed": allowed,
            "ratio": ratio,
            "delta": delta,
        }

    measurement_contract = None
    try:
        mc = (
            guard_metrics.get("measurement_contract")
            if isinstance(guard_metrics, dict)
            else None
        )
        if isinstance(mc, dict) and mc:
            measurement_contract = mc
    except Exception:
        measurement_contract = None

    mc_hash = _measurement_contract_digest(measurement_contract)
    baseline_hash = _measurement_contract_digest(baseline_contract)

    result: dict[str, Any] = {
        "tier": tier,
        "edge_risk_by_family_base": dict(edge_base),
        "edge_risk_by_family": dict(edge_cur),
        "epsilon_default": float(epsilon_default),
        "epsilon_by_family": dict(epsilon_map),
        "epsilon_violations": list(epsilon_violations),
        "stable": stable,
        "status": "stable" if stable else "unstable",
        "max_edge_ratio": max(ratios) if ratios else None,
        "max_edge_delta": max(deltas) if deltas else None,
        "mean_edge_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "families": family_breakdown,
        "evaluated": bool(rmt_guard),
    }
    if policy_out:
        result["policy"] = policy_out
    if measurement_contract is not None:
        result["measurement_contract"] = measurement_contract
    if mc_hash:
        result["measurement_contract_hash"] = mc_hash
    if baseline_hash:
        result["baseline_measurement_contract_hash"] = baseline_hash
    if mc_hash and baseline_hash:
        result["measurement_contract_match"] = bool(mc_hash == baseline_hash)
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
