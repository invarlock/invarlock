"""Validation-flag computation extracted from report_builder."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

GetTierPoliciesFn = Callable[[], dict[str, Any]]


def _coerce_finite_float(value: Any) -> float | None:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    return coerced if math.isfinite(coerced) else None


def compute_validation_flags(
    ppl: dict[str, Any],
    spectral: dict[str, Any],
    rmt: dict[str, Any],
    invariants: dict[str, Any],
    tier: str = "balanced",
    _ppl_metrics: dict[str, Any] | None = None,
    target_ratio: float | None = None,
    guard_overhead: dict[str, Any] | None = None,
    primary_metric: dict[str, Any] | None = None,
    moe: dict[str, Any] | None = None,
    dataset_capacity: dict[str, Any] | None = None,
    pm_acceptance_range: dict[str, float] | None = None,
    pm_drift_band: dict[str, float] | None = None,
    pm_tail: dict[str, Any] | None = None,
    tiny_relax: bool = False,
    *,
    pm_drift_band_default: tuple[float, float] = (0.95, 1.05),
    get_tier_policies_fn: GetTierPoliciesFn | None = None,
) -> dict[str, bool]:
    """Compute validation flags for the evaluation report including canonical gates."""

    if get_tier_policies_fn is None:
        from invarlock.core.auto_tuning import get_tier_policies

        tier_policies_fn: GetTierPoliciesFn = get_tier_policies
    else:
        tier_policies_fn = get_tier_policies_fn

    tier = (tier or "balanced").lower()
    if not tiny_relax:
        tiny_relax = str(
            os.environ.get("INVARLOCK_TINY_RELAX", "")
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    # Tiny-relax is policy/config-driven and applies only when explicitly set.
    if tiny_relax:
        tier = "aggressive"

    tier_thresholds = {
        "conservative": 1.05,
        "balanced": 1.10,
        "aggressive": 1.20,
        "none": 1.10,
    }
    tier_policies = tier_policies_fn()
    tier_policy = tier_policies.get(tier, tier_policies.get("balanced", {}))
    metrics_policy = (
        tier_policy.get("metrics", {}) if isinstance(tier_policy, dict) else {}
    )
    pm_policy = (
        metrics_policy.get("pm_ratio", {}) if isinstance(metrics_policy, dict) else {}
    )
    ratio_limit_base = _coerce_finite_float(pm_policy.get("ratio_limit_base"))
    if ratio_limit_base is None:
        ratio_limit_base = float(tier_thresholds.get(tier, 1.10))
    acceptance = pm_acceptance_range if isinstance(pm_acceptance_range, dict) else {}
    ratio_min_bound = None
    ratio_max_bound = None
    acceptance_min = _coerce_finite_float(acceptance.get("min"))
    if acceptance_min is not None:
        ratio_min_bound = acceptance_min
    acceptance_max = _coerce_finite_float(acceptance.get("max"))
    if acceptance_max is not None:
        ratio_max_bound = acceptance_max

    ratio_limit = (
        ratio_max_bound if ratio_max_bound is not None else float(ratio_limit_base)
    )
    if isinstance(target_ratio, int | float) and target_ratio > 0:
        ratio_limit = min(ratio_limit, float(target_ratio))

    # Canonical Gates
    # 1. Drift gate: by default 0.95 ≤ final/preview ≤ 1.05 (configurable)
    drift_ratio = ppl.get("preview_final_ratio", 1.0)
    drift_min, drift_max = pm_drift_band_default
    if isinstance(pm_drift_band, dict):
        try:
            cand_min = pm_drift_band.get("min")
            cand_max = pm_drift_band.get("max")
            if isinstance(cand_min, int | float) and isinstance(cand_max, int | float):
                cand_min_f = float(cand_min)
                cand_max_f = float(cand_max)
                if (
                    math.isfinite(cand_min_f)
                    and math.isfinite(cand_max_f)
                    and 0 < cand_min_f < cand_max_f
                ):
                    drift_min = cand_min_f
                    drift_max = cand_max_f
        except Exception:  # pragma: no cover
            pass
    preview_final_drift_acceptable = drift_min <= drift_ratio <= drift_max
    if tiny_relax:
        # Treat drift identity as informational in tiny dev demos
        preview_final_drift_acceptable = True

    # 2. Primary metric vs baseline: edited/baseline ≤ tier threshold (ratio for ppl-like)
    ratio_vs_baseline = ppl.get("ratio_vs_baseline", 1.0)
    # Prefer primary_metric ratio when present
    if not (
        isinstance(ratio_vs_baseline, int | float) and math.isfinite(ratio_vs_baseline)
    ):
        try:
            pm_try = primary_metric if isinstance(primary_metric, dict) else {}
            pm_ratio = (
                pm_try.get("ratio_vs_baseline") if isinstance(pm_try, dict) else None
            )
            if isinstance(pm_ratio, int | float) and math.isfinite(pm_ratio):
                ratio_vs_baseline = float(pm_ratio)
        except Exception:  # pragma: no cover
            pass
    # Hysteresis and sample-size floors from tier policies
    hysteresis_ratio = float(pm_policy.get("hysteresis_ratio", 0.0))
    min_tokens = int(pm_policy.get("min_tokens", 0))
    # Evaluate sample-size sufficiency
    tokens_ok = True
    if isinstance(_ppl_metrics, dict):
        pt = _ppl_metrics.get("preview_total_tokens")
        ft = _ppl_metrics.get("final_total_tokens")
        pt_value = _coerce_finite_float(pt)
        ft_value = _coerce_finite_float(ft)
        if pt_value is not None and ft_value is not None and min_tokens > 0:
            try:
                total_tokens = int(pt_value) + int(ft_value)
                # Dataset-scale aware floors: use fraction of available tokens when provided
                eff_min_tokens = max(0, int(min_tokens))
                try:
                    if isinstance(dataset_capacity, dict):
                        frac = float(pm_policy.get("min_token_fraction", 0.0) or 0.0)
                        avail_tokens = dataset_capacity.get("tokens_available")
                        if isinstance(avail_tokens, int | float) and frac > 0.0:
                            eff_min_tokens = max(
                                eff_min_tokens,
                                int(math.ceil(float(avail_tokens) * frac)),
                            )
                except Exception:  # pragma: no cover
                    pass
                tokens_ok = total_tokens >= eff_min_tokens
                if not tokens_ok:
                    coverage_ok = False
                    try:
                        coverage = _ppl_metrics.get("bootstrap", {}).get("coverage")
                        if isinstance(coverage, dict):
                            prev_cov = coverage.get("preview")
                            fin_cov = coverage.get("final")
                            if isinstance(prev_cov, dict) and isinstance(fin_cov, dict):
                                prev_used = prev_cov.get("used")
                                prev_req = prev_cov.get("required")
                                fin_used = fin_cov.get("used")
                                fin_req = fin_cov.get("required")
                                prev_ok = bool(prev_cov.get("ok")) or (
                                    isinstance(prev_used, int | float)
                                    and isinstance(prev_req, int | float)
                                    and float(prev_used) >= float(prev_req)
                                )
                                fin_ok = bool(fin_cov.get("ok")) or (
                                    isinstance(fin_used, int | float)
                                    and isinstance(fin_req, int | float)
                                    and float(fin_used) >= float(fin_req)
                                )
                                coverage_ok = prev_ok and fin_ok
                    except Exception:  # pragma: no cover
                        coverage_ok = False

                    if coverage_ok:
                        try:
                            tolerance_ratio = float(
                                pm_policy.get("min_tokens_tolerance", 0.02) or 0.0
                            )
                        except Exception:
                            tolerance_ratio = 0.0
                        if tolerance_ratio < 0.0:
                            tolerance_ratio = 0.0
                        relaxed_floor = int(
                            math.floor(float(eff_min_tokens) * (1.0 - tolerance_ratio))
                        )
                        tokens_ok = total_tokens >= max(relaxed_floor, 0)
            except Exception:  # pragma: no cover
                tokens_ok = True
    # Under tiny_relax, treat token floors as informational only
    tokens_ok_eff = tokens_ok or tiny_relax
    # Apply hysteresis to ratio limit if needed
    ratio_limit_with_hyst = ratio_limit + max(0.0, hysteresis_ratio)
    lower_bound_ok = True
    if ratio_min_bound is not None and isinstance(ratio_vs_baseline, (int | float)):
        try:
            lower_bound_ok = math.isfinite(float(ratio_vs_baseline)) and (
                float(ratio_vs_baseline) >= float(ratio_min_bound)
            )
        except Exception:
            lower_bound_ok = True
    compression_acceptable = (
        isinstance(ratio_vs_baseline, int | float)
        and math.isfinite(ratio_vs_baseline)
        and lower_bound_ok
        and ratio_vs_baseline <= ratio_limit_with_hyst
        and tokens_ok_eff
    )
    if tiny_relax:
        # In tiny demos, allow undefined ratio and relax floors
        if not isinstance(ratio_vs_baseline, int | float) or not math.isfinite(
            ratio_vs_baseline
        ):
            compression_acceptable = True
    ratio_ci = ppl.get("ratio_ci")
    if (
        isinstance(ratio_ci, tuple | list)
        and len(ratio_ci) == 2
        and all(isinstance(x, int | float) and math.isfinite(x) for x in ratio_ci)
    ):
        compression_acceptable = (
            compression_acceptable
            and ratio_ci[1] <= ratio_limit_with_hyst
            and (ratio_min_bound is None or ratio_ci[0] >= ratio_min_bound)
        )

    # 3. RMT ε-rule compliance
    rmt_stable = rmt.get("stable", True)

    summary = spectral.get("summary", {}) if isinstance(spectral, dict) else {}
    max_caps = spectral.get("max_caps") or summary.get("max_caps")
    if max_caps is None:
        default_spectral = (
            tier_policy.get("spectral", {}) if isinstance(tier_policy, dict) else {}
        )
        max_caps = default_spectral.get("max_caps", 5)
    spectral_stable = spectral.get("caps_applied", 0) <= int(max_caps)
    if spectral.get("caps_exceeded"):
        spectral_stable = False

    guard_overhead_pass = True
    if isinstance(guard_overhead, dict) and guard_overhead:
        if "passed" in guard_overhead:
            guard_overhead_pass = bool(guard_overhead.get("passed"))
            if tiny_relax and (
                not bool(guard_overhead.get("evaluated", True))
                or guard_overhead.get("errors")
            ):
                guard_overhead_pass = True
        else:
            ratio = guard_overhead.get("overhead_ratio")
            threshold = guard_overhead.get("overhead_threshold", 0.01)
            ratio_val = _coerce_finite_float(ratio)
            threshold_val = _coerce_finite_float(threshold)
            if threshold_val is None:
                threshold_val = 0.01
            if tiny_relax and threshold_val < 0.10:
                threshold_val = 0.10
            if ratio_val is None:
                # In dev/Compare-&-Evaluate flows we often lack a bare run; treat missing metric as pass
                guard_overhead_pass = True
            else:
                guard_overhead_pass = ratio_val <= (1.0 + max(0.0, threshold_val))

    flags = {
        "preview_final_drift_acceptable": preview_final_drift_acceptable,
        "primary_metric_acceptable": compression_acceptable,
        "invariants_pass": invariants.get("status") not in {"fail", "error"},
        "spectral_stable": spectral_stable,
        "rmt_stable": rmt_stable,  # RMT ε-rule compliance
        "guard_overhead_acceptable": guard_overhead_pass,
    }
    # Mark hysteresis application when ratio exceeds base limit but passes with hysteresis
    try:
        base_ok = (
            isinstance(ratio_vs_baseline, int | float)
            and math.isfinite(ratio_vs_baseline)
            and ratio_vs_baseline <= ratio_limit
        )
        if not base_ok and compression_acceptable:
            flags["hysteresis_applied"] = True
    except Exception:  # pragma: no cover
        pass

    # Optional primary metric gating (metric-v1)
    try:
        if isinstance(primary_metric, dict) and primary_metric:
            kind = str(primary_metric.get("kind", "")).lower()
            if kind in {"ppl_causal", "ppl_mlm", "ppl_seq2seq"}:
                # Apply the same hysteresis and sample-size floors as primary_metric_acceptable
                pm_ratio = primary_metric.get("ratio_vs_baseline")
                if isinstance(pm_ratio, int | float) and math.isfinite(pm_ratio):
                    ok = (pm_ratio <= ratio_limit_with_hyst) and bool(tokens_ok_eff)
                else:
                    # Fall back to compression_acceptable when PM ratio is unavailable
                    ok = bool(compression_acceptable)
                flags["primary_metric_acceptable"] = bool(ok)
            elif kind in {"accuracy", "vqa_accuracy"}:
                # Read thresholds from tier policy if available
                acc_policy = (
                    metrics_policy.get("accuracy", {})
                    if isinstance(metrics_policy, dict)
                    else {}
                )
                delta_min_pp = float(acc_policy.get("delta_min_pp", -1.0))
                min_examples = int(acc_policy.get("min_examples", 200))
                hysteresis_pp = float(acc_policy.get("hysteresis_delta_pp", 0.0))
                delta = primary_metric.get("ratio_vs_baseline")
                meets_delta = (
                    isinstance(delta, int | float)
                    and math.isfinite(delta)
                    and (delta >= (delta_min_pp - max(0.0, hysteresis_pp)))
                )
                if tiny_relax and not (
                    isinstance(delta, int | float) and math.isfinite(delta)
                ):
                    meets_delta = True
                n_fin = primary_metric.get("n_final")
                meets_n = True
                if isinstance(n_fin, int | float):
                    # Dataset-scale aware min_examples when available
                    eff_min_examples = int(min_examples)
                    try:
                        if isinstance(dataset_capacity, dict):
                            frac = float(
                                acc_policy.get("min_examples_fraction", 0.0) or 0.0
                            )
                            avail_ex = dataset_capacity.get("examples_available")
                            if isinstance(avail_ex, int | float) and frac > 0.0:
                                eff_min_examples = max(
                                    eff_min_examples,
                                    int(math.ceil(float(avail_ex) * frac)),
                                )
                    except Exception:  # pragma: no cover
                        pass
                    meets_n = int(n_fin) >= eff_min_examples
                    if tiny_relax:
                        # In tiny demos accept smaller sample sizes
                        meets_n = True
                flags["primary_metric_acceptable"] = bool(meets_delta and meets_n)
                try:
                    if (
                        isinstance(delta, int | float)
                        and delta < delta_min_pp
                        and meets_delta
                    ):
                        flags["hysteresis_applied"] = True
                except Exception:  # pragma: no cover
                    pass
    except Exception:  # pragma: no cover
        # Fail-closed to False if something goes wrong
        flags["primary_metric_acceptable"] = False

    # Reconcile: if ppl-like primary_metric ratio is present and within hysteresis-adjusted
    # limit, prefer that decision to avoid spurious FAILs from upstream fallbacks.
    try:
        if isinstance(primary_metric, dict) and primary_metric:
            kind2 = str(primary_metric.get("kind", "")).lower()
            if kind2 in {"ppl_causal", "ppl_mlm", "ppl_seq2seq"}:
                pmr = primary_metric.get("ratio_vs_baseline")
                if (
                    isinstance(pmr, int | float)
                    and math.isfinite(float(pmr))
                    and float(pmr) <= (ratio_limit + max(0.0, hysteresis_ratio))
                    and bool(tokens_ok_eff)
                ):
                    flags["primary_metric_acceptable"] = True
    except Exception:  # pragma: no cover
        pass

    # MoE observability flags (non-gating)
    try:
        if isinstance(moe, dict) and moe:
            flags["moe_observed"] = True
            flags["moe_identity_ok"] = True
    except Exception:  # pragma: no cover
        pass

    # Primary metric tail gate (warn/fail; default non-blocking)
    try:
        tail_ok = True
        if isinstance(pm_tail, dict) and pm_tail:
            mode = str(pm_tail.get("mode", "warn") or "warn").strip().lower()
            evaluated = bool(pm_tail.get("evaluated", False))
            passed = bool(pm_tail.get("passed", True))
            if mode == "fail" and evaluated and (not passed):
                tail_ok = False
        flags["primary_metric_tail_acceptable"] = bool(tail_ok)
    except Exception:  # pragma: no cover
        flags["primary_metric_tail_acceptable"] = True

    return flags
