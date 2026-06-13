"""Validation-flag computation for evaluation report generation."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from invarlock.core.metric_kind_contract import (
    MetricKindContractError,
    is_ppl_metric_kind,
    normalize_metric_kind,
)

GetTierPoliciesFn = Callable[[], dict[str, Any]]
_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _coerce_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    return coerced if math.isfinite(coerced) else None


def _is_non_bool_finite_number(value: Any) -> bool:
    return _coerce_finite_float(value) is not None


def _guard_overhead_has_error_diagnostic(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, list | tuple):
        return False
    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        if str(item.get("severity", "")).strip().lower() == "error":
            return True
    return False


def _resolve_drift_bounds(
    pm_drift_band: dict[str, float] | None,
    *,
    default: tuple[float, float],
) -> tuple[float, float]:
    drift_min, drift_max = default
    if not isinstance(pm_drift_band, dict):
        return drift_min, drift_max
    try:
        cand_min_f = _coerce_finite_float(pm_drift_band.get("min"))
        cand_max_f = _coerce_finite_float(pm_drift_band.get("max"))
        if (
            cand_min_f is not None
            and cand_max_f is not None
            and 0 < cand_min_f < cand_max_f
        ):
            return cand_min_f, cand_max_f
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass
    return drift_min, drift_max


def _resolve_effective_min_tokens(
    *,
    min_tokens: int,
    pm_policy: dict[str, Any],
    dataset_capacity: dict[str, Any] | None,
) -> int:
    eff_min_tokens = max(0, int(min_tokens))
    try:
        if isinstance(dataset_capacity, dict):
            frac = float(pm_policy.get("min_token_fraction", 0.0) or 0.0)
            avail_tokens_value = _coerce_finite_float(
                dataset_capacity.get("tokens_available")
            )
            if avail_tokens_value is not None and frac > 0.0:
                eff_min_tokens = max(
                    eff_min_tokens,
                    int(math.ceil(avail_tokens_value * frac)),
                )
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass
    return eff_min_tokens


def _coverage_meets_floor(_ppl_metrics: dict[str, Any]) -> bool:
    coverage_ok = False
    try:
        coverage = _ppl_metrics.get("bootstrap", {}).get("coverage")
        if not isinstance(coverage, dict):
            return False
        prev_cov = coverage.get("preview")
        fin_cov = coverage.get("final")
        if not (isinstance(prev_cov, dict) and isinstance(fin_cov, dict)):
            return False
        prev_used_value = _coerce_finite_float(prev_cov.get("used"))
        prev_req_value = _coerce_finite_float(prev_cov.get("required"))
        fin_used_value = _coerce_finite_float(fin_cov.get("used"))
        fin_req_value = _coerce_finite_float(fin_cov.get("required"))
        prev_ok = bool(prev_cov.get("ok")) or (
            prev_used_value is not None
            and prev_req_value is not None
            and prev_used_value >= prev_req_value
        )
        fin_ok = bool(fin_cov.get("ok")) or (
            fin_used_value is not None
            and fin_req_value is not None
            and fin_used_value >= fin_req_value
        )
        coverage_ok = prev_ok and fin_ok
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        coverage_ok = False
    return coverage_ok


def _resolve_tokens_ok(
    _ppl_metrics: dict[str, Any] | None,
    *,
    min_tokens: int,
    pm_policy: dict[str, Any],
    dataset_capacity: dict[str, Any] | None,
) -> bool:
    if not isinstance(_ppl_metrics, dict):
        return True
    pt_value = _coerce_finite_float(_ppl_metrics.get("preview_total_tokens"))
    ft_value = _coerce_finite_float(_ppl_metrics.get("final_total_tokens"))
    if pt_value is None or ft_value is None or min_tokens <= 0:
        return True
    try:
        total_tokens = int(pt_value) + int(ft_value)
        eff_min_tokens = _resolve_effective_min_tokens(
            min_tokens=min_tokens,
            pm_policy=pm_policy,
            dataset_capacity=dataset_capacity,
        )
        tokens_ok = total_tokens >= eff_min_tokens
        if tokens_ok or not _coverage_meets_floor(_ppl_metrics):
            return tokens_ok
        try:
            tolerance_ratio = float(pm_policy.get("min_tokens_tolerance", 0.02) or 0.0)
        except _NON_FATAL_EXCEPTIONS:
            tolerance_ratio = 0.0
        if tolerance_ratio < 0.0:
            tolerance_ratio = 0.0
        relaxed_floor = int(math.floor(float(eff_min_tokens) * (1.0 - tolerance_ratio)))
        return total_tokens >= max(relaxed_floor, 0)
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        return True


def _resolve_spectral_stable(
    spectral: dict[str, Any],
    *,
    tier_policy: dict[str, Any],
) -> bool:
    summary = spectral.get("summary", {}) if isinstance(spectral, dict) else {}
    max_caps = spectral.get("max_caps") or summary.get("max_caps")
    if max_caps is None:
        default_spectral = (
            tier_policy.get("spectral", {}) if isinstance(tier_policy, dict) else {}
        )
        max_caps = default_spectral.get("max_caps", 5)
    spectral_stable = bool(spectral.get("caps_applied", 0) <= int(max_caps))
    if spectral.get("caps_exceeded"):
        spectral_stable = False
    return spectral_stable


def _resolve_guard_overhead_pass(
    guard_overhead: dict[str, Any] | None,
    *,
    tiny_relax: bool,
) -> bool:
    if not (isinstance(guard_overhead, dict) and guard_overhead):
        return True
    if "passed" in guard_overhead:
        guard_overhead_pass = bool(guard_overhead.get("passed"))
        if tiny_relax and (
            not bool(guard_overhead.get("evaluated", True))
            or _guard_overhead_has_error_diagnostic(guard_overhead)
        ):
            return True
        return guard_overhead_pass
    ratio_val = _coerce_finite_float(guard_overhead.get("overhead_ratio"))
    threshold_val = _coerce_finite_float(guard_overhead.get("overhead_threshold", 0.01))
    if threshold_val is None:
        threshold_val = 0.01
    if tiny_relax and threshold_val < 0.10:
        threshold_val = 0.10
    if ratio_val is None:
        return True
    return ratio_val <= (1.0 + max(0.0, threshold_val))


def _apply_metric_specific_primary_metric_gate(
    flags: dict[str, bool],
    *,
    primary_metric: dict[str, Any] | None,
    metrics_policy: dict[str, Any],
    ratio_limit_with_hyst: float,
    tokens_ok_eff: bool,
    compression_acceptable: bool,
    tiny_relax: bool,
    dataset_capacity: dict[str, Any] | None,
) -> None:
    if not (isinstance(primary_metric, dict) and primary_metric):
        return
    try:
        kind = normalize_metric_kind(primary_metric.get("kind"))
    except (MetricKindContractError, ValueError):
        flags["primary_metric_acceptable"] = False
        return
    if kind is None:
        flags["primary_metric_acceptable"] = False
        return
    if is_ppl_metric_kind(kind):
        pm_ratio_value = _coerce_finite_float(primary_metric.get("ratio_vs_baseline"))
        if pm_ratio_value is not None:
            ok = (pm_ratio_value <= ratio_limit_with_hyst) and bool(tokens_ok_eff)
        else:
            ok = bool(compression_acceptable)
        flags["primary_metric_acceptable"] = bool(ok)
        return
    if kind != "accuracy":
        flags["primary_metric_acceptable"] = False
        return

    acc_policy = (
        metrics_policy.get("accuracy", {}) if isinstance(metrics_policy, dict) else {}
    )
    delta_min_pp = float(acc_policy.get("delta_min_pp", -1.0))
    min_examples = int(acc_policy.get("min_examples", 200))
    hysteresis_pp = float(acc_policy.get("hysteresis_delta_pp", 0.0))
    delta_value = _coerce_finite_float(primary_metric.get("ratio_vs_baseline"))
    meets_delta = delta_value is not None and (
        delta_value >= (delta_min_pp - max(0.0, hysteresis_pp))
    )
    if tiny_relax and delta_value is None:
        meets_delta = True

    meets_n = True
    n_fin_value = _coerce_finite_float(primary_metric.get("n_final"))
    if n_fin_value is not None:
        eff_min_examples = int(min_examples)
        try:
            if isinstance(dataset_capacity, dict):
                frac = float(acc_policy.get("min_examples_fraction", 0.0) or 0.0)
                avail_ex_value = _coerce_finite_float(
                    dataset_capacity.get("examples_available")
                )
                if avail_ex_value is not None and frac > 0.0:
                    eff_min_examples = max(
                        eff_min_examples,
                        int(math.ceil(avail_ex_value * frac)),
                    )
        except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pass
        meets_n = int(n_fin_value) >= eff_min_examples
        if tiny_relax:
            meets_n = True
    elif "n_final" in primary_metric:
        meets_n = False

    flags["primary_metric_acceptable"] = bool(meets_delta and meets_n)
    try:
        if delta_value is not None and delta_value < delta_min_pp and meets_delta:
            flags["hysteresis_applied"] = True
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass


def _apply_ppl_primary_metric_reconcile(
    flags: dict[str, bool],
    *,
    primary_metric: dict[str, Any] | None,
    ratio_limit: float,
    hysteresis_ratio: float,
    tokens_ok_eff: bool,
) -> None:
    try:
        if not (isinstance(primary_metric, dict) and primary_metric):
            return
        kind2 = normalize_metric_kind(primary_metric.get("kind"))
        if kind2 is None:
            flags["primary_metric_acceptable"] = False
            return
        if not is_ppl_metric_kind(kind2):
            return
        pmr_value = _coerce_finite_float(primary_metric.get("ratio_vs_baseline"))
        if (
            pmr_value is not None
            and pmr_value <= (ratio_limit + max(0.0, hysteresis_ratio))
            and bool(tokens_ok_eff)
        ):
            flags["primary_metric_acceptable"] = True
    except (MetricKindContractError, ValueError):
        flags["primary_metric_acceptable"] = False
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass


def _apply_optional_observability_flags(
    flags: dict[str, bool],
    *,
    moe: dict[str, Any] | None,
    pm_tail: dict[str, Any] | None,
) -> None:
    try:
        if isinstance(moe, dict) and moe:
            flags["moe_observed"] = True
            flags["moe_identity_ok"] = True
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass
    try:
        tail_ok = True
        if isinstance(pm_tail, dict) and pm_tail:
            mode = str(pm_tail.get("mode", "warn") or "warn").strip().lower()
            evaluated = bool(pm_tail.get("evaluated", False))
            passed = bool(pm_tail.get("passed", True))
            if mode == "fail" and evaluated and (not passed):
                tail_ok = False
        flags["primary_metric_tail_acceptable"] = bool(tail_ok)
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        flags["primary_metric_tail_acceptable"] = False


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
    # target_pm_ratio is an auto-tuning objective, not a report acceptance gate.
    # Keep gate evaluation anchored to the resolved tier / explicit acceptance range.

    # Canonical Gates
    # 1. Drift gate: by default 0.95 ≤ final/preview ≤ 1.05 (configurable)
    drift_ratio = _coerce_finite_float(ppl.get("preview_final_ratio", 1.0))
    drift_min, drift_max = _resolve_drift_bounds(
        pm_drift_band,
        default=pm_drift_band_default,
    )
    preview_final_drift_acceptable = (
        drift_ratio is not None and drift_min <= drift_ratio <= drift_max
    )
    if isinstance(primary_metric, dict):
        try:
            pm_kind = normalize_metric_kind(primary_metric.get("kind"))
        except (MetricKindContractError, ValueError):
            pm_kind = None
        if pm_kind == "accuracy":
            pm_preview = _coerce_finite_float(primary_metric.get("preview"))
            pm_final = _coerce_finite_float(primary_metric.get("final"))
            if pm_preview is not None and pm_final is not None:
                same_accuracy = math.isclose(
                    pm_preview,
                    pm_final,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                if same_accuracy:
                    preview_final_drift_acceptable = True
    if tiny_relax:
        # Treat drift identity as informational in tiny dev demos
        preview_final_drift_acceptable = True

    # 2. Primary metric vs baseline: edited/baseline ≤ tier threshold (ratio for ppl-like)
    ratio_vs_baseline = ppl.get("ratio_vs_baseline", 1.0)
    # Prefer primary_metric ratio when present
    if not _is_non_bool_finite_number(ratio_vs_baseline):
        try:
            pm_try = primary_metric if isinstance(primary_metric, dict) else {}
            pm_ratio = (
                pm_try.get("ratio_vs_baseline") if isinstance(pm_try, dict) else None
            )
            pm_ratio_value = _coerce_finite_float(pm_ratio)
            if pm_ratio_value is not None:
                ratio_vs_baseline = pm_ratio_value
        except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
            pass
    # Hysteresis and sample-size floors from tier policies
    hysteresis_ratio = float(pm_policy.get("hysteresis_ratio", 0.0))
    min_tokens = int(pm_policy.get("min_tokens", 0))
    # Evaluate sample-size sufficiency
    tokens_ok = _resolve_tokens_ok(
        _ppl_metrics,
        min_tokens=min_tokens,
        pm_policy=pm_policy,
        dataset_capacity=dataset_capacity,
    )
    # Under tiny_relax, treat token floors as informational only
    tokens_ok_eff = tokens_ok or tiny_relax
    # Apply hysteresis to ratio limit if needed
    ratio_limit_with_hyst = ratio_limit + max(0.0, hysteresis_ratio)
    lower_bound_ok = True
    if ratio_min_bound is not None and _is_non_bool_finite_number(ratio_vs_baseline):
        try:
            lower_bound_ok = float(ratio_vs_baseline) >= float(ratio_min_bound)
        except _NON_FATAL_EXCEPTIONS:
            lower_bound_ok = True
    compression_acceptable = (
        _is_non_bool_finite_number(ratio_vs_baseline)
        and lower_bound_ok
        and ratio_vs_baseline <= ratio_limit_with_hyst
        and tokens_ok_eff
    )
    if tiny_relax:
        # In tiny demos, allow undefined ratio and relax floors
        if not _is_non_bool_finite_number(ratio_vs_baseline):
            compression_acceptable = True
    ratio_ci = ppl.get("ratio_ci")
    if (
        isinstance(ratio_ci, tuple | list)
        and len(ratio_ci) == 2
        and all(_is_non_bool_finite_number(x) for x in ratio_ci)
    ):
        compression_acceptable = (
            compression_acceptable
            and ratio_ci[1] <= ratio_limit_with_hyst
            and (ratio_min_bound is None or ratio_ci[0] >= ratio_min_bound)
        )

    # 3. RMT ε-rule compliance
    rmt_stable = rmt.get("stable", True)

    effective_tier_policy = tier_policy if isinstance(tier_policy, dict) else {}
    spectral_stable = _resolve_spectral_stable(
        spectral,
        tier_policy=effective_tier_policy,
    )
    guard_overhead_pass = _resolve_guard_overhead_pass(
        guard_overhead,
        tiny_relax=tiny_relax,
    )

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
            _is_non_bool_finite_number(ratio_vs_baseline)
            and ratio_vs_baseline <= ratio_limit
        )
        if not base_ok and compression_acceptable:
            flags["hysteresis_applied"] = True
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass

    # Optional primary metric gating (metric-v1)
    try:
        _apply_metric_specific_primary_metric_gate(
            flags,
            primary_metric=primary_metric,
            metrics_policy=metrics_policy,
            ratio_limit_with_hyst=ratio_limit_with_hyst,
            tokens_ok_eff=tokens_ok_eff,
            compression_acceptable=compression_acceptable,
            tiny_relax=tiny_relax,
            dataset_capacity=dataset_capacity,
        )
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        # Fail-closed to False if something goes wrong
        flags["primary_metric_acceptable"] = False

    _apply_ppl_primary_metric_reconcile(
        flags,
        primary_metric=primary_metric,
        ratio_limit=ratio_limit,
        hysteresis_ratio=hysteresis_ratio,
        tokens_ok_eff=tokens_ok_eff,
    )
    _apply_optional_observability_flags(flags, moe=moe, pm_tail=pm_tail)

    return flags
