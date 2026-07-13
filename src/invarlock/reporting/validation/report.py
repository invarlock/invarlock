"""Validation-flag computation for evaluation report generation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from invarlock.core.metric_kind_contract import (
    MetricKindContractError,
    is_ppl_metric_kind,
    normalize_metric_kind,
)
from invarlock.primary_metric_tail import (
    PrimaryMetricTailContractError,
    require_primary_metric_tail,
)

from . import metric_impact as _metric_impact
from .guard_flags import (
    resolve_invariants_pass,
    resolve_rmt_stable,
    resolve_spectral_stable,
)
from .thresholds import resolve_drift_bounds, resolve_tokens_ok

GetTierPoliciesFn = Callable[[], dict[str, Any]]
_guard_metric_impact_has_error_diagnostic = (
    _metric_impact.guard_metric_impact_has_error_diagnostic
)
_resolve_guard_metric_impact_pass = _metric_impact.resolve_guard_metric_impact_pass
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


def _resolve_preview_drift(
    *,
    ppl: dict[str, Any],
    primary_metric: dict[str, Any] | None,
    metrics_policy: object,
    drift_min: float,
    drift_max: float,
    tiny_relax: bool,
) -> bool:
    drift_ratio = _coerce_finite_float(ppl.get("preview_final_ratio"))
    acceptable = drift_ratio is not None and drift_min <= drift_ratio <= drift_max
    evidence_present = drift_ratio is not None
    if isinstance(primary_metric, dict):
        try:
            pm_kind = normalize_metric_kind(primary_metric.get("kind"))
        except (MetricKindContractError, ValueError):
            pm_kind = None
        pm_preview = _coerce_finite_float(primary_metric.get("preview"))
        pm_final = _coerce_finite_float(primary_metric.get("final"))
        if pm_kind in {"ppl_causal", "ppl_mlm", "ppl_seq2seq"}:
            if pm_preview is not None and pm_preview > 0.0 and pm_final is not None:
                evidence_present = True
                acceptable = drift_min <= pm_final / pm_preview <= drift_max
        elif pm_kind == "accuracy" and pm_preview is not None and pm_final is not None:
            evidence_present = True
            accuracy_policy = (
                metrics_policy.get("accuracy", {})
                if isinstance(metrics_policy, dict)
                else {}
            )
            limit = _coerce_finite_float(
                accuracy_policy.get("preview_final_delta_pp_max")
            )
            if limit is None:
                limit = _coerce_finite_float(accuracy_policy.get("hysteresis_delta_pp"))
            acceptable = abs(pm_final - pm_preview) <= max(
                0.0, 0.1 if limit is None else limit
            )
    return True if tiny_relax and evidence_present else acceptable


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
    delta_value = _coerce_finite_float(primary_metric.get("delta_vs_baseline_pp"))
    meets_delta = delta_value is not None and (
        delta_value >= (delta_min_pp - max(0.0, hysteresis_pp))
    )
    # Accuracy evidence is only meaningful when the evaluated-example count is
    # explicit.  Treat a missing count as a failed gate; otherwise a submitted
    # report could omit ``n_final`` and silently bypass the policy floor.
    meets_n = False
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
    ratio_min_bound: float | None = None,
    ratio_ci_pass: bool = True,
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
        if pmr_value is None:
            # The normalized PPL analysis may already contain the canonical
            # baseline ratio even when the source RunReport did not persist it.
            return
        # Reconciliation may confirm a passing point estimate, but it must not
        # erase a failure from the canonical CI or acceptance-range checks.
        flags["primary_metric_acceptable"] = bool(
            pmr_value is not None
            and (ratio_min_bound is None or pmr_value >= ratio_min_bound)
            and pmr_value <= (ratio_limit + max(0.0, hysteresis_ratio))
            and bool(tokens_ok_eff)
            and ratio_ci_pass
        )
    except (MetricKindContractError, ValueError):
        flags["primary_metric_acceptable"] = False
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass


def _apply_optional_observability_flags(
    flags: dict[str, bool],
    *,
    moe: dict[str, Any] | None,
    pm_tail: dict[str, Any] | None,
    tail_required: bool,
) -> None:
    try:
        if isinstance(moe, dict) and moe:
            flags["moe_observed"] = True
            flags["moe_identity_ok"] = True
    except _NON_FATAL_EXCEPTIONS:  # pragma: no cover
        pass
    if pm_tail is None:
        flags["primary_metric_tail_acceptable"] = not tail_required
        return
    try:
        outcome = require_primary_metric_tail(pm_tail)
        flags["primary_metric_tail_acceptable"] = outcome.acceptable
    except (PrimaryMetricTailContractError, *_NON_FATAL_EXCEPTIONS):
        flags["primary_metric_tail_acceptable"] = False


def compute_validation_flags(
    ppl: dict[str, Any],
    spectral: dict[str, Any],
    rmt: dict[str, Any],
    invariants: dict[str, Any],
    tier: str = "balanced",
    _ppl_metrics: dict[str, Any] | None = None,
    target_ratio: float | None = None,
    guard_metric_impact: dict[str, Any] | None = None,
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
    drift_min, drift_max = resolve_drift_bounds(
        pm_drift_band,
        default=pm_drift_band_default,
    )
    preview_final_drift_acceptable = _resolve_preview_drift(
        ppl=ppl,
        primary_metric=primary_metric,
        metrics_policy=metrics_policy,
        drift_min=drift_min,
        drift_max=drift_max,
        tiny_relax=tiny_relax,
    )

    # 2. Primary metric vs baseline: edited/baseline ≤ tier threshold (ratio for ppl-like)
    ratio_vs_baseline = ppl.get("ratio_vs_baseline")
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
    tokens_ok = resolve_tokens_ok(
        _ppl_metrics,
        min_tokens=min_tokens,
        pm_policy=pm_policy,
        dataset_capacity=dataset_capacity,
    )
    # Under tiny_relax, treat token floors as informational only
    tokens_ok_eff = tokens_ok or tiny_relax
    # Apply hysteresis to ratio limit if needed
    ratio_limit_with_hyst = ratio_limit + max(0.0, hysteresis_ratio)
    ratio_value = _coerce_finite_float(ratio_vs_baseline)
    lower_bound_ok = True
    if ratio_min_bound is not None and ratio_value is not None:
        try:
            lower_bound_ok = ratio_value >= float(ratio_min_bound)
        except _NON_FATAL_EXCEPTIONS:
            lower_bound_ok = True
    compression_acceptable = (
        ratio_value is not None
        and lower_bound_ok
        and ratio_value <= ratio_limit_with_hyst
        and tokens_ok_eff
    )
    ratio_ci = ppl.get("ratio_ci")
    ratio_ci_pass = True
    if (
        isinstance(ratio_ci, tuple | list)
        and len(ratio_ci) == 2
        and all(_is_non_bool_finite_number(x) for x in ratio_ci)
    ):
        ratio_ci_pass = bool(
            ratio_ci[1] <= ratio_limit_with_hyst
            and (ratio_min_bound is None or ratio_ci[0] >= ratio_min_bound)
        )
        compression_acceptable = compression_acceptable and ratio_ci_pass

    # 3. RMT ε-rule compliance
    rmt_stable = resolve_rmt_stable(rmt)

    effective_tier_policy = tier_policy if isinstance(tier_policy, Mapping) else {}
    spectral_stable = resolve_spectral_stable(
        spectral,
        tier_policy=effective_tier_policy,
    )
    guard_metric_impact_pass = _resolve_guard_metric_impact_pass(
        guard_metric_impact,
        tiny_relax=tiny_relax,
    )

    flags = {
        "preview_final_drift_acceptable": preview_final_drift_acceptable,
        "primary_metric_acceptable": compression_acceptable,
        "invariants_pass": resolve_invariants_pass(invariants),
        "spectral_stable": spectral_stable,
        "rmt_stable": rmt_stable,  # RMT ε-rule compliance
        "guard_metric_impact_acceptable": guard_metric_impact_pass,
    }
    # Mark hysteresis application when ratio exceeds base limit but passes with hysteresis
    try:
        base_ok = ratio_value is not None and ratio_value <= ratio_limit
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
        ratio_min_bound=ratio_min_bound,
        ratio_ci_pass=ratio_ci_pass,
    )
    tail_required = False
    if isinstance(primary_metric, dict):
        try:
            tail_required = is_ppl_metric_kind(primary_metric.get("kind"))
        except MetricKindContractError:
            tail_required = False
    _apply_optional_observability_flags(
        flags,
        moe=moe,
        pm_tail=pm_tail,
        tail_required=tail_required,
    )

    return flags
