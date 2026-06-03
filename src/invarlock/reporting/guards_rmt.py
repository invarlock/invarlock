from __future__ import annotations

import math
from typing import Any

from invarlock.core.auto_tuning import get_tier_policies

from . import policy_utils as report_policy_utils_mod
from .report_types import RunReport
from .verify_check_helpers_consistency import (
    _baseline_guard_payload,
    _measurement_contract_digest,
)

_GUARD_PARSE_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
)


def _to_float_or_none(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    try:
        return float(value)
    except _GUARD_PARSE_EXCEPTIONS:
        return None


def _numeric_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(raw)
        for key, raw in value.items()
        if isinstance(raw, int | float) and math.isfinite(float(raw))
    }


def _tier_rmt_defaults(report: RunReport) -> tuple[str, dict[str, float], float]:
    tier = report_policy_utils_mod._resolve_policy_tier(report)
    tier_policies = get_tier_policies()
    tier_defaults = tier_policies.get(tier, tier_policies.get("balanced", {}))

    default_epsilon_map = (
        tier_defaults.get("rmt", {}).get("epsilon_by_family")
        if isinstance(tier_defaults, dict)
        else {}
    )

    epsilon_default = 0.1
    try:
        eps_def = (
            tier_defaults.get("rmt", {}).get("epsilon_default")
            if isinstance(tier_defaults, dict)
            else None
        )
        if isinstance(eps_def, int | float) and math.isfinite(float(eps_def)):
            epsilon_default = float(eps_def)
    except _GUARD_PARSE_EXCEPTIONS:
        pass

    return tier, _numeric_map(default_epsilon_map), epsilon_default


def _baseline_rmt_context(
    baseline: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any] | None]:
    baseline_rmt = _baseline_guard_payload(baseline, "rmt")
    baseline_edge_by_family: dict[str, float] = {}
    baseline_contract = None
    if isinstance(baseline_rmt, dict) and baseline_rmt:
        bc = baseline_rmt.get("measurement_contract")
        if isinstance(bc, dict) and bc:
            baseline_contract = bc
        base = baseline_rmt.get("edge_risk_by_family") or baseline_rmt.get(
            "edge_risk_by_family_base"
        )
        baseline_edge_by_family = _numeric_map(base)
    return baseline_edge_by_family, baseline_contract


def _find_rmt_guard(
    report: RunReport,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    for guard in report.get("guards", []) or []:
        if str(guard.get("name", "")).lower() == "rmt":
            metrics = guard.get("metrics", {}) or {}
            policy = guard.get("policy", {}) or {}
            return (
                dict(guard),
                metrics if isinstance(metrics, dict) else {},
                policy if isinstance(policy, dict) else {},
            )
    return None, {}, {}


def _resolve_epsilon_policy(
    guard_metrics: dict[str, Any],
    guard_policy: dict[str, Any],
    epsilon_default: float,
) -> tuple[dict[str, Any] | None, float, dict[str, float]]:
    policy_out: dict[str, Any] | None = None
    if guard_policy:
        policy_out = dict(guard_policy)
        epsilon_default_value = _to_float_or_none(policy_out.get("epsilon_default"))
        if epsilon_default_value is not None and math.isfinite(epsilon_default_value):
            epsilon_default = epsilon_default_value

    metric_epsilon_default = _to_float_or_none(guard_metrics.get("epsilon_default"))
    if metric_epsilon_default is not None and math.isfinite(metric_epsilon_default):
        epsilon_default = metric_epsilon_default

    eps_src = guard_metrics.get("epsilon_by_family") or {}
    if not eps_src:
        eps_src = guard_policy.get("epsilon_by_family") or {}
    return policy_out, epsilon_default, _numeric_map(eps_src)


def _edge_risk_maps(
    guard_metrics: dict[str, Any],
    baseline_edge_by_family: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    edge_base = _numeric_map(guard_metrics.get("edge_risk_by_family_base") or {})
    edge_cur = _numeric_map(guard_metrics.get("edge_risk_by_family") or {})
    if not edge_base and baseline_edge_by_family:
        edge_base = dict(baseline_edge_by_family)
    return edge_base, edge_cur


def _epsilon_violations(
    guard_metrics: dict[str, Any],
    *,
    edge_base: dict[str, float],
    edge_cur: dict[str, float],
    epsilon_map: dict[str, float],
    default_epsilon_map: dict[str, float],
    epsilon_default: float,
) -> list[Any]:
    explicit = guard_metrics.get("epsilon_violations") or []
    if isinstance(explicit, list) and explicit:
        return list(explicit)

    violations: list[Any] = []
    for family in set(edge_cur) | set(edge_base):
        base = float(edge_base.get(family, 0.0) or 0.0)
        cur = float(edge_cur.get(family, 0.0) or 0.0)
        if base <= 0.0:
            continue
        eps_value = _to_float_or_none(
            epsilon_map.get(family, default_epsilon_map.get(family, epsilon_default))
        )
        eps = epsilon_default if eps_value is None else eps_value
        threshold = (1.0 + eps) * base
        if cur > threshold:
            violations.append(
                {
                    "family": family,
                    "edge_base": base,
                    "edge_cur": cur,
                    "delta": float((cur / base) - 1.0),
                    "allowed": threshold,
                    "epsilon": eps,
                }
            )
    return violations


def _family_breakdown(
    *,
    edge_base: dict[str, float],
    edge_cur: dict[str, float],
    epsilon_map: dict[str, float],
    default_epsilon_map: dict[str, float],
    epsilon_default: float,
) -> tuple[dict[str, dict[str, Any]], list[float], list[float]]:
    families_all = sorted(
        set(edge_base) | set(edge_cur) | set(epsilon_map) | set(default_epsilon_map)
    )
    family_breakdown: dict[str, dict[str, Any]] = {}
    ratios: list[float] = []
    deltas: list[float] = []
    for family in families_all:
        base = float(edge_base.get(family, 0.0) or 0.0)
        cur = float(edge_cur.get(family, 0.0) or 0.0)
        eps_value = _to_float_or_none(
            epsilon_map.get(family, default_epsilon_map.get(family, epsilon_default))
        )
        eps = epsilon_default if eps_value is None else eps_value
        allowed: float | None = (1.0 + eps) * base if base > 0.0 else None
        ratio: float | None = (cur / base) if base > 0.0 else None
        delta: float | None = ((cur / base) - 1.0) if base > 0.0 else None
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
    return family_breakdown, ratios, deltas


def _measurement_contract(guard_metrics: dict[str, Any]) -> dict[str, Any] | None:
    try:
        mc = guard_metrics.get("measurement_contract")
        if isinstance(mc, dict) and mc:
            return mc
    except _GUARD_PARSE_EXCEPTIONS:
        return None
    return None


def _extract_rmt_analysis(
    report: RunReport, baseline: dict[str, Any]
) -> dict[str, Any]:
    """Extract RMT analysis using activation edge-risk epsilon-band semantics."""
    tier, default_epsilon_map, epsilon_default = _tier_rmt_defaults(report)
    baseline_edge_by_family, baseline_contract = _baseline_rmt_context(baseline)
    rmt_guard, guard_metrics, guard_policy = _find_rmt_guard(report)
    policy_out, epsilon_default, epsilon_map = _resolve_epsilon_policy(
        guard_metrics,
        guard_policy,
        epsilon_default,
    )
    edge_base, edge_cur = _edge_risk_maps(guard_metrics, baseline_edge_by_family)
    epsilon_violations = _epsilon_violations(
        guard_metrics,
        edge_base=edge_base,
        edge_cur=edge_cur,
        epsilon_map=epsilon_map,
        default_epsilon_map=default_epsilon_map,
        epsilon_default=epsilon_default,
    )
    stable = bool(guard_metrics.get("stable", not epsilon_violations))
    family_breakdown, ratios, deltas = _family_breakdown(
        edge_base=edge_base,
        edge_cur=edge_cur,
        epsilon_map=epsilon_map,
        default_epsilon_map=default_epsilon_map,
        epsilon_default=epsilon_default,
    )
    measurement_contract = _measurement_contract(guard_metrics)
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
    mode = None
    if isinstance(measurement_contract, dict):
        raw_mode = measurement_contract.get("kind")
        if isinstance(raw_mode, str) and raw_mode.strip():
            mode = raw_mode.strip()
    if mode is None and result["evaluated"]:
        mode = "activation_edge_risk"
    if mode is not None:
        result["mode"] = mode
    if mc_hash:
        result["measurement_contract_hash"] = mc_hash
    if baseline_hash:
        result["baseline_measurement_contract_hash"] = baseline_hash
    if mc_hash and baseline_hash:
        result["measurement_contract_match"] = bool(mc_hash == baseline_hash)
    return result
