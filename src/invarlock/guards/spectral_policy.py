from __future__ import annotations

import math
from typing import Any

from invarlock.core.exceptions import ValidationError


def default_family_caps() -> dict[str, dict[str, float]]:
    """Default per-family spectral z-score caps."""
    return {
        "ffn": {"kappa": 2.5},
        "attn": {"kappa": 2.8},
        "embed": {"kappa": 3.0},
        "other": {"kappa": 3.0},
    }


def normalize_family_caps(
    caps: Any, *, default: bool = True
) -> dict[str, dict[str, float]]:
    """Normalize family cap configuration into canonical mapping."""
    if not isinstance(caps, dict) or not caps:
        return default_family_caps() if default else {}

    normalized: dict[str, dict[str, float]] = {}
    for family, values in caps.items():
        entry: dict[str, float] = {}
        if isinstance(values, dict):
            for key, val in values.items():
                if isinstance(val, int | float) and math.isfinite(float(val)):
                    entry[str(key)] = float(val)
        elif isinstance(values, int | float) and math.isfinite(float(values)):
            entry["kappa"] = float(values)
        if entry:
            normalized[str(family)] = entry

    if normalized:
        return normalized
    return default_family_caps() if default else {}


def serialize_policy(guard: Any) -> dict[str, Any]:
    """Snapshot current guard policy for report serialization."""
    return {
        "scope": guard.scope,
        "sigma_quantile": float(guard.sigma_quantile),
        "deadband": float(guard.deadband),
        "max_caps": int(guard.max_caps),
        "max_spectral_norm": (
            float(guard.max_spectral_norm)
            if guard.max_spectral_norm is not None
            else None
        ),
        "family_caps": guard.family_caps,
        "multiple_testing": guard.multiple_testing,
        "estimator": guard.estimator,
        "degeneracy": guard.degeneracy,
        "correction_enabled": bool(guard.correction_enabled),
        "ignore_preview_inflation": bool(guard.ignore_preview_inflation),
    }


def apply_policy_overrides(guard: Any, policy: dict[str, Any]) -> None:
    """Hydrate a spectral guard from an override policy block."""
    sigma_value = policy.get("sigma_quantile")
    if "contraction" in policy or "kappa" in policy:
        raise ValueError(
            "Spectral policy keys 'contraction'/'kappa' are not supported; "
            "use 'sigma_quantile'."
        )
    if sigma_value is not None:
        guard.sigma_quantile = float(sigma_value)
        policy["sigma_quantile"] = guard.sigma_quantile
    guard.config["sigma_quantile"] = guard.sigma_quantile

    for key in [
        "sigma_quantile",
        "deadband",
        "scope",
        "max_spectral_norm",
        "correction_enabled",
        "max_caps",
    ]:
        if key in policy:
            setattr(guard, key, policy[key])
            guard.config[key] = policy[key]

    if guard.max_spectral_norm is not None:
        guard.max_spectral_norm = float(guard.max_spectral_norm)
    guard.config["max_spectral_norm"] = guard.max_spectral_norm

    if "family_caps" in policy:
        guard.family_caps = normalize_family_caps(policy["family_caps"], default=True)
        guard.config["family_caps"] = guard.family_caps

    if "ignore_preview_inflation" in policy:
        guard.ignore_preview_inflation = bool(policy["ignore_preview_inflation"])
        guard.config["ignore_preview_inflation"] = guard.ignore_preview_inflation

    if "baseline_family_stats" in policy and isinstance(
        policy["baseline_family_stats"], dict
    ):
        guard.baseline_family_stats = {
            family: stats.copy()
            for family, stats in policy["baseline_family_stats"].items()
            if isinstance(stats, dict)
        }
        guard.config["baseline_family_stats"] = guard.baseline_family_stats

    if "multipletesting" in policy:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={
                "param": "multipletesting",
                "hint": "Use spectral.multiple_testing instead.",
            },
        )

    mt_policy = policy.get("multiple_testing")
    if isinstance(mt_policy, dict):
        guard.multiple_testing = mt_policy.copy()
        policy["multiple_testing"] = guard.multiple_testing
        guard.config["multiple_testing"] = guard.multiple_testing

    estimator_policy = policy.get("estimator")
    if isinstance(estimator_policy, dict):
        try:
            est_iters = int(estimator_policy.get("iters", 4) or 4)
        except Exception:
            est_iters = 4
        if est_iters < 1:
            est_iters = 1
        est_init = str(estimator_policy.get("init", "ones") or "ones").strip().lower()
        if est_init not in {"ones", "e0"}:
            est_init = "ones"
        guard.estimator = {"type": "power_iter", "iters": est_iters, "init": est_init}
        guard.config["estimator"] = guard.estimator

    degeneracy_policy = policy.get("degeneracy")
    if isinstance(degeneracy_policy, dict):
        stable_rank_cfg = degeneracy_policy.get("stable_rank")
        norm_collapse_cfg = degeneracy_policy.get("norm_collapse")
        guard.degeneracy = {
            "enabled": bool(degeneracy_policy.get("enabled", True)),
            "stable_rank": {
                "warn_ratio": float((stable_rank_cfg or {}).get("warn_ratio", 0.5)),
                "fatal_ratio": float((stable_rank_cfg or {}).get("fatal_ratio", 0.25)),
            },
            "norm_collapse": {
                "warn_ratio": float((norm_collapse_cfg or {}).get("warn_ratio", 0.25)),
                "fatal_ratio": float(
                    (norm_collapse_cfg or {}).get("fatal_ratio", 0.10)
                ),
            },
        }
        guard.config["degeneracy"] = guard.degeneracy


__all__ = [
    "apply_policy_overrides",
    "default_family_caps",
    "normalize_family_caps",
    "serialize_policy",
]
