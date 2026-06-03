"""Guard policy presets, resolution helpers, and validation gates."""

from __future__ import annotations

import math
import os
from typing import Any, Literal, NotRequired, TypedDict, cast

from invarlock.core.exceptions import GuardError, PolicyViolationError, ValidationError

from .rmt_policy import RMTPolicyDict
from .spectral_policy import normalize_family_caps, normalize_multiple_testing_config
from .tier_config import (
    GuardType,
    TierName,
    get_tier_guard_config,
)
from .tier_config import (
    check_drift as check_tier_drift,
)


class SpectralPolicy(TypedDict, total=False):
    """Type definition for spectral guard policy configuration."""

    sigma_quantile: float
    deadband: float
    scope: str
    estimator: dict[str, Any]
    degeneracy: dict[str, Any]
    correction_enabled: bool
    family_caps: dict[str, dict[str, float]]
    ignore_preview_inflation: bool
    max_caps: int
    max_spectral_norm: float | None
    multiple_testing: dict[str, Any]


def _is_non_bool_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


SPECTRAL_CONSERVATIVE: SpectralPolicy = {
    "sigma_quantile": 0.90,
    "deadband": 0.05,
    "scope": "ffn",
    "correction_enabled": True,
    "max_caps": 3,
    "max_spectral_norm": None,
    "multiple_testing": {"method": "bonferroni", "alpha": 0.02, "m": 4},
}

SPECTRAL_BALANCED: SpectralPolicy = {
    "sigma_quantile": 0.95,
    "deadband": 0.10,
    "scope": "ffn",
    "correction_enabled": False,
    "max_caps": 5,
    "max_spectral_norm": None,
    "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
}

SPECTRAL_AGGRESSIVE: SpectralPolicy = {
    "sigma_quantile": 0.98,
    "deadband": 0.15,
    "scope": "all",
    "correction_enabled": True,
    "max_caps": 8,
    "max_spectral_norm": None,
    "multiple_testing": {"method": "bh", "alpha": 0.1, "m": 4},
}

SPECTRAL_ATTN_AWARE: SpectralPolicy = {
    "sigma_quantile": 0.95,
    "deadband": 0.10,
    "scope": "attn",
    "correction_enabled": False,
    "max_caps": 5,
    "max_spectral_norm": None,
    "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
}

RMT_CONSERVATIVE: RMTPolicyDict = {
    "q": "auto",
    "deadband": 0.05,
    "margin": 1.3,
    "correct": True,
    "epsilon_default": 0.06,
    "epsilon_by_family": {"attn": 0.05, "ffn": 0.06, "embed": 0.07, "other": 0.07},
}

RMT_BALANCED: RMTPolicyDict = {
    "q": "auto",
    "deadband": 0.10,
    "margin": 1.5,
    "correct": False,
    "epsilon_default": 0.10,
    "epsilon_by_family": {"attn": 0.08, "ffn": 0.10, "embed": 0.12, "other": 0.12},
}

RMT_AGGRESSIVE: RMTPolicyDict = {
    "q": "auto",
    "deadband": 0.15,
    "margin": 1.8,
    "correct": True,
    "epsilon_default": 0.15,
    "epsilon_by_family": {"attn": 0.15, "ffn": 0.15, "embed": 0.15, "other": 0.15},
}


class VariancePolicyRequired(TypedDict):
    """TypedDict for variance guard policy configuration."""

    min_gain: float
    max_calib: int
    scope: Literal["ffn", "attn", "both"]
    clamp: tuple[float, float]
    deadband: float
    seed: int
    mode: Literal["delta", "ci"]
    min_rel_gain: float
    alpha: float


class VariancePolicyDict(VariancePolicyRequired, total=False):
    """Extended variance policy allowing optional calibration overrides."""

    calibration: dict[str, Any]
    tie_breaker_deadband: NotRequired[float]
    min_effect_lognll: NotRequired[float]
    min_abs_adjust: NotRequired[float]
    max_scale_step: NotRequired[float]
    topk_backstop: NotRequired[int]
    predictive_gate: NotRequired[bool]
    monitor_only: NotRequired[bool]
    target_modules: NotRequired[list[str]]
    tap: NotRequired[str | list[str]]


VARIANCE_CONSERVATIVE: VariancePolicyDict = {
    "min_gain": 0.02,
    "max_calib": 160,
    "scope": "ffn",
    "clamp": (0.85, 1.12),
    "deadband": 0.03,
    "seed": 42,
    "mode": "ci",
    "min_rel_gain": 0.002,
    "alpha": 0.05,
    "tie_breaker_deadband": 0.005,
    "min_effect_lognll": 0.0018,
    "min_abs_adjust": 0.02,
    "max_scale_step": 0.015,
    "topk_backstop": 0,
    "predictive_gate": True,
    "tap": "transformer.h.*.mlp.c_proj",
    "calibration": {
        "windows": 16,
        "min_coverage": 12,
        "seed": 42,
    },
}

VARIANCE_BALANCED: VariancePolicyDict = {
    "min_gain": 0.0,
    "max_calib": 160,
    "scope": "ffn",
    "clamp": (0.85, 1.12),
    "deadband": 0.02,
    "seed": 123,
    "mode": "ci",
    "min_rel_gain": 0.001,
    "alpha": 0.05,
    "tie_breaker_deadband": 0.001,
    "min_effect_lognll": 0.0009,
    "min_abs_adjust": 0.012,
    "max_scale_step": 0.03,
    "topk_backstop": 1,
    "predictive_gate": True,
    "tap": "transformer.h.*.mlp.c_proj",
    "calibration": {
        "windows": 12,
        "min_coverage": 10,
        "seed": 123,
    },
}

VARIANCE_AGGRESSIVE: VariancePolicyDict = {
    "min_gain": 0.0,
    "max_calib": 240,
    "scope": "both",
    "clamp": (0.3, 3.0),
    "deadband": 0.12,
    "seed": 456,
    "mode": "ci",
    "min_rel_gain": 0.0025,
    "alpha": 0.05,
    "tie_breaker_deadband": 0.0005,
    "min_effect_lognll": 0.0005,
    "calibration": {
        "windows": 6,
        "min_coverage": 4,
        "seed": 456,
    },
}

DEFAULT_SPECTRAL_POLICIES: dict[str, SpectralPolicy] = {
    "conservative": SPECTRAL_CONSERVATIVE,
    "balanced": SPECTRAL_BALANCED,
    "aggressive": SPECTRAL_AGGRESSIVE,
    "attn_aware": SPECTRAL_ATTN_AWARE,
}

DEFAULT_RMT_POLICIES: dict[str, RMTPolicyDict] = {
    "conservative": RMT_CONSERVATIVE,
    "balanced": RMT_BALANCED,
    "aggressive": RMT_AGGRESSIVE,
}

DEFAULT_VARIANCE_POLICIES: dict[str, VariancePolicyDict] = {
    "conservative": VARIANCE_CONSERVATIVE,
    "balanced": VARIANCE_BALANCED,
    "aggressive": VARIANCE_AGGRESSIVE,
}

VALIDATION_GATE_STRICT: dict[str, Any] = {
    "max_capping_rate": 0.3,
    "max_ppl_degradation": 0.01,
    "require_branch_balance": True,
}

VALIDATION_GATE_STANDARD: dict[str, Any] = {
    "max_capping_rate": 0.5,
    "max_ppl_degradation": 0.02,
    "require_branch_balance": True,
}

VALIDATION_GATE_PERMISSIVE: dict[str, Any] = {
    "max_capping_rate": 0.7,
    "max_ppl_degradation": 0.05,
    "require_branch_balance": False,
}

DEFAULT_VALIDATION_GATES: dict[str, dict[str, Any]] = {
    "strict": VALIDATION_GATE_STRICT,
    "standard": VALIDATION_GATE_STANDARD,
    "permissive": VALIDATION_GATE_PERMISSIVE,
}


def guard_assert(cond: bool, msg: str) -> None:
    """Enable lightweight runtime contracts when INVARLOCK_ASSERT_GUARDS=1."""
    if os.getenv("INVARLOCK_ASSERT_GUARDS", "0") == "1" and not bool(cond):
        raise AssertionError(msg)


def get_spectral_policy(
    name: str = "balanced", *, use_yaml: bool = True
) -> SpectralPolicy:
    """
    Get a spectral policy by name.

    Loads values from tiers.yaml (calibrated source of truth) when available,
    falling back to hardcoded defaults for robustness.
    """
    if name not in DEFAULT_SPECTRAL_POLICIES:
        available = list(DEFAULT_SPECTRAL_POLICIES.keys())
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": available},
        )

    policy = DEFAULT_SPECTRAL_POLICIES[name].copy()

    if use_yaml and name in ("balanced", "conservative", "aggressive"):
        try:
            tier_name = cast(TierName, name)
            guard_key: GuardType = "spectral_guard"
            tier_config = get_tier_guard_config(tier_name, guard_key)
            if tier_config:
                if "sigma_quantile" in tier_config:
                    sigma_quantile = tier_config["sigma_quantile"]
                    if _is_non_bool_number(sigma_quantile):
                        policy["sigma_quantile"] = float(sigma_quantile)
                if "deadband" in tier_config:
                    deadband = tier_config["deadband"]
                    if _is_non_bool_number(deadband):
                        policy["deadband"] = float(deadband)
                if "scope" in tier_config:
                    policy["scope"] = tier_config["scope"]
                if "max_caps" in tier_config:
                    max_caps = tier_config["max_caps"]
                    if _is_non_bool_number(max_caps):
                        policy["max_caps"] = int(max_caps)
                if "max_spectral_norm" in tier_config:
                    max_spectral_norm = tier_config["max_spectral_norm"]
                    if _is_non_bool_number(max_spectral_norm):
                        policy["max_spectral_norm"] = float(max_spectral_norm)
                if "family_caps" in tier_config:
                    policy["family_caps"] = normalize_family_caps(
                        tier_config["family_caps"], default=True
                    )
                if "multiple_testing" in tier_config:
                    policy["multiple_testing"] = normalize_multiple_testing_config(
                        tier_config["multiple_testing"]
                    )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    return policy


def create_custom_spectral_policy(
    sigma_quantile: float = 0.95,
    deadband: float = 0.10,
    scope: str = "ffn",
) -> SpectralPolicy:
    """Create a custom spectral policy."""
    if (
        not _is_non_bool_number(sigma_quantile)
        or not 0.0 <= float(sigma_quantile) <= 1.0
    ):
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "sigma_quantile", "value": sigma_quantile},
        )

    if not _is_non_bool_number(deadband) or not 0.0 <= float(deadband) <= 0.5:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "deadband", "value": deadband},
        )

    if scope not in ["ffn", "attn", "all"]:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "scope", "value": scope},
        )

    return SpectralPolicy(
        sigma_quantile=sigma_quantile,
        deadband=deadband,
        scope=scope,
    )


def get_policy_for_model_size(param_count: int) -> SpectralPolicy:
    """Get recommended spectral policy based on model size."""
    if param_count < 100_000_000:
        return get_spectral_policy("aggressive")
    if param_count < 1_000_000_000:
        return get_spectral_policy("balanced")
    return get_spectral_policy("conservative")


def get_rmt_policy(name: str = "balanced", *, use_yaml: bool = True) -> RMTPolicyDict:
    """
    Get an RMT policy by name.

    Loads values from tiers.yaml (calibrated source of truth) when available,
    falling back to hardcoded defaults for robustness.
    """
    if name not in DEFAULT_RMT_POLICIES:
        available = list(DEFAULT_RMT_POLICIES.keys())
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": available},
        )

    policy = DEFAULT_RMT_POLICIES[name].copy()

    if use_yaml and name in ("balanced", "conservative", "aggressive"):
        try:
            tier_name = cast(TierName, name)
            guard_key: GuardType = "rmt_guard"
            tier_config = get_tier_guard_config(tier_name, guard_key)
            if tier_config:
                if "deadband" in tier_config:
                    deadband = tier_config["deadband"]
                    if _is_non_bool_number(deadband):
                        policy["deadband"] = float(deadband)
                if "margin" in tier_config:
                    margin = tier_config["margin"]
                    if _is_non_bool_number(margin):
                        policy["margin"] = float(margin)
                if "epsilon_default" in tier_config:
                    epsilon_default = tier_config["epsilon_default"]
                    if _is_non_bool_number(epsilon_default):
                        policy["epsilon_default"] = float(epsilon_default)
                if "epsilon_by_family" in tier_config:
                    epsilon_by_family = tier_config["epsilon_by_family"]
                    if isinstance(epsilon_by_family, dict):
                        normalized_epsilon_by_family = {
                            str(family): float(value)
                            for family, value in epsilon_by_family.items()
                            if _is_non_bool_number(value)
                        }
                        if normalized_epsilon_by_family:
                            policy["epsilon_by_family"] = normalized_epsilon_by_family
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    return policy


def create_custom_rmt_policy(
    q: float | Literal["auto"] = "auto",
    deadband: float = 0.10,
    margin: float = 1.5,
    correct: bool = True,
) -> RMTPolicyDict:
    """Create a custom RMT policy."""
    if q != "auto" and (not _is_non_bool_number(q) or not 0.1 <= float(q) <= 10.0):
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "q", "value": q},
        )

    if not _is_non_bool_number(deadband) or not 0.0 <= float(deadband) <= 0.5:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "deadband", "value": deadband},
        )

    if not _is_non_bool_number(margin) or not float(margin) >= 1.0:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "margin", "value": margin},
        )

    return RMTPolicyDict(q=q, deadband=deadband, margin=margin, correct=correct)


def get_rmt_policy_for_model_size(param_count: int) -> RMTPolicyDict:
    """Get recommended RMT policy based on model size."""
    if param_count < 100_000_000:
        return get_rmt_policy("aggressive")
    if param_count < 1_000_000_000:
        return get_rmt_policy("balanced")
    return get_rmt_policy("conservative")


def get_variance_policy(
    name: str = "balanced", *, use_yaml: bool = True
) -> VariancePolicyDict:
    """
    Get a variance policy by name.

    Loads values from tiers.yaml (calibrated source of truth) when available,
    falling back to hardcoded defaults for robustness.
    """
    if name not in DEFAULT_VARIANCE_POLICIES:
        available = list(DEFAULT_VARIANCE_POLICIES.keys())
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": available},
        )

    policy = DEFAULT_VARIANCE_POLICIES[name].copy()

    if use_yaml and name in ("balanced", "conservative", "aggressive"):
        try:
            tier_name = cast(TierName, name)
            guard_key: GuardType = "variance_guard"
            tier_config = get_tier_guard_config(tier_name, guard_key)
            if tier_config:
                if "deadband" in tier_config:
                    deadband = tier_config["deadband"]
                    if _is_non_bool_number(deadband):
                        policy["deadband"] = float(deadband)
                if "min_effect_lognll" in tier_config:
                    min_effect_lognll = tier_config["min_effect_lognll"]
                    if _is_non_bool_number(min_effect_lognll):
                        policy["min_effect_lognll"] = float(min_effect_lognll)
                if "min_abs_adjust" in tier_config:
                    min_abs_adjust = tier_config["min_abs_adjust"]
                    if _is_non_bool_number(min_abs_adjust):
                        policy["min_abs_adjust"] = float(min_abs_adjust)
                if "max_scale_step" in tier_config:
                    max_scale_step = tier_config["max_scale_step"]
                    if _is_non_bool_number(max_scale_step):
                        policy["max_scale_step"] = float(max_scale_step)
                if "topk_backstop" in tier_config:
                    topk_backstop = tier_config["topk_backstop"]
                    if _is_non_bool_number(topk_backstop):
                        policy["topk_backstop"] = int(topk_backstop)
                if "predictive_one_sided" in tier_config:
                    pass
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    return policy


def create_custom_variance_policy(
    min_gain: float = 0.30,
    max_calib: int = 200,
    scope: Literal["ffn", "attn", "both"] = "both",
    clamp: tuple[float, float] = (0.5, 2.0),
    deadband: float = 0.10,
    seed: int = 123,
    mode: Literal["delta", "ci"] = "ci",
    min_rel_gain: float = 0.005,
    alpha: float = 0.05,
) -> VariancePolicyDict:
    """Create a custom variance policy."""
    if not 0.0 <= min_gain <= 1.0:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "min_gain", "value": min_gain},
        )

    if not 50 <= max_calib <= 1000:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "max_calib", "value": max_calib},
        )

    if scope not in ["ffn", "attn", "both"]:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "scope", "value": scope},
        )

    clamp_min, clamp_max = clamp
    if not (0.1 <= clamp_min < clamp_max <= 5.0):
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "clamp", "value": clamp},
        )

    if not 0.0 <= deadband <= 0.5:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "deadband", "value": deadband},
        )

    if mode not in {"delta", "ci"}:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "mode", "value": mode},
        )

    if not 0.0 <= min_rel_gain < 1.0:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "min_rel_gain", "value": min_rel_gain},
        )

    if not 0.0 < alpha < 1.0:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "alpha", "value": alpha},
        )

    return VariancePolicyDict(
        min_gain=min_gain,
        max_calib=max_calib,
        scope=scope,
        clamp=clamp,
        deadband=deadband,
        seed=seed,
        mode=mode,
        min_rel_gain=min_rel_gain,
        alpha=alpha,
    )


def get_variance_policy_for_model_size(param_count: int) -> VariancePolicyDict:
    """Get recommended variance policy based on model size."""
    if param_count < 100_000_000:
        return get_variance_policy("aggressive")
    if param_count < 1_000_000_000:
        return get_variance_policy("balanced")
    return get_variance_policy("conservative")


def get_validation_gate(name: str = "standard") -> dict[str, Any]:
    """Get validation gate configuration by name."""
    if name not in DEFAULT_VALIDATION_GATES:
        available = list(DEFAULT_VALIDATION_GATES.keys())
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": available},
        )

    return DEFAULT_VALIDATION_GATES[name].copy()


def enforce_validation_gate(metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    """Enforce validation gate thresholds."""
    violations: list[dict[str, Any]] = []

    try:
        caps_value = metrics.get("caps_applied", 0)
        total_value = metrics.get("total_layers", 0)
        if not (_is_non_bool_number(caps_value) and _is_non_bool_number(total_value)):
            raise TypeError("caps_applied and total_layers must be numeric")
        caps = float(caps_value)
        total = float(total_value)
        if total > 0:
            rate = caps / total
            limit = float(gate.get("max_capping_rate", 1.0))
            if rate > limit:
                violations.append(
                    {
                        "type": "capping_rate",
                        "actual": rate,
                        "limit": limit,
                    }
                )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    try:
        ratio = metrics.get("primary_metric_ratio")
        if (
            isinstance(ratio, int | float)
            and not isinstance(ratio, bool)
            and math.isfinite(float(ratio))
        ):
            ratio_f = float(ratio)
            limit = float(gate.get("max_ppl_degradation", 1.0))
            degradation = ratio_f - 1.0
            if degradation > limit:
                violations.append(
                    {
                        "type": "primary_metric_degradation",
                        "actual": degradation,
                        "limit": limit,
                    }
                )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    if isinstance(gate.get("require_branch_balance"), bool) and gate.get(
        "require_branch_balance"
    ):
        if metrics.get("branch_balance_ok") is False:
            violations.append(
                {"type": "branch_balance", "actual": False, "limit": True}
            )

    if violations:
        raise PolicyViolationError(
            code="E503",
            message="VALIDATION-GATE-FAILED",
            details={"violations": violations, "metrics": metrics, "gate": gate},
        )


def check_policy_drift(*, silent: bool = False) -> dict[str, list[str]]:
    """Check for drift between tiers.yaml and hardcoded policy fallbacks."""
    return check_tier_drift(silent=silent)


__all__ = [
    "DEFAULT_RMT_POLICIES",
    "DEFAULT_SPECTRAL_POLICIES",
    "DEFAULT_VALIDATION_GATES",
    "DEFAULT_VARIANCE_POLICIES",
    "NotRequired",
    "RMT_AGGRESSIVE",
    "RMT_BALANCED",
    "RMT_CONSERVATIVE",
    "RMTPolicyDict",
    "SPECTRAL_AGGRESSIVE",
    "SPECTRAL_ATTN_AWARE",
    "SPECTRAL_BALANCED",
    "SPECTRAL_CONSERVATIVE",
    "SpectralPolicy",
    "TypedDict",
    "VALIDATION_GATE_PERMISSIVE",
    "VALIDATION_GATE_STANDARD",
    "VALIDATION_GATE_STRICT",
    "VARIANCE_AGGRESSIVE",
    "VARIANCE_BALANCED",
    "VARIANCE_CONSERVATIVE",
    "VariancePolicyDict",
    "VariancePolicyRequired",
    "_is_non_bool_number",
    "check_policy_drift",
    "create_custom_rmt_policy",
    "create_custom_spectral_policy",
    "create_custom_variance_policy",
    "enforce_validation_gate",
    "get_policy_for_model_size",
    "get_rmt_policy",
    "get_rmt_policy_for_model_size",
    "get_spectral_policy",
    "get_tier_guard_config",
    "get_validation_gate",
    "get_variance_policy",
    "get_variance_policy_for_model_size",
    "guard_assert",
]
