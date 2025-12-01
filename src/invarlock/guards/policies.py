"""
InvarLock Guards - Default Policy Presets
====================================

Default policy configurations for various guard types and use cases.
Provides sensible defaults for different model architectures and safety requirements.

Policy values are loaded from tiers.yaml (the calibrated source of truth) with
hardcoded fallbacks for robustness. Use check_policy_drift() to verify that
code and config are synchronized.
"""

import math
from typing import Any, Literal

try:  # Python 3.12+
    from typing import NotRequired, TypedDict
except ImportError:  # Legacy fallback
    from typing import NotRequired

    from typing_extensions import TypedDict

from invarlock.core.exceptions import (
    GuardError,
    PolicyViolationError,
    ValidationError,
)

from .rmt import RMTPolicyDict
from .spectral import SpectralPolicy
from .tier_config import check_drift as check_tier_drift
from .tier_config import get_tier_guard_config

# === Spectral Guard Policies ===

# Conservative policy - tight control for production use
SPECTRAL_CONSERVATIVE: SpectralPolicy = {
    "sigma_quantile": 0.90,  # Allow only 90% of baseline spectral norm
    "deadband": 0.05,  # 5% deadband - strict threshold
    "scope": "ffn",  # FFN layers only (safest)
    "correction_enabled": True,
    "max_caps": 3,
    "multiple_testing": {"method": "bonferroni", "alpha": 0.02, "m": 4},
}

# Balanced policy - good for most use cases
SPECTRAL_BALANCED: SpectralPolicy = {
    "sigma_quantile": 0.95,  # Allow 95% of baseline spectral norm
    "deadband": 0.10,  # 10% deadband - reasonable tolerance
    "scope": "ffn",  # FFN layers only
    "correction_enabled": False,
    "max_caps": 5,
    "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
}

# Aggressive policy - for research/experimental use
SPECTRAL_AGGRESSIVE: SpectralPolicy = {
    "sigma_quantile": 0.98,  # Allow 98% of baseline spectral norm
    "deadband": 0.15,  # 15% deadband - more permissive
    "scope": "all",  # All layers including attention
    "correction_enabled": True,
    "max_caps": 8,
    "multiple_testing": {"method": "bh", "alpha": 0.1, "m": 4},
}

# Attention-aware policy - includes attention projections
SPECTRAL_ATTN_AWARE: SpectralPolicy = {
    "sigma_quantile": 0.95,  # Standard scaling factor
    "deadband": 0.10,  # Standard deadband
    "scope": "attn",  # Attention layers only
    "correction_enabled": False,
    "max_caps": 5,
    "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
}

# === RMT Guard Policies ===

# Conservative RMT policy - tight control for production use
RMT_CONSERVATIVE: RMTPolicyDict = {
    "q": "auto",  # Auto-derive MP aspect ratio from weight shapes
    "deadband": 0.05,  # 5% deadband - strict threshold
    "margin": 1.3,  # Lower margin for conservative detection
    "correct": True,  # Enable automatic correction
    "epsilon": {"attn": 0.05, "ffn": 0.06, "embed": 0.07, "other": 0.07},
}

# Balanced RMT policy - good for most use cases
RMT_BALANCED: RMTPolicyDict = {
    "q": "auto",  # Auto-derive MP aspect ratio from weight shapes
    "deadband": 0.10,  # 10% deadband - reasonable tolerance
    "margin": 1.5,  # Standard margin for outlier detection
    "correct": False,  # Monitor-only by default
    "epsilon": {"attn": 0.08, "ffn": 0.10, "embed": 0.12, "other": 0.12},
}

# Aggressive RMT policy - for research/experimental use
RMT_AGGRESSIVE: RMTPolicyDict = {
    "q": "auto",  # Auto-derive MP aspect ratio from weight shapes
    "deadband": 0.15,  # 15% deadband - more permissive
    "margin": 1.8,  # Higher margin allows more deviation
    "correct": True,  # Enable automatic correction
    "epsilon": {"attn": 0.15, "ffn": 0.15, "embed": 0.15, "other": 0.15},
}

# === Variance Guard Policies ===


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


# Conservative variance policy - strict A/B gate for production use
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

# Balanced variance policy - good for most use cases
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

# Aggressive variance policy - for research/experimental use
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

# === Policy Collections ===

DEFAULT_SPECTRAL_POLICIES: dict[str, SpectralPolicy] = {
    "conservative": SPECTRAL_CONSERVATIVE,
    "balanced": SPECTRAL_BALANCED,
    "aggressive": SPECTRAL_AGGRESSIVE,
    "attn_aware": SPECTRAL_ATTN_AWARE,
}

# === RMT Policy Collections ===

DEFAULT_RMT_POLICIES: dict[str, RMTPolicyDict] = {
    "conservative": RMT_CONSERVATIVE,
    "balanced": RMT_BALANCED,
    "aggressive": RMT_AGGRESSIVE,
}

# === Variance Policy Collections ===

DEFAULT_VARIANCE_POLICIES: dict[str, VariancePolicyDict] = {
    "conservative": VARIANCE_CONSERVATIVE,
    "balanced": VARIANCE_BALANCED,
    "aggressive": VARIANCE_AGGRESSIVE,
}

# === Utility Functions ===


def get_spectral_policy(
    name: str = "balanced", *, use_yaml: bool = True
) -> SpectralPolicy:
    """
    Get a spectral policy by name.

    Loads values from tiers.yaml (calibrated source of truth) when available,
    falling back to hardcoded defaults for robustness.

    Args:
        name: Policy name ("conservative", "balanced", "aggressive", "attn_aware")
        use_yaml: If True, attempt to load calibrated values from tiers.yaml

    Returns:
        SpectralPolicy configuration with calibrated thresholds

    Raises:
        GuardError(E502): If policy name not found
    """
    if name not in DEFAULT_SPECTRAL_POLICIES:
        available = list(DEFAULT_SPECTRAL_POLICIES.keys())
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": available},
        )

    # Start with hardcoded defaults
    policy = DEFAULT_SPECTRAL_POLICIES[name].copy()

    # Overlay calibrated values from tiers.yaml if available
    if use_yaml and name in ("balanced", "conservative", "aggressive"):
        try:
            tier_config = get_tier_guard_config(name, "spectral_guard")  # type: ignore[arg-type]
            if tier_config:
                # Update with calibrated values
                if "sigma_quantile" in tier_config:
                    policy["sigma_quantile"] = tier_config["sigma_quantile"]
                if "deadband" in tier_config:
                    policy["deadband"] = tier_config["deadband"]
                if "scope" in tier_config:
                    policy["scope"] = tier_config["scope"]
                if "max_caps" in tier_config:
                    policy["max_caps"] = tier_config["max_caps"]
                if "family_caps" in tier_config:
                    policy["family_caps"] = tier_config["family_caps"]
                if "multiple_testing" in tier_config:
                    policy["multiple_testing"] = tier_config["multiple_testing"]
        except Exception:
            # Fallback to hardcoded values on any error
            pass

    return policy


def create_custom_spectral_policy(
    sigma_quantile: float = 0.95,
    deadband: float = 0.10,
    scope: str = "ffn",
) -> SpectralPolicy:
    """
    Create a custom spectral policy.

    Args:
        sigma_quantile: Baseline spectral percentile (0.0-1.0)
        deadband: Tolerance margin (0.0-0.5)
        scope: Module scope ("ffn", "attn", "all")

    Returns:
        Custom SpectralPolicy configuration

    Raises:
        ValidationError(E501): If parameters are out of valid ranges
    """
    if not 0.0 <= sigma_quantile <= 1.0:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "sigma_quantile", "value": sigma_quantile},
        )

    if not 0.0 <= deadband <= 0.5:
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
    """
    Get recommended spectral policy based on model size.

    Args:
        param_count: Number of model parameters

    Returns:
        Recommended SpectralPolicy
    """
    if param_count < 100_000_000:  # < 100M params
        return get_spectral_policy("aggressive")
    elif param_count < 1_000_000_000:  # < 1B params
        return get_spectral_policy("balanced")
    else:  # >= 1B params
        return get_spectral_policy("conservative")


def get_rmt_policy(name: str = "balanced", *, use_yaml: bool = True) -> RMTPolicyDict:
    """
    Get a RMT policy by name.

    Loads values from tiers.yaml (calibrated source of truth) when available,
    falling back to hardcoded defaults for robustness.

    Args:
        name: Policy name ("conservative", "balanced", "aggressive")
        use_yaml: If True, attempt to load calibrated values from tiers.yaml

    Returns:
        RMTPolicyDict configuration with calibrated epsilon values

    Raises:
        GuardError(E502): If policy name not found
    """
    if name not in DEFAULT_RMT_POLICIES:
        available = list(DEFAULT_RMT_POLICIES.keys())
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": available},
        )

    # Start with hardcoded defaults
    policy = DEFAULT_RMT_POLICIES[name].copy()

    # Overlay calibrated values from tiers.yaml if available
    if use_yaml and name in ("balanced", "conservative", "aggressive"):
        try:
            tier_config = get_tier_guard_config(name, "rmt_guard")  # type: ignore[arg-type]
            if tier_config:
                # Update with calibrated values
                if "deadband" in tier_config:
                    policy["deadband"] = tier_config["deadband"]
                if "margin" in tier_config:
                    policy["margin"] = tier_config["margin"]
                # Use epsilon_by_family as the epsilon dict
                if "epsilon_by_family" in tier_config:
                    policy["epsilon"] = tier_config["epsilon_by_family"]
        except Exception:
            # Fallback to hardcoded values on any error
            pass

    return policy


def create_custom_rmt_policy(
    q: float | Literal["auto"] = "auto",
    deadband: float = 0.10,
    margin: float = 1.5,
    correct: bool = True,
) -> RMTPolicyDict:
    """
    Create a custom RMT policy.

    Args:
        q: MP aspect ratio (auto-derived or manual, 0.1-10.0)
        deadband: Tolerance margin (0.0-0.5)
        margin: RMT threshold ratio (>= 1.0)
        correct: Enable automatic correction

    Returns:
        Custom RMTPolicyDict configuration

    Raises:
        ValidationError(E501): If parameters are out of valid ranges
    """
    if isinstance(q, float) and not 0.1 <= q <= 10.0:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "q", "value": q},
        )

    if not 0.0 <= deadband <= 0.5:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "deadband", "value": deadband},
        )

    if not margin >= 1.0:
        raise ValidationError(
            code="E501",
            message="POLICY-PARAM-INVALID",
            details={"param": "margin", "value": margin},
        )

    return RMTPolicyDict(q=q, deadband=deadband, margin=margin, correct=correct)


def get_rmt_policy_for_model_size(param_count: int) -> RMTPolicyDict:
    """
    Get recommended RMT policy based on model size.

    Args:
        param_count: Number of model parameters

    Returns:
        Recommended RMTPolicyDict
    """
    if param_count < 100_000_000:  # < 100M params
        return get_rmt_policy("aggressive")
    elif param_count < 1_000_000_000:  # < 1B params
        return get_rmt_policy("balanced")
    else:  # >= 1B params
        return get_rmt_policy("conservative")


def get_variance_policy(
    name: str = "balanced", *, use_yaml: bool = True
) -> VariancePolicyDict:
    """
    Get a variance policy by name.

    Loads values from tiers.yaml (calibrated source of truth) when available,
    falling back to hardcoded defaults for robustness.

    Args:
        name: Policy name ("conservative", "balanced", "aggressive")
        use_yaml: If True, attempt to load calibrated values from tiers.yaml

    Returns:
        VariancePolicyDict configuration with calibrated thresholds

    Raises:
        GuardError(E502): If policy name not found
    """
    if name not in DEFAULT_VARIANCE_POLICIES:
        available = list(DEFAULT_VARIANCE_POLICIES.keys())
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": available},
        )

    # Start with hardcoded defaults
    policy = DEFAULT_VARIANCE_POLICIES[name].copy()

    # Overlay calibrated values from tiers.yaml if available
    if use_yaml and name in ("balanced", "conservative", "aggressive"):
        try:
            tier_config = get_tier_guard_config(name, "variance_guard")  # type: ignore[arg-type]
            if tier_config:
                # Update with calibrated values
                if "deadband" in tier_config:
                    policy["deadband"] = tier_config["deadband"]
                if "min_effect_lognll" in tier_config:
                    policy["min_effect_lognll"] = tier_config["min_effect_lognll"]
                if "min_abs_adjust" in tier_config:
                    policy["min_abs_adjust"] = tier_config["min_abs_adjust"]
                if "max_scale_step" in tier_config:
                    policy["max_scale_step"] = tier_config["max_scale_step"]
                if "topk_backstop" in tier_config:
                    policy["topk_backstop"] = tier_config["topk_backstop"]
                if "predictive_one_sided" in tier_config:
                    # Map predictive_one_sided to predictive_gate behavior
                    pass  # This is handled elsewhere in variance guard
        except Exception:
            # Fallback to hardcoded values on any error
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
    """
    Create a custom variance policy.

    Args:
        min_gain: Minimum primary-metric improvement to enable VE (0.0-1.0)
        max_calib: Maximum calibration samples (50-1000)
        scope: Module scope ("ffn", "attn", "both")
        clamp: Scaling factor limits (min, max) where 0.1 <= min < max <= 5.0
        deadband: Tolerance margin (0.0-0.5)
        seed: Random seed for deterministic evaluation
        mode: Gate mode (\"ci\" or \"delta\")
        min_rel_gain: Minimum relative gain required under CI mode
        alpha: Confidence interval significance level

    Returns:
        Custom VariancePolicyDict configuration

    Raises:
        ValidationError(E501): If parameters are out of valid ranges
    """
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
    """
    Get recommended variance policy based on model size.

    Args:
        param_count: Number of model parameters

    Returns:
        Recommended VariancePolicyDict
    """
    if param_count < 100_000_000:  # < 100M params
        return get_variance_policy("aggressive")
    elif param_count < 1_000_000_000:  # < 1B params
        return get_variance_policy("balanced")
    else:  # >= 1B params
        return get_variance_policy("conservative")


# === Validation Gate Presets ===

VALIDATION_GATE_STRICT: dict[str, Any] = {
    "max_capping_rate": 0.3,  # Max 30% of layers can be capped
    "max_ppl_degradation": 0.01,  # Max 1% primary-metric degradation (ppl-like)
    "require_branch_balance": True,
}

VALIDATION_GATE_STANDARD: dict[str, Any] = {
    "max_capping_rate": 0.5,  # Max 50% of layers can be capped
    "max_ppl_degradation": 0.02,  # Max 2% primary-metric degradation (ppl-like)
    "require_branch_balance": True,
}

VALIDATION_GATE_PERMISSIVE: dict[str, Any] = {
    "max_capping_rate": 0.7,  # Max 70% of layers can be capped
    "max_ppl_degradation": 0.05,  # Max 5% primary-metric degradation (ppl-like)
    "require_branch_balance": False,
}

DEFAULT_VALIDATION_GATES: dict[str, dict[str, Any]] = {
    "strict": VALIDATION_GATE_STRICT,
    "standard": VALIDATION_GATE_STANDARD,
    "permissive": VALIDATION_GATE_PERMISSIVE,
}


def get_validation_gate(name: str = "standard") -> dict[str, Any]:
    """
    Get validation gate configuration by name.

    Args:
        name: Gate configuration name

    Returns:
        Validation gate configuration
    """
    if name not in DEFAULT_VALIDATION_GATES:
        available = list(DEFAULT_VALIDATION_GATES.keys())
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": available},
        )

    return DEFAULT_VALIDATION_GATES[name].copy()


def enforce_validation_gate(metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    """Enforce validation gate thresholds.

    Raises PolicyViolationError(E503) with a 'violations' list in details
    when one or more constraints are exceeded.
    """
    violations: list[dict[str, Any]] = []

    try:
        caps = float(metrics.get("caps_applied", 0))
        total = float(metrics.get("total_layers", 0))
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
    except Exception:
        # Ignore malformed metrics here; gating purely best-effort
        pass

    try:
        ratio = metrics.get("primary_metric_ratio")
        if isinstance(ratio, int | float) and math.isfinite(float(ratio)):
            limit = float(gate.get("max_ppl_degradation", 1.0))
            # ppl-like ratio: degradation ~ ratio-1; gate on allowed extra
            if ratio - 1.0 > limit:
                violations.append(
                    {
                        "type": "primary_metric_degradation",
                        "actual": float(ratio - 1.0),
                        "limit": limit,
                    }
                )
    except Exception:
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
    """
    Check for drift between tiers.yaml and hardcoded policy fallbacks.

    This helps detect when tiers.yaml has been updated but hardcoded
    fallbacks in this module haven't been synchronized.

    Args:
        silent: If True, don't emit warnings (just return drift info)

    Returns:
        Dict of tier -> list of drift descriptions.
        Empty dict means no drift detected.

    Example:
        >>> drift = check_policy_drift()
        >>> if drift:
        ...     print("Policy drift detected:", drift)
        ...     print("Consider updating hardcoded defaults in policies.py")
    """
    return check_tier_drift(silent=silent)


__all__ = [
    # Spectral policy constants
    "SPECTRAL_CONSERVATIVE",
    "SPECTRAL_BALANCED",
    "SPECTRAL_AGGRESSIVE",
    "SPECTRAL_ATTN_AWARE",
    "DEFAULT_SPECTRAL_POLICIES",
    # RMT policy constants
    "RMT_CONSERVATIVE",
    "RMT_BALANCED",
    "RMT_AGGRESSIVE",
    "DEFAULT_RMT_POLICIES",
    # Variance policy constants
    "VariancePolicyDict",
    "VARIANCE_CONSERVATIVE",
    "VARIANCE_BALANCED",
    "VARIANCE_AGGRESSIVE",
    "DEFAULT_VARIANCE_POLICIES",
    # Validation gate constants
    "VALIDATION_GATE_STRICT",
    "VALIDATION_GATE_STANDARD",
    "VALIDATION_GATE_PERMISSIVE",
    "DEFAULT_VALIDATION_GATES",
    # Utility functions
    "get_spectral_policy",
    "create_custom_spectral_policy",
    "get_policy_for_model_size",
    "get_rmt_policy",
    "create_custom_rmt_policy",
    "get_rmt_policy_for_model_size",
    "get_variance_policy",
    "create_custom_variance_policy",
    "get_variance_policy_for_model_size",
    "get_validation_gate",
    "check_policy_drift",
]
