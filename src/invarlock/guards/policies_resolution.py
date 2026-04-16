"""Policy resolution helpers backed by preset defaults and tier overlays."""

from typing import Literal, cast

from invarlock.core.exceptions import GuardError, ValidationError

from .policies_presets import (
    DEFAULT_RMT_POLICIES,
    DEFAULT_SPECTRAL_POLICIES,
    DEFAULT_VARIANCE_POLICIES,
    RMTPolicyDict,
    SpectralPolicy,
    VariancePolicyDict,
    _is_non_bool_number,
)
from .spectral_policy import normalize_family_caps, normalize_multiple_testing_config
from .tier_config import GuardType, TierName


def _helpers():
    from invarlock.guards import policies as helpers

    return helpers


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
            helpers = _helpers()
            tier_name = cast(TierName, name)
            guard_key: GuardType = "spectral_guard"
            tier_config = helpers.get_tier_guard_config(tier_name, guard_key)
            if tier_config:
                # Update with calibrated values
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
            helpers = _helpers()
            tier_name = cast(TierName, name)
            guard_key: GuardType = "rmt_guard"
            tier_config = helpers.get_tier_guard_config(tier_name, guard_key)
            if tier_config:
                # Update with calibrated values
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
            helpers = _helpers()
            tier_name = cast(TierName, name)
            guard_key: GuardType = "variance_guard"
            tier_config = helpers.get_tier_guard_config(tier_name, guard_key)
            if tier_config:
                # Update with calibrated values
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
                    # Map predictive_one_sided to predictive_gate behavior
                    pass  # This is handled elsewhere in variance guard
        except (AttributeError, RuntimeError, TypeError, ValueError):
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


__all__ = [
    "create_custom_rmt_policy",
    "get_spectral_policy",
    "create_custom_spectral_policy",
    "get_policy_for_model_size",
    "get_rmt_policy",
    "get_rmt_policy_for_model_size",
    "get_variance_policy",
    "create_custom_variance_policy",
    "get_variance_policy_for_model_size",
]
