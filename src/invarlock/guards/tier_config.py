"""
Tier Configuration Loader
=========================

Loads guard policy values from the calibrated tiers.yaml source of truth.
Provides fallback to hardcoded defaults if the YAML file is unavailable.

The tiers.yaml file contains calibration values derived from pilot runs
(November 2025 certification). This loader ensures code and config stay
synchronized.
"""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Path to bundled tiers.yaml
_TIERS_YAML_PATH = Path(__file__).parent.parent / "_data" / "runtime" / "tiers.yaml"

# Hardcoded fallbacks (November 2025 calibration values)
# These are used if tiers.yaml cannot be loaded
_FALLBACK_CONFIG: dict[str, dict[str, Any]] = {
    "balanced": {
        "variance_guard": {
            "deadband": 0.02,
            "min_abs_adjust": 0.012,
            "max_scale_step": 0.03,
            "min_effect_lognll": 0.0009,
            "predictive_one_sided": True,
            "topk_backstop": 1,
            "max_adjusted_modules": 1,
        },
        "spectral_guard": {
            "sigma_quantile": 0.95,
            "deadband": 0.10,
            "scope": "all",
            "max_caps": 5,
            "max_spectral_norm": None,
            "family_caps": {
                "ffn": 2.5,
                "attn": 2.8,
                "embed": 3.0,
                "other": 3.0,
            },
            "multiple_testing": {
                "method": "bh",
                "alpha": 0.05,
                "m": 4,
            },
        },
        "rmt_guard": {
            "deadband": 0.10,
            "margin": 1.5,
            "epsilon_default": 0.10,
            "epsilon_by_family": {
                "ffn": 0.10,
                "attn": 0.08,
                "embed": 0.12,
                "other": 0.12,
            },
        },
    },
    "conservative": {
        "variance_guard": {
            "deadband": 0.03,
            "min_abs_adjust": 0.02,
            "max_scale_step": 0.015,
            "min_effect_lognll": 0.0018,
            "predictive_one_sided": False,
            "topk_backstop": 0,
            "max_adjusted_modules": 0,
        },
        "spectral_guard": {
            "sigma_quantile": 0.90,
            "deadband": 0.05,
            "scope": "ffn",
            "max_caps": 3,
            "family_caps": {
                "ffn": 2.3,
                "attn": 2.6,
                "embed": 2.8,
                "other": 2.8,
            },
            "multiple_testing": {
                "method": "bonferroni",
                "alpha": 0.02,
                "m": 4,
            },
        },
        "rmt_guard": {
            "deadband": 0.05,
            "margin": 1.3,
            "epsilon_default": 0.06,
            "epsilon_by_family": {
                "ffn": 0.06,
                "attn": 0.05,
                "embed": 0.07,
                "other": 0.07,
            },
        },
    },
    "aggressive": {
        "variance_guard": {
            "deadband": 0.12,
            "min_effect_lognll": 0.0005,
        },
        "spectral_guard": {
            "sigma_quantile": 0.98,
            "deadband": 0.15,
            "scope": "all",
            "max_caps": 8,
            "family_caps": {
                "ffn": 3.0,
                "attn": 3.2,
                "embed": 3.5,
                "other": 3.5,
            },
            "multiple_testing": {
                "method": "bh",
                "alpha": 0.1,
                "m": 4,
            },
        },
        "rmt_guard": {
            "deadband": 0.15,
            "margin": 1.8,
            "epsilon_default": 0.15,
            "epsilon_by_family": {
                "ffn": 0.15,
                "attn": 0.15,
                "embed": 0.15,
                "other": 0.15,
            },
        },
    },
}

TierName = Literal["balanced", "conservative", "aggressive"]
GuardType = Literal["spectral_guard", "rmt_guard", "variance_guard"]


def _load_yaml() -> dict[str, Any] | None:
    """Load tiers.yaml if available."""
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed; using fallback tier config")
        return None

    if not _TIERS_YAML_PATH.exists():
        logger.debug("tiers.yaml not found at %s; using fallback", _TIERS_YAML_PATH)
        return None

    try:
        with open(_TIERS_YAML_PATH) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            logger.warning("tiers.yaml did not parse as dict; using fallback")
            return None
        return data
    except Exception as exc:
        logger.warning("Failed to load tiers.yaml: %s; using fallback", exc)
        return None


@lru_cache(maxsize=1)
def load_tier_config() -> dict[str, dict[str, Any]]:
    """
    Load tier configuration from tiers.yaml with fallback to hardcoded defaults.

    Returns:
        Dict mapping tier names to guard configurations.
        Structure: {tier: {guard_type: {param: value}}}

    The result is cached for efficiency. Call clear_tier_config_cache() to reload.
    """
    yaml_data = _load_yaml()
    if yaml_data is None:
        return _FALLBACK_CONFIG.copy()

    # Merge yaml data with fallbacks for any missing keys
    result: dict[str, dict[str, Any]] = {}
    for tier in ("balanced", "conservative", "aggressive"):
        tier_data = yaml_data.get(tier, {})
        fallback_tier = _FALLBACK_CONFIG.get(tier, {})

        result[tier] = {}
        for guard in ("spectral_guard", "rmt_guard", "variance_guard"):
            guard_data = tier_data.get(guard, {})
            fallback_guard = fallback_tier.get(guard, {})

            # Deep merge: YAML values override fallbacks
            merged = _deep_merge(fallback_guard, guard_data)
            result[tier][guard] = merged

    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def clear_tier_config_cache() -> None:
    """Clear cached tier config to force reload on next access."""
    load_tier_config.cache_clear()


def get_tier_guard_config(
    tier: TierName,
    guard: GuardType,
) -> dict[str, Any]:
    """
    Get configuration for a specific tier and guard type.

    Args:
        tier: Tier name ("balanced", "conservative", "aggressive")
        guard: Guard type ("spectral_guard", "rmt_guard", "variance_guard")

    Returns:
        Guard configuration dict with all calibrated values.

    Example:
        >>> config = get_tier_guard_config("balanced", "rmt_guard")
        >>> config["epsilon_by_family"]["ffn"]
        0.10
    """
    config = load_tier_config()
    tier_config = config.get(tier, config.get("balanced", {}))
    return tier_config.get(guard, {}).copy()


def get_spectral_caps(tier: TierName = "balanced") -> dict[str, float]:
    """Get spectral κ caps for a tier (family -> kappa value)."""
    config = get_tier_guard_config(tier, "spectral_guard")
    return config.get("family_caps", {}).copy()


def get_rmt_epsilon(tier: TierName = "balanced") -> dict[str, float]:
    """Get RMT ε values for a tier (family -> epsilon value)."""
    config = get_tier_guard_config(tier, "rmt_guard")
    return config.get("epsilon_by_family", {}).copy()


def get_variance_min_effect(tier: TierName = "balanced") -> float:
    """Get VE min_effect_lognll for a tier."""
    config = get_tier_guard_config(tier, "variance_guard")
    return config.get("min_effect_lognll", 0.0009)


def check_drift(
    *,
    silent: bool = False,
) -> dict[str, list[str]]:
    """
    Check for drift between tiers.yaml and hardcoded fallbacks.

    This helps detect when tiers.yaml has been updated but hardcoded
    fallbacks haven't been synchronized.

    Args:
        silent: If True, don't emit warnings (just return drift info)

    Returns:
        Dict of tier -> list of drift descriptions.
        Empty dict means no drift detected.

    Example:
        >>> drift = check_drift()
        >>> if drift:
        ...     print("Drift detected:", drift)
    """
    yaml_data = _load_yaml()
    if yaml_data is None:
        # No YAML means we're using fallbacks exclusively
        return {}

    drifts: dict[str, list[str]] = {}

    for tier in ("balanced", "conservative", "aggressive"):
        tier_drifts: list[str] = []
        yaml_tier = yaml_data.get(tier, {})
        fallback_tier = _FALLBACK_CONFIG.get(tier, {})

        for guard in ("spectral_guard", "rmt_guard", "variance_guard"):
            yaml_guard = yaml_tier.get(guard, {})
            fallback_guard = fallback_tier.get(guard, {})

            drift_keys = _find_drifts(yaml_guard, fallback_guard, prefix=f"{guard}.")
            tier_drifts.extend(drift_keys)

        if tier_drifts:
            drifts[tier] = tier_drifts
            if not silent:
                warnings.warn(
                    f"Tier config drift detected in '{tier}': "
                    f"{', '.join(tier_drifts[:3])}{'...' if len(tier_drifts) > 3 else ''}. "
                    "Consider updating hardcoded fallbacks in tier_config.py",
                    UserWarning,
                    stacklevel=2,
                )

    return drifts


def _find_drifts(
    yaml_data: dict[str, Any],
    fallback_data: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    """Find keys where YAML differs from fallback."""
    drifts: list[str] = []

    all_keys = set(yaml_data.keys()) | set(fallback_data.keys())
    for key in all_keys:
        yaml_val = yaml_data.get(key)
        fallback_val = fallback_data.get(key)

        # Skip if fallback doesn't have this key (YAML-only keys are fine)
        if fallback_val is None:
            continue

        if isinstance(yaml_val, dict) and isinstance(fallback_val, dict):
            nested = _find_drifts(yaml_val, fallback_val, prefix=f"{prefix}{key}.")
            drifts.extend(nested)
        elif yaml_val != fallback_val:
            drifts.append(f"{prefix}{key}: yaml={yaml_val} vs fallback={fallback_val}")

    return drifts


__all__ = [
    "load_tier_config",
    "clear_tier_config_cache",
    "get_tier_guard_config",
    "get_spectral_caps",
    "get_rmt_epsilon",
    "get_variance_min_effect",
    "check_drift",
    "TierName",
    "GuardType",
]
