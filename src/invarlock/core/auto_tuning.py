"""
InvarLock Auto-Tuning System
========================

Tier-based policy resolution for GuardChain safety postures.
Maps tier settings (conservative/balanced/aggressive) to specific guard parameters.
"""

import copy
from typing import Any

__all__ = ["resolve_tier_policies", "TIER_POLICIES", "EDIT_ADJUSTMENTS"]


# Base tier policy mappings
TIER_POLICIES: dict[str, dict[str, dict[str, Any]]] = {
    "conservative": {
        "metrics": {
            "pm_ratio": {
                # Lower token floor for CI-friendly smokes while retaining
                # dataset-fraction guardrails via min_token_fraction.
                "min_tokens": 20000,
                "hysteresis_ratio": 0.002,
                "min_token_fraction": 0.01,
            },
            "accuracy": {
                "delta_min_pp": -0.5,
                "min_examples": 200,
                "hysteresis_delta_pp": 0.1,
                "min_examples_fraction": 0.01,
            },
        },
        "spectral": {
            "sigma_quantile": 0.90,  # Tighter spectral percentile
            "deadband": 0.05,  # Smaller no-op zone
            "scope": "ffn",
            "family_caps": {
                "ffn": {"kappa": 2.3},
                "attn": {"kappa": 2.6},
                "embed": {"kappa": 2.8},
                "other": {"kappa": 2.8},
            },
            "ignore_preview_inflation": True,
            "max_caps": 3,
            "multiple_testing": {"method": "bonferroni", "alpha": 0.02, "m": 4},
        },
        "rmt": {
            "margin": 1.40,  # Lower spike allowance
            "deadband": 0.10,  # Standard deadband
            "correct": True,
            "epsilon": {"attn": 0.05, "ffn": 0.06, "embed": 0.07, "other": 0.07},
        },
        "variance": {
            "min_gain": 0.01,
            "min_rel_gain": 0.002,
            "max_calib": 160,
            "scope": "ffn",
            "clamp": (0.85, 1.12),
            "deadband": 0.03,
            "seed": 42,
            "mode": "ci",
            "alpha": 0.05,
            "tie_breaker_deadband": 0.005,
            "min_effect_lognll": 0.0018,
            "calibration": {
                "windows": 10,
                "min_coverage": 8,
                "seed": 42,
            },
            "min_abs_adjust": 0.02,
            "max_scale_step": 0.015,
            "topk_backstop": 0,
            "max_adjusted_modules": 0,
            "predictive_one_sided": False,
            "tap": "transformer.h.*.mlp.c_proj",
            "predictive_gate": True,
        },
    },
    "balanced": {
        "metrics": {
            "pm_ratio": {
                "min_tokens": 50000,
                "hysteresis_ratio": 0.002,
                "min_token_fraction": 0.01,
            },
            "accuracy": {
                "delta_min_pp": -1.0,
                "min_examples": 200,
                "hysteresis_delta_pp": 0.1,
                "min_examples_fraction": 0.01,
            },
        },
        "spectral": {
            "sigma_quantile": 0.95,  # Default spectral percentile
            "deadband": 0.10,  # Standard no-op zone
            "scope": "all",
            "family_caps": {
                "ffn": {"kappa": 2.5},
                "attn": {"kappa": 2.8},
                "embed": {"kappa": 3.0},
                "other": {"kappa": 3.0},
            },
            "ignore_preview_inflation": True,
            "max_caps": 5,
            "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
            "max_spectral_norm": None,
        },
        "rmt": {
            "margin": 1.50,  # Default spike allowance
            "deadband": 0.10,  # Standard deadband
            "correct": True,
            "epsilon": {"attn": 0.08, "ffn": 0.10, "embed": 0.12, "other": 0.12},
        },
        "variance": {
            "min_gain": 0.0,
            "min_rel_gain": 0.001,
            "max_calib": 200,
            "scope": "ffn",
            "clamp": (0.85, 1.12),
            "deadband": 0.02,
            "seed": 123,
            "mode": "ci",
            "alpha": 0.05,
            "tie_breaker_deadband": 0.001,
            "min_effect_lognll": 0.0009,
            "min_abs_adjust": 0.012,
            "max_scale_step": 0.03,
            "topk_backstop": 1,
            "max_adjusted_modules": 1,
            "predictive_one_sided": True,
            "tap": "transformer.h.*.mlp.c_proj",
            "predictive_gate": True,
            "calibration": {
                "windows": 8,
                "min_coverage": 6,
                "seed": 123,
            },
        },
    },
    "aggressive": {
        "metrics": {
            "pm_ratio": {
                "min_tokens": 50000,
                "hysteresis_ratio": 0.002,
                "min_token_fraction": 0.01,
            },
            "accuracy": {
                "delta_min_pp": -2.0,
                "min_examples": 200,
                "hysteresis_delta_pp": 0.1,
                "min_examples_fraction": 0.01,
            },
        },
        "spectral": {
            "sigma_quantile": 0.98,  # Looser spectral percentile
            "deadband": 0.15,  # Larger no-op zone
            "scope": "ffn",
            "family_caps": {
                "ffn": {"kappa": 3.0},
                "attn": {"kappa": 3.5},
                "embed": {"kappa": 2.5},
                "other": {"kappa": 3.5},
            },
            "ignore_preview_inflation": True,
        },
        "rmt": {
            "margin": 1.70,  # Higher spike allowance
            "deadband": 0.15,  # Larger deadband
            "correct": True,
            "epsilon": {"attn": 0.15, "ffn": 0.15, "embed": 0.15, "other": 0.15},
        },
        "variance": {
            "min_gain": 0.0,
            "min_rel_gain": 0.0025,
            "max_calib": 240,
            "scope": "both",
            "clamp": (0.3, 3.0),
            "deadband": 0.12,
            "seed": 456,
            "mode": "ci",
            "alpha": 0.05,
            "tie_breaker_deadband": 0.0005,
            "min_effect_lognll": 0.0005,
            "tap": ["transformer.h.*.mlp.c_proj", "transformer.h.*.attn.c_proj"],
            "predictive_gate": True,
            "calibration": {
                "windows": 6,
                "min_coverage": 4,
                "seed": 456,
            },
        },
    },
}

# Edit-specific policy adjustments
EDIT_ADJUSTMENTS: dict[str, dict[str, dict[str, Any]]] = {
    "quant_rtn": {"rmt": {"deadband": 0.15}}
}


def resolve_tier_policies(
    tier: str,
    edit_name: str | None = None,
    explicit_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Resolve tier-based guard policies with edit-specific adjustments and explicit overrides.

    Args:
        tier: Policy tier ("conservative", "balanced", "aggressive")
        edit_name: Name of the edit being applied (for edit-specific adjustments)
        explicit_overrides: Explicit guard policy overrides from config

    Returns:
        Dictionary mapping guard names to their resolved policy parameters

    Raises:
        ValueError: If tier is not recognized
    """
    if tier not in TIER_POLICIES:
        raise ValueError(
            f"Unknown tier '{tier}'. Valid tiers: {list(TIER_POLICIES.keys())}"
        )

    # Start with base tier policies
    policies: dict[str, dict[str, Any]] = copy.deepcopy(TIER_POLICIES[tier])

    # Apply edit-specific adjustments
    if edit_name and edit_name in EDIT_ADJUSTMENTS:
        edit_adjustments = EDIT_ADJUSTMENTS[edit_name]
        for guard_name, adjustments in edit_adjustments.items():
            if guard_name in policies:
                guard_policy = policies[guard_name]
                assert isinstance(guard_policy, dict)
                guard_policy.update(adjustments)

    # Apply explicit overrides (highest precedence)
    if explicit_overrides:
        for guard_name, overrides in explicit_overrides.items():
            if guard_name in policies:
                guard_policy = policies[guard_name]
                assert isinstance(guard_policy, dict)
                guard_policy.update(overrides)
            else:
                # Create new guard policy if not in base tier
                policies[guard_name] = overrides.copy()

    return policies


def get_tier_summary(tier: str, edit_name: str | None = None) -> dict[str, Any]:
    """
    Get a summary of what policies will be applied for a given tier and edit.

    Args:
        tier: Policy tier
        edit_name: Optional edit name for edit-specific adjustments

    Returns:
        Summary dictionary with tier info and resolved policies
    """
    try:
        policies = resolve_tier_policies(tier, edit_name)

        return {
            "tier": tier,
            "edit_name": edit_name,
            "policies": policies,
            "description": _get_tier_description(tier),
        }
    except ValueError as e:
        return {
            "tier": tier,
            "edit_name": edit_name,
            "error": str(e),
            "valid_tiers": list(TIER_POLICIES.keys()),
        }


def _get_tier_description(tier: str) -> str:
    """Get human-readable description of tier behavior."""
    descriptions = {
        "conservative": "Safest posture with tighter guard thresholds (more likely to cap/rollback)",
        "balanced": "Default safety posture with standard guard thresholds",
        "aggressive": "Looser guard thresholds with more headroom (fewer caps)",
    }
    return descriptions.get(tier, f"Unknown tier: {tier}")


def validate_tier_config(config: Any) -> tuple[bool, str | None]:
    """
    Validate tier configuration for correctness.

    Args:
        config: Auto-tuning configuration (can be any type for validation)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "Config must be a dictionary"

    if "tier" not in config:
        return False, "Missing 'tier' in auto configuration"

    tier = config["tier"]
    if tier not in TIER_POLICIES:
        valid_options = list(TIER_POLICIES.keys())
        return False, f"Invalid tier '{tier}'. Valid options: {valid_options}"

    if "enabled" in config and not isinstance(config["enabled"], bool):
        return False, "'enabled' must be a boolean"

    if "probes" in config and not isinstance(config["probes"], int):
        return False, "'probes' must be an integer"

    return True, None
