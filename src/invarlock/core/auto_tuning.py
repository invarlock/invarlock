"""
InvarLock Auto-Tuning System
========================

Tier-based policy resolution for GuardChain safety postures.
Maps tier settings (conservative/balanced/aggressive) to specific guard parameters.
"""

import copy
import os
from functools import lru_cache
from importlib import resources as _ires
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "clear_tier_policies_cache",
    "get_tier_policies",
    "resolve_tier_policies",
    "TIER_POLICIES",
    "EDIT_ADJUSTMENTS",
]


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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_runtime_yaml(
    config_root: str | None, *rel_parts: str
) -> dict[str, Any] | None:
    """Load YAML from runtime config locations.

    Search order:
      1) $INVARLOCK_CONFIG_ROOT/runtime/...
      2) invarlock._data.runtime package resources
    """
    if config_root:
        p = Path(config_root) / "runtime"
        for part in rel_parts:
            p = p / part
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise ValueError("Runtime YAML must be a mapping")
            return data

    try:
        base = _ires.files("invarlock._data.runtime")
        res = base
        for part in rel_parts:
            res = res.joinpath(part)
        if getattr(res, "is_file", None) and res.is_file():  # type: ignore[attr-defined]
            text = res.read_text(encoding="utf-8")  # type: ignore[assignment]
            data = yaml.safe_load(text) or {}
            if not isinstance(data, dict):
                raise ValueError("Runtime YAML must be a mapping")
            return data
    except Exception:
        return None

    return None


def _normalize_family_caps(caps: Any) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    if not isinstance(caps, dict):
        return normalized
    for family, value in caps.items():
        family_key = str(family)
        if isinstance(value, dict):
            kappa = value.get("kappa")
            if isinstance(kappa, int | float):
                normalized[family_key] = {"kappa": float(kappa)}
        elif isinstance(value, int | float):
            normalized[family_key] = {"kappa": float(value)}
    return normalized


def _normalize_multiple_testing(mt: Any) -> dict[str, Any]:
    if not isinstance(mt, dict):
        return {}
    out: dict[str, Any] = {}
    method = mt.get("method")
    if method is not None:
        out["method"] = str(method).lower()
    alpha = mt.get("alpha")
    try:
        if alpha is not None:
            out["alpha"] = float(alpha)
    except Exception:
        pass
    m_val = mt.get("m")
    try:
        if m_val is not None:
            out["m"] = int(m_val)
    except Exception:
        pass
    return out


def _tier_entry_to_policy(tier_entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map a tiers.yaml entry to the canonical policy shape."""
    out: dict[str, dict[str, Any]] = {}

    metrics = tier_entry.get("metrics")
    if isinstance(metrics, dict):
        out["metrics"] = copy.deepcopy(metrics)

    spectral_src = tier_entry.get("spectral") or tier_entry.get("spectral_guard")
    if isinstance(spectral_src, dict):
        spectral = copy.deepcopy(spectral_src)
        if "family_caps" in spectral:
            spectral["family_caps"] = _normalize_family_caps(
                spectral.get("family_caps")
            )
        if "multiple_testing" in spectral:
            spectral["multiple_testing"] = _normalize_multiple_testing(
                spectral.get("multiple_testing")
            )
        out["spectral"] = spectral

    rmt_src = tier_entry.get("rmt") or tier_entry.get("rmt_guard")
    if isinstance(rmt_src, dict):
        rmt = copy.deepcopy(rmt_src)
        eps = rmt.get("epsilon_by_family")
        if isinstance(eps, dict):
            rmt["epsilon_by_family"] = {
                str(k): float(v) for k, v in eps.items() if isinstance(v, int | float)
            }
            # Backward-compat: keep epsilon alias
            rmt["epsilon"] = dict(rmt["epsilon_by_family"])
        out["rmt"] = rmt

    variance_src = tier_entry.get("variance") or tier_entry.get("variance_guard")
    if isinstance(variance_src, dict):
        out["variance"] = copy.deepcopy(variance_src)

    return out


@lru_cache(maxsize=8)
def _load_tier_policies_cached(config_root: str | None) -> dict[str, dict[str, Any]]:
    tiers = _load_runtime_yaml(config_root, "tiers.yaml") or {}
    merged: dict[str, dict[str, Any]] = {}

    # Start from defaults, then overlay tiers.yaml per-tier.
    for tier_name, defaults in TIER_POLICIES.items():
        merged[str(tier_name).lower()] = copy.deepcopy(defaults)

    for tier_name, entry in tiers.items():
        if not isinstance(entry, dict):
            continue
        tier_key = str(tier_name).lower()
        resolved_entry = _tier_entry_to_policy(entry)
        if tier_key not in merged:
            merged[tier_key] = {}
        merged[tier_key] = _deep_merge(merged[tier_key], resolved_entry)

    return merged


def get_tier_policies(*, config_root: str | None = None) -> dict[str, dict[str, Any]]:
    """Return tier policies loaded from runtime tiers.yaml (with safe defaults)."""
    root = config_root
    if root is None:
        root = os.getenv("INVARLOCK_CONFIG_ROOT") or None
    return _load_tier_policies_cached(root)


def clear_tier_policies_cache() -> None:
    _load_tier_policies_cached.cache_clear()


def _load_profile_overrides(
    profile: str | None, *, config_root: str | None
) -> dict[str, Any]:
    if not profile:
        return {}
    prof = str(profile).strip().lower()
    candidate = _load_runtime_yaml(config_root, "profiles", f"{prof}.yaml")
    if candidate is None and prof == "ci":
        candidate = _load_runtime_yaml(config_root, "profiles", "ci_cpu.yaml") or {}
    if not isinstance(candidate, dict):
        return {}
    return candidate


def resolve_tier_policies(
    tier: str,
    edit_name: str | None = None,
    explicit_overrides: dict[str, dict[str, Any]] | None = None,
    *,
    profile: str | None = None,
    config_root: str | None = None,
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
    tier_key = str(tier).lower()
    tier_policies = get_tier_policies(config_root=config_root)
    if tier_key not in tier_policies:
        raise ValueError(
            f"Unknown tier '{tier}'. Valid tiers: {list(tier_policies.keys())}"
        )

    # Start with base tier policies
    policies: dict[str, dict[str, Any]] = copy.deepcopy(tier_policies[tier_key])

    # Apply profile overrides (when available)
    overrides = _load_profile_overrides(profile, config_root=config_root)
    guards = overrides.get("guards") if isinstance(overrides, dict) else None
    if isinstance(guards, dict):
        for guard_name, guard_overrides in guards.items():
            key = str(guard_name).lower()
            if not isinstance(guard_overrides, dict):
                continue
            if key in policies and isinstance(policies[key], dict):
                policies[key] = _deep_merge(policies[key], guard_overrides)
            else:
                policies[key] = copy.deepcopy(guard_overrides)

    # Apply edit-specific adjustments
    if edit_name and edit_name in EDIT_ADJUSTMENTS:
        edit_adjustments = EDIT_ADJUSTMENTS[edit_name]
        for guard_name, adjustments in edit_adjustments.items():
            if guard_name in policies and isinstance(policies.get(guard_name), dict):
                policies[guard_name] = _deep_merge(policies[guard_name], adjustments)

    # Apply explicit overrides (highest precedence)
    if explicit_overrides:
        for guard_name, overrides in explicit_overrides.items():
            if guard_name in policies and isinstance(policies.get(guard_name), dict):
                if isinstance(overrides, dict):
                    policies[guard_name] = _deep_merge(policies[guard_name], overrides)
            elif isinstance(overrides, dict):
                # Create new guard policy if not in base tier
                policies[guard_name] = copy.deepcopy(overrides)

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
            "valid_tiers": list(get_tier_policies().keys()),
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
    tier_policies = get_tier_policies()
    if tier not in tier_policies:
        valid_options = list(tier_policies.keys())
        return False, f"Invalid tier '{tier}'. Valid options: {valid_options}"

    if "enabled" in config and not isinstance(config["enabled"], bool):
        return False, "'enabled' must be a boolean"

    if "probes" in config and not isinstance(config["probes"], int):
        return False, "'probes' must be an integer"

    return True, None
