"""
Tier Configuration Loader
=========================

Loads guard policy defaults from the packaged ``_data/runtime/tiers.yaml``
resource. The packaged resource is the sole runtime authority: missing,
malformed, or incomplete policy data is an explicit configuration error.

These values are operational policy settings. Loading them does not establish
a calibration result or statistical error guarantee for a new model, dataset,
or edit family.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

# Path to bundled tiers.yaml
_TIERS_YAML_PATH = Path(__file__).parent.parent / "_data" / "runtime" / "tiers.yaml"

TierName = Literal["balanced", "conservative", "aggressive"]
GuardType = Literal["spectral_guard", "rmt_guard", "variance_guard"]

_REQUIRED_TIERS = frozenset({"balanced", "conservative", "aggressive"})
_REQUIRED_GUARDS = frozenset({"spectral_guard", "rmt_guard", "variance_guard"})
_REQUIRED_TIER_KEYS = _REQUIRED_GUARDS | {"metrics", "guard_authority"}
_REQUIRED_GUARD_KEYS = {
    "spectral_guard": frozenset(
        {
            "sigma_quantile",
            "deadband",
            "scope",
            "correction_enabled",
            "ignore_preview_inflation",
            "family_caps",
            "max_caps",
            "max_spectral_norm",
            "multiple_testing",
        }
    ),
    "rmt_guard": frozenset(
        {
            "q",
            "deadband",
            "margin",
            "correct",
            "epsilon_default",
            "epsilon_by_family",
        }
    ),
    "variance_guard": frozenset(
        {
            "min_gain",
            "min_rel_gain",
            "max_calib",
            "scope",
            "clamp",
            "deadband",
            "seed",
            "mode",
            "alpha",
            "tie_breaker_deadband",
            "min_abs_adjust",
            "max_scale_step",
            "min_effect_lognll",
            "predictive_one_sided",
            "topk_backstop",
            "max_adjusted_modules",
            "tap",
            "predictive_gate",
            "calibration",
        }
    ),
}


class TierConfigError(RuntimeError):
    """Packaged tier policy could not be loaded or validated."""


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate_spectral_guard_values(prefix: str, data: dict[str, Any]) -> None:
    for key in ("sigma_quantile", "deadband"):
        if not _is_number(data.get(key)):
            raise TierConfigError(f"{prefix} key {key!r} must be numeric")
    if data.get("scope") not in {"ffn", "attn", "all"}:
        raise TierConfigError(f"{prefix} has invalid scope")
    if not isinstance(data.get("correction_enabled"), bool):
        raise TierConfigError(f"{prefix} correction_enabled must be boolean")
    if not isinstance(data.get("ignore_preview_inflation"), bool):
        raise TierConfigError(f"{prefix} ignore_preview_inflation must be boolean")
    max_spectral_norm = data.get("max_spectral_norm")
    if max_spectral_norm is not None and not _is_number(max_spectral_norm):
        raise TierConfigError(f"{prefix} max_spectral_norm must be numeric or null")
    if isinstance(data.get("max_caps"), bool) or not isinstance(
        data.get("max_caps"), int
    ):
        raise TierConfigError(f"{prefix} max_caps must be an integer")
    family_caps = data.get("family_caps")
    if not isinstance(family_caps, dict) or set(family_caps) != {
        "ffn",
        "attn",
        "embed",
        "other",
    }:
        raise TierConfigError(f"{prefix} family_caps has invalid inventory")
    if not all(_is_number(value) for value in family_caps.values()):
        raise TierConfigError(f"{prefix} family_caps values must be numeric")
    multiple_testing = data.get("multiple_testing")
    if not isinstance(multiple_testing, dict) or set(multiple_testing) != {
        "method",
        "alpha",
        "m",
    }:
        raise TierConfigError(f"{prefix} multiple_testing must be a mapping")
    if (
        multiple_testing.get("method") not in {"bh", "bonferroni"}
        or not _is_number(multiple_testing.get("alpha"))
        or isinstance(multiple_testing.get("m"), bool)
        or not isinstance(multiple_testing.get("m"), int)
    ):
        raise TierConfigError(f"{prefix} multiple_testing is invalid")


def _validate_guard_values(tier: str, guard: str, data: dict[str, Any]) -> None:
    prefix = f"Tier {tier!r} guard {guard!r}"
    if guard == "spectral_guard":
        _validate_spectral_guard_values(prefix, data)
        return

    if guard == "rmt_guard":
        q_value = data.get("q")
        if q_value != "auto" and not _is_number(q_value):
            raise TierConfigError(f"{prefix} q must be 'auto' or numeric")
        for key in ("deadband", "margin", "epsilon_default"):
            if not _is_number(data.get(key)):
                raise TierConfigError(f"{prefix} key {key!r} must be numeric")
        if not isinstance(data.get("correct"), bool):
            raise TierConfigError(f"{prefix} correct must be boolean")
        epsilon = data.get("epsilon_by_family")
        if not isinstance(epsilon, dict) or set(epsilon) != {
            "ffn",
            "attn",
            "embed",
            "other",
        }:
            raise TierConfigError(f"{prefix} epsilon_by_family has invalid inventory")
        if not all(_is_number(value) for value in epsilon.values()):
            raise TierConfigError(f"{prefix} epsilon_by_family values must be numeric")
        return

    for key in (
        "min_gain",
        "min_rel_gain",
        "deadband",
        "alpha",
        "tie_breaker_deadband",
        "min_abs_adjust",
        "max_scale_step",
        "min_effect_lognll",
    ):
        if not _is_number(data.get(key)):
            raise TierConfigError(f"{prefix} key {key!r} must be numeric")
    for key in ("max_calib", "seed", "topk_backstop", "max_adjusted_modules"):
        if isinstance(data.get(key), bool) or not isinstance(data.get(key), int):
            raise TierConfigError(f"{prefix} key {key!r} must be an integer")
    if data.get("scope") not in {"ffn", "attn", "both"}:
        raise TierConfigError(f"{prefix} has invalid scope")
    if data.get("mode") not in {"delta", "ci"}:
        raise TierConfigError(f"{prefix} has invalid mode")
    for key in ("predictive_gate", "predictive_one_sided"):
        if not isinstance(data.get(key), bool):
            raise TierConfigError(f"{prefix} key {key!r} must be boolean")
    clamp = data.get("clamp")
    if (
        not isinstance(clamp, list | tuple)
        or len(clamp) != 2
        or not all(_is_number(value) for value in clamp)
    ):
        raise TierConfigError(f"{prefix} clamp must contain two numeric bounds")
    tap = data.get("tap")
    if not (
        isinstance(tap, str)
        or (
            isinstance(tap, list)
            and bool(tap)
            and all(isinstance(value, str) and value for value in tap)
        )
    ):
        raise TierConfigError(f"{prefix} tap must be a string or string list")
    calibration = data.get("calibration")
    if not isinstance(calibration, dict) or set(calibration) != {
        "windows",
        "min_coverage",
        "seed",
    }:
        raise TierConfigError(f"{prefix} calibration has invalid inventory")
    if any(
        isinstance(calibration.get(key), bool)
        or not isinstance(calibration.get(key), int)
        for key in ("windows", "min_coverage", "seed")
    ):
        raise TierConfigError(f"{prefix} calibration values must be integers")


def _load_yaml() -> dict[str, Any]:
    """Load the packaged tier policy or raise an explicit configuration error."""
    try:
        import yaml
    except ImportError as exc:
        raise TierConfigError(
            "PyYAML is required to load packaged tier policy"
        ) from exc
    tier_config_load_errors = (OSError, TypeError, ValueError, yaml.YAMLError)

    if not _TIERS_YAML_PATH.is_file():
        raise TierConfigError(f"Packaged tier policy is missing: {_TIERS_YAML_PATH}")

    try:
        data = yaml.safe_load(_TIERS_YAML_PATH.read_text(encoding="utf-8"))
    except tier_config_load_errors as exc:
        raise TierConfigError("Failed to load packaged tier policy") from exc
    if not isinstance(data, dict):
        raise TierConfigError("Packaged tier policy must be a mapping")
    return data


def _validate_tier_config(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tier_names = set(data)
    if tier_names != _REQUIRED_TIERS:
        missing = sorted(_REQUIRED_TIERS - tier_names)
        unknown = sorted(tier_names - _REQUIRED_TIERS)
        raise TierConfigError(
            "Packaged tier policy has invalid tier inventory "
            f"(missing={missing}, unknown={unknown})"
        )

    result: dict[str, dict[str, Any]] = {}
    for tier in sorted(_REQUIRED_TIERS):
        tier_data = data[tier]
        if not isinstance(tier_data, dict):
            raise TierConfigError(f"Tier {tier!r} must be a mapping")
        if set(tier_data) != _REQUIRED_TIER_KEYS:
            raise TierConfigError(f"Tier {tier!r} has invalid section inventory")
        if not isinstance(tier_data["metrics"], dict) or not tier_data["metrics"]:
            raise TierConfigError(f"Tier {tier!r} metrics must be a non-empty mapping")
        authority = tier_data["guard_authority"]
        if not isinstance(authority, dict) or set(authority) != {
            "spectral",
            "rmt",
            "variance",
        }:
            raise TierConfigError(f"Tier {tier!r} guard_authority is invalid")
        if any(value not in {"observe", "enforce"} for value in authority.values()):
            raise TierConfigError(f"Tier {tier!r} guard_authority is invalid")
        result[tier] = {}
        for guard in sorted(_REQUIRED_GUARDS):
            guard_data = tier_data[guard]
            if not isinstance(guard_data, dict):
                raise TierConfigError(
                    f"Tier {tier!r} guard {guard!r} must be a mapping"
                )
            if set(guard_data) != _REQUIRED_GUARD_KEYS[guard]:
                raise TierConfigError(
                    f"Tier {tier!r} guard {guard!r} has invalid key inventory"
                )
            _validate_guard_values(tier, guard, guard_data)
            result[tier][guard] = deepcopy(guard_data)
    return result


@lru_cache(maxsize=1)
def load_tier_config() -> dict[str, dict[str, Any]]:
    """
    Load and validate tier configuration from the packaged policy resource.

    Returns:
        Dict mapping tier names to guard configurations.
        Structure: {tier: {guard_type: {param: value}}}

    The result is cached for efficiency. Call clear_tier_config_cache() to reload.
    """
    return _validate_tier_config(_load_yaml())


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
        Guard configuration dictionary with the resolved policy values.

    Example:
        >>> config = get_tier_guard_config("balanced", "rmt_guard")
        >>> config["epsilon_by_family"]["ffn"]
        0.01
    """
    config = load_tier_config()
    if tier not in config:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {sorted(config)}")
    tier_config = config[tier]
    if guard not in tier_config:
        raise ValueError(
            f"Unknown guard {guard!r}; expected one of {sorted(tier_config)}"
        )
    return deepcopy(tier_config[guard])


def get_spectral_caps(tier: TierName = "balanced") -> dict[str, float]:
    """Get spectral κ caps for a tier (family -> kappa value)."""
    config = get_tier_guard_config(tier, "spectral_guard")
    return deepcopy(config["family_caps"])


def get_rmt_epsilon(tier: TierName = "balanced") -> dict[str, float]:
    """Get RMT ε values for a tier (family -> epsilon value)."""
    config = get_tier_guard_config(tier, "rmt_guard")
    return deepcopy(config["epsilon_by_family"])


def get_variance_min_effect(tier: TierName = "balanced") -> float:
    """Get VE min_effect_lognll for a tier."""
    config = get_tier_guard_config(tier, "variance_guard")
    return float(config["min_effect_lognll"])


__all__ = [
    "TierConfigError",
    "load_tier_config",
    "clear_tier_config_cache",
    "get_tier_guard_config",
    "get_spectral_caps",
    "get_rmt_epsilon",
    "get_variance_min_effect",
    "TierName",
    "GuardType",
]
