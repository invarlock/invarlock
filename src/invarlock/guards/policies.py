"""Guard policy presets, resolution helpers, and validation gates."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Literal, NotRequired, TypedDict, cast

from invarlock.core.exceptions import GuardError, ValidationError

from . import policy_validation as _policy_validation
from .rmt_policy import RMTPolicyDict
from .spectral_policy import normalize_family_caps, normalize_multiple_testing_config
from .tier_config import (
    TierConfigError,
    TierName,
    get_tier_guard_config,
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


SPECTRAL_ATTN_AWARE: SpectralPolicy = {
    "sigma_quantile": 0.95,
    "deadband": 0.10,
    "scope": "attn",
    "correction_enabled": False,
    "max_caps": 5,
    "max_spectral_norm": None,
    "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
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
    max_adjusted_modules: NotRequired[int]
    predictive_gate: NotRequired[bool]
    predictive_one_sided: NotRequired[bool]
    monitor_only: NotRequired[bool]
    calibration_max_seq_len: NotRequired[int]
    target_modules: NotRequired[list[str]]
    tap: NotRequired[str | list[str]]


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


def get_spectral_policy(name: str = "balanced") -> SpectralPolicy:
    """
    Get a spectral policy by name.

    Loads values from the packaged ``_data/runtime/tiers.yaml`` policy resource
    and fails explicitly if that resource cannot be loaded or validated.
    """
    if name == "attn_aware":
        return deepcopy(SPECTRAL_ATTN_AWARE)
    tier_names = ("balanced", "conservative", "aggressive")
    if name not in tier_names:
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": [*tier_names, "attn_aware"]},
        )

    try:
        tier_config = get_tier_guard_config(cast(TierName, name), "spectral_guard")
        max_spectral_norm = tier_config.get("max_spectral_norm")
        policy = SpectralPolicy(
            sigma_quantile=float(tier_config["sigma_quantile"]),
            deadband=float(tier_config["deadband"]),
            scope=str(tier_config["scope"]),
            correction_enabled=bool(tier_config["correction_enabled"]),
            family_caps=normalize_family_caps(tier_config["family_caps"]),
            ignore_preview_inflation=bool(tier_config["ignore_preview_inflation"]),
            max_caps=int(tier_config["max_caps"]),
            max_spectral_norm=(
                float(max_spectral_norm) if max_spectral_norm is not None else None
            ),
            multiple_testing=normalize_multiple_testing_config(
                tier_config["multiple_testing"]
            ),
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise TierConfigError(
            f"Failed to resolve packaged spectral policy {name!r}"
        ) from exc
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


def get_rmt_policy(name: str = "balanced") -> RMTPolicyDict:
    """
    Get an RMT policy by name.

    Loads values from the packaged ``_data/runtime/tiers.yaml`` policy resource
    and fails explicitly if that resource cannot be loaded or validated.
    """
    tier_names = ("balanced", "conservative", "aggressive")
    if name not in tier_names:
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": list(tier_names)},
        )

    try:
        tier_config = get_tier_guard_config(cast(TierName, name), "rmt_guard")
        q_value = tier_config["q"]
        epsilon_by_family = tier_config["epsilon_by_family"]
        policy = RMTPolicyDict(
            q=("auto" if q_value == "auto" else float(q_value)),
            deadband=float(tier_config["deadband"]),
            margin=float(tier_config["margin"]),
            correct=bool(tier_config["correct"]),
            epsilon_default=float(tier_config["epsilon_default"]),
            epsilon_by_family={
                str(family): float(value) for family, value in epsilon_by_family.items()
            },
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise TierConfigError(
            f"Failed to resolve packaged RMT policy {name!r}"
        ) from exc
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


def get_variance_policy(name: str = "balanced") -> VariancePolicyDict:
    """
    Get a variance policy by name.

    Loads values from the packaged ``_data/runtime/tiers.yaml`` policy resource
    and fails explicitly if that resource cannot be loaded or validated.
    """
    tier_names = ("balanced", "conservative", "aggressive")
    if name not in tier_names:
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": list(tier_names)},
        )

    try:
        tier_config = get_tier_guard_config(cast(TierName, name), "variance_guard")
        clamp = tier_config["clamp"]
        calibration = tier_config["calibration"]
        tap = tier_config["tap"]
        policy = VariancePolicyDict(
            min_gain=float(tier_config["min_gain"]),
            min_rel_gain=float(tier_config["min_rel_gain"]),
            max_calib=int(tier_config["max_calib"]),
            scope=cast(Literal["ffn", "attn", "both"], tier_config["scope"]),
            clamp=(float(clamp[0]), float(clamp[1])),
            deadband=float(tier_config["deadband"]),
            seed=int(tier_config["seed"]),
            mode=cast(Literal["delta", "ci"], tier_config["mode"]),
            alpha=float(tier_config["alpha"]),
            tie_breaker_deadband=float(tier_config["tie_breaker_deadband"]),
            min_effect_lognll=float(tier_config["min_effect_lognll"]),
            min_abs_adjust=float(tier_config["min_abs_adjust"]),
            max_scale_step=float(tier_config["max_scale_step"]),
            topk_backstop=int(tier_config["topk_backstop"]),
            max_adjusted_modules=int(tier_config["max_adjusted_modules"]),
            predictive_gate=bool(tier_config["predictive_gate"]),
            predictive_one_sided=bool(tier_config["predictive_one_sided"]),
            tap=(list(tap) if isinstance(tap, list) else str(tap)),
            calibration={
                key: int(calibration[key])
                for key in ("windows", "min_coverage", "seed")
            },
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise TierConfigError(
            f"Failed to resolve packaged variance policy {name!r}"
        ) from exc
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
    return _policy_validation.get_validation_gate(
        name,
        gates=DEFAULT_VALIDATION_GATES,
    )


def enforce_validation_gate(metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    """Enforce validation gate thresholds."""
    _policy_validation.enforce_validation_gate(metrics, gate)


__all__ = [
    "DEFAULT_VALIDATION_GATES",
    "NotRequired",
    "RMTPolicyDict",
    "SPECTRAL_ATTN_AWARE",
    "SpectralPolicy",
    "TypedDict",
    "VALIDATION_GATE_PERMISSIVE",
    "VALIDATION_GATE_STANDARD",
    "VALIDATION_GATE_STRICT",
    "VariancePolicyDict",
    "VariancePolicyRequired",
    "_is_non_bool_number",
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
