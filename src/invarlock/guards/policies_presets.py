"""Default policy presets shared by guard policy helpers."""

from typing import Any, Literal, NotRequired, TypedDict

from .rmt import RMTPolicyDict
from .spectral_types import SpectralPolicy


def _is_non_bool_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


# === Spectral Guard Policies ===

# Conservative policy - tight control for production use
SPECTRAL_CONSERVATIVE: SpectralPolicy = {
    "sigma_quantile": 0.90,  # Allow only 90% of baseline spectral norm
    "deadband": 0.05,  # 5% deadband - strict threshold
    "scope": "ffn",  # FFN layers only (safest)
    "correction_enabled": True,
    "max_caps": 3,
    "max_spectral_norm": None,
    "multiple_testing": {"method": "bonferroni", "alpha": 0.02, "m": 4},
}

# Balanced policy - good for most use cases
SPECTRAL_BALANCED: SpectralPolicy = {
    "sigma_quantile": 0.95,  # Allow 95% of baseline spectral norm
    "deadband": 0.10,  # 10% deadband - reasonable tolerance
    "scope": "ffn",  # FFN layers only
    "correction_enabled": False,
    "max_caps": 5,
    "max_spectral_norm": None,
    "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
}

# Aggressive policy - for research/experimental use
SPECTRAL_AGGRESSIVE: SpectralPolicy = {
    "sigma_quantile": 0.98,  # Allow 98% of baseline spectral norm
    "deadband": 0.15,  # 15% deadband - more permissive
    "scope": "all",  # All layers including attention
    "correction_enabled": True,
    "max_caps": 8,
    "max_spectral_norm": None,
    "multiple_testing": {"method": "bh", "alpha": 0.1, "m": 4},
}

# Attention-aware policy - includes attention projections
SPECTRAL_ATTN_AWARE: SpectralPolicy = {
    "sigma_quantile": 0.95,  # Standard scaling factor
    "deadband": 0.10,  # Standard deadband
    "scope": "attn",  # Attention layers only
    "correction_enabled": False,
    "max_caps": 5,
    "max_spectral_norm": None,
    "multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4},
}

# === RMT Guard Policies ===

# Conservative RMT policy - tight control for production use
RMT_CONSERVATIVE: RMTPolicyDict = {
    "q": "auto",  # Auto-derive MP aspect ratio from weight shapes
    "deadband": 0.05,  # 5% deadband - strict threshold
    "margin": 1.3,  # Lower margin for conservative detection
    "correct": True,  # Enable automatic correction
    "epsilon_default": 0.06,
    "epsilon_by_family": {"attn": 0.05, "ffn": 0.06, "embed": 0.07, "other": 0.07},
}

# Balanced RMT policy - good for most use cases
RMT_BALANCED: RMTPolicyDict = {
    "q": "auto",  # Auto-derive MP aspect ratio from weight shapes
    "deadband": 0.10,  # 10% deadband - reasonable tolerance
    "margin": 1.5,  # Standard margin for outlier detection
    "correct": False,  # Monitor-only by default
    "epsilon_default": 0.10,
    "epsilon_by_family": {"attn": 0.08, "ffn": 0.10, "embed": 0.12, "other": 0.12},
}

# Aggressive RMT policy - for research/experimental use
RMT_AGGRESSIVE: RMTPolicyDict = {
    "q": "auto",  # Auto-derive MP aspect ratio from weight shapes
    "deadband": 0.15,  # 15% deadband - more permissive
    "margin": 1.8,  # Higher margin allows more deviation
    "correct": True,  # Enable automatic correction
    "epsilon_default": 0.15,
    "epsilon_by_family": {"attn": 0.15, "ffn": 0.15, "embed": 0.15, "other": 0.15},
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

__all__ = [
    "DEFAULT_RMT_POLICIES",
    "DEFAULT_SPECTRAL_POLICIES",
    "DEFAULT_VARIANCE_POLICIES",
    "RMT_AGGRESSIVE",
    "RMT_BALANCED",
    "RMT_CONSERVATIVE",
    "SPECTRAL_AGGRESSIVE",
    "SPECTRAL_ATTN_AWARE",
    "SPECTRAL_BALANCED",
    "SPECTRAL_CONSERVATIVE",
    "VARIANCE_AGGRESSIVE",
    "VARIANCE_BALANCED",
    "VARIANCE_CONSERVATIVE",
    "VariancePolicyDict",
    "VariancePolicyRequired",
    "_is_non_bool_number",
]
