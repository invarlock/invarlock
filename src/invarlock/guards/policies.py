"""Facade for guard policy presets and helpers."""

from __future__ import annotations

import builtins as _builtins

from . import tier_config as _tier_config
from .policies_presets import (
    DEFAULT_RMT_POLICIES,
    DEFAULT_SPECTRAL_POLICIES,
    DEFAULT_VARIANCE_POLICIES,
    RMT_AGGRESSIVE,
    RMT_BALANCED,
    RMT_CONSERVATIVE,
    SPECTRAL_AGGRESSIVE,
    SPECTRAL_ATTN_AWARE,
    SPECTRAL_BALANCED,
    SPECTRAL_CONSERVATIVE,
    VARIANCE_AGGRESSIVE,
    VARIANCE_BALANCED,
    VARIANCE_CONSERVATIVE,
    VariancePolicyDict,
    VariancePolicyRequired,
    _is_non_bool_number,
)
from .policies_resolution import (
    create_custom_rmt_policy,
    create_custom_spectral_policy,
    create_custom_variance_policy,
    get_policy_for_model_size,
    get_rmt_policy,
    get_rmt_policy_for_model_size,
    get_spectral_policy,
    get_variance_policy,
    get_variance_policy_for_model_size,
)
from .policies_validation import (
    DEFAULT_VALIDATION_GATES,
    VALIDATION_GATE_PERMISSIVE,
    VALIDATION_GATE_STANDARD,
    VALIDATION_GATE_STRICT,
    check_policy_drift,
    enforce_validation_gate,
    get_validation_gate,
)

try:
    from typing import NotRequired as _NotRequired
    from typing import TypedDict as _TypedDict
except _builtins.ImportError:  # pragma: no cover - Python <3.11 fallback
    import importlib

    _compat_typing = importlib.import_module("typing" + "_extensions")

    _NotRequired = _compat_typing.NotRequired
    _TypedDict = _compat_typing.TypedDict

NotRequired = _NotRequired
TypedDict = _TypedDict

get_tier_guard_config = _tier_config.get_tier_guard_config

__all__ = [
    "DEFAULT_RMT_POLICIES",
    "DEFAULT_SPECTRAL_POLICIES",
    "DEFAULT_VALIDATION_GATES",
    "DEFAULT_VARIANCE_POLICIES",
    "NotRequired",
    "RMT_AGGRESSIVE",
    "RMT_BALANCED",
    "RMT_CONSERVATIVE",
    "SPECTRAL_AGGRESSIVE",
    "SPECTRAL_ATTN_AWARE",
    "SPECTRAL_BALANCED",
    "SPECTRAL_CONSERVATIVE",
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
]
