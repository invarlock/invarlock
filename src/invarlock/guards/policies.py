"""Facade for guard policy presets and helpers."""

from __future__ import annotations

import builtins as _builtins

from . import policies_impl as _impl
from . import tier_config as _tier_config

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

_is_non_bool_number = _impl._is_non_bool_number
VariancePolicyDict = _impl.VariancePolicyDict
VariancePolicyRequired = _impl.VariancePolicyRequired
SPECTRAL_CONSERVATIVE = _impl.SPECTRAL_CONSERVATIVE
SPECTRAL_BALANCED = _impl.SPECTRAL_BALANCED
SPECTRAL_AGGRESSIVE = _impl.SPECTRAL_AGGRESSIVE
SPECTRAL_ATTN_AWARE = _impl.SPECTRAL_ATTN_AWARE
RMT_CONSERVATIVE = _impl.RMT_CONSERVATIVE
RMT_BALANCED = _impl.RMT_BALANCED
RMT_AGGRESSIVE = _impl.RMT_AGGRESSIVE
VARIANCE_CONSERVATIVE = _impl.VARIANCE_CONSERVATIVE
VARIANCE_BALANCED = _impl.VARIANCE_BALANCED
VARIANCE_AGGRESSIVE = _impl.VARIANCE_AGGRESSIVE
DEFAULT_SPECTRAL_POLICIES = _impl.DEFAULT_SPECTRAL_POLICIES
DEFAULT_RMT_POLICIES = _impl.DEFAULT_RMT_POLICIES
DEFAULT_VARIANCE_POLICIES = _impl.DEFAULT_VARIANCE_POLICIES
VALIDATION_GATE_STRICT = _impl.VALIDATION_GATE_STRICT
VALIDATION_GATE_STANDARD = _impl.VALIDATION_GATE_STANDARD
VALIDATION_GATE_PERMISSIVE = _impl.VALIDATION_GATE_PERMISSIVE
DEFAULT_VALIDATION_GATES = _impl.DEFAULT_VALIDATION_GATES
get_spectral_policy = _impl.get_spectral_policy
create_custom_spectral_policy = _impl.create_custom_spectral_policy
get_policy_for_model_size = _impl.get_policy_for_model_size
get_rmt_policy = _impl.get_rmt_policy
create_custom_rmt_policy = _impl.create_custom_rmt_policy
get_rmt_policy_for_model_size = _impl.get_rmt_policy_for_model_size
get_variance_policy = _impl.get_variance_policy
create_custom_variance_policy = _impl.create_custom_variance_policy
get_variance_policy_for_model_size = _impl.get_variance_policy_for_model_size
get_validation_gate = _impl.get_validation_gate
enforce_validation_gate = _impl.enforce_validation_gate
check_policy_drift = _impl.check_policy_drift

__all__ = list(_impl.__all__)
