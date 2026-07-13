"""Guard-specific validation flags for canonical evaluation reports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    return coerced if math.isfinite(coerced) else None


def resolve_spectral_stable(
    spectral: Any,
    *,
    tier_policy: Mapping[str, Any] | None,
) -> bool:
    """Validate an explicit spectral cap result against its resolved budget."""

    if not isinstance(spectral, Mapping):
        return False
    caps_applied = _finite_float(spectral.get("caps_applied"))
    if caps_applied is None or caps_applied < 0 or not caps_applied.is_integer():
        return False
    summary = spectral.get("summary", {})
    max_caps = spectral.get("max_caps")
    if max_caps is None and isinstance(summary, Mapping):
        max_caps = summary.get("max_caps")
    if max_caps is None:
        default_spectral = (
            tier_policy.get("spectral", {}) if isinstance(tier_policy, Mapping) else {}
        )
        max_caps = (
            default_spectral.get("max_caps", 5)
            if isinstance(default_spectral, Mapping)
            else 5
        )
    max_caps_value = _finite_float(max_caps)
    if max_caps_value is None or max_caps_value < 0 or not max_caps_value.is_integer():
        return False
    return caps_applied <= max_caps_value and not bool(spectral.get("caps_exceeded"))


def resolve_rmt_stable(rmt: Any) -> bool:
    """Require an explicit boolean RMT verdict; absence is not evidence."""

    return isinstance(rmt, Mapping) and rmt.get("stable") is True


def resolve_invariants_pass(invariants: Any) -> bool:
    """Require an explicit successful invariant status."""

    if not isinstance(invariants, Mapping):
        return False
    status = invariants.get("status")
    return isinstance(status, str) and status.strip().lower() in {"ok", "pass"}


__all__ = [
    "resolve_invariants_pass",
    "resolve_rmt_stable",
    "resolve_spectral_stable",
]
