"""Versioned acceptance authority for assurance guard findings.

Authority changes whether a complete, replayable guard finding blocks acceptance.
It never changes the canonical guard chain or relaxes evidence completeness.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

GUARD_AUTHORITY_GUARDS = ("spectral", "rmt", "variance")
GUARD_AUTHORITY_VALUES = frozenset({"enforce", "observe"})
DEFAULT_GUARD_AUTHORITY = {
    "spectral": "enforce",
    "rmt": "enforce",
    "variance": "enforce",
}


def guard_authority_errors(value: Any, *, path: str) -> list[str]:
    """Return exact-shape errors for one guard-authority mapping."""

    if not isinstance(value, Mapping):
        return [f"{path} must be an object."]
    if set(value) != set(GUARD_AUTHORITY_GUARDS):
        return [f"{path} must contain exactly spectral, rmt, and variance."]
    errors: list[str] = []
    for guard in GUARD_AUTHORITY_GUARDS:
        authority = value.get(guard)
        if authority not in GUARD_AUTHORITY_VALUES:
            errors.append(f"{path}.{guard} must be observe or enforce.")
    return errors


def resolved_guard_authority(
    report: Mapping[str, Any],
) -> tuple[dict[str, str], list[str], bool]:
    """Resolve report authority, defaulting legacy reports to all-enforce.

    The final boolean is true only when the report explicitly carries the v2
    mapping. Invalid explicit mappings remain v2-shaped and fail closed.
    """

    resolved = report.get("resolved_policy")
    if not isinstance(resolved, Mapping) or "guard_authority" not in resolved:
        return dict(DEFAULT_GUARD_AUTHORITY), [], False
    raw = resolved.get("guard_authority")
    errors = guard_authority_errors(raw, path="resolved_policy.guard_authority")
    if errors or not isinstance(raw, Mapping):
        return dict(DEFAULT_GUARD_AUTHORITY), errors, True
    return (
        {guard: str(raw[guard]) for guard in GUARD_AUTHORITY_GUARDS},
        [],
        True,
    )


def guard_is_enforced(authority: Mapping[str, str], guard: str) -> bool:
    """Return whether a guard finding has acceptance authority."""

    return guard not in GUARD_AUTHORITY_GUARDS or authority.get(guard) != "observe"


__all__ = [
    "DEFAULT_GUARD_AUTHORITY",
    "GUARD_AUTHORITY_GUARDS",
    "GUARD_AUTHORITY_VALUES",
    "guard_authority_errors",
    "guard_is_enforced",
    "resolved_guard_authority",
]
