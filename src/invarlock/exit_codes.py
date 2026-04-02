from __future__ import annotations

from invarlock.core.exceptions import (
    ConfigError,
    DataError,
    InvarlockError,
    ValidationError,
)


def resolve_command_exit_code(exc: Exception, *, profile: str | None) -> int:
    """Resolve shell exit codes deterministically from error class and profile."""

    prof = str(profile or "").strip().lower()

    if isinstance(exc, (ConfigError, ValidationError, DataError)):
        return 2
    if isinstance(exc, ValueError) and "invalid runreport" in str(exc).strip().lower():
        return 2
    if isinstance(exc, InvarlockError):
        return 3 if prof in {"ci", "ci_cpu", "release"} else 1
    return 1


__all__ = ["resolve_command_exit_code"]
