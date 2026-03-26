from __future__ import annotations

from .exceptions import ConfigError, DataError, InvarlockError, ValidationError


def resolve_command_exit_code(exc: Exception, *, profile: str | None) -> int:
    """Resolve command exit codes deterministically from error class and profile."""

    try:
        prof = (profile or "").strip().lower()
    except Exception:
        prof = ""

    if isinstance(exc, (ConfigError, ValidationError, DataError)):
        return 2
    if isinstance(exc, ValueError) and "Invalid RunReport" in str(exc):
        return 2
    if isinstance(exc, InvarlockError) and prof in {"ci", "release"}:
        return 3
    return 1


__all__ = ["resolve_command_exit_code"]
