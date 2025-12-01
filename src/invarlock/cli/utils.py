from __future__ import annotations

from typing import Any


def coerce_option(value: Any, fallback: Any | None = None) -> Any:
    """Return a Typer option's concrete value.

    - If `value` looks like a Typer OptionInfo, return its `.default`.
    - Otherwise, return `value` unless it is None, in which case `fallback`.
    This keeps CLI command functions callable both by Typer and directly in tests.
    """
    try:
        # Typer OptionInfo typically exposes a `.default` attribute
        default = getattr(value, "default", None)
        # If it's an OptionInfo-like object, prefer its default
        if default is not None or value.__class__.__name__ == "OptionInfo":
            return default if default is not None else fallback
    except Exception:
        pass
    return value if value is not None else fallback


def coerce_float(value: Any, default: float) -> float:
    """Coerce arbitrary input to float, falling back to `default` on error."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: Any, default: int) -> int:
    """Coerce arbitrary input to int, falling back to `default` on error."""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
