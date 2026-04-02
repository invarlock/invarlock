from __future__ import annotations

from typing import Any


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
