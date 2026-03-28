from __future__ import annotations

import math
from typing import Any

_NUMERIC_EXCEPTIONS = (OverflowError, TypeError, ValueError)


def coerce_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except _NUMERIC_EXCEPTIONS:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def coerce_bool_like(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None
