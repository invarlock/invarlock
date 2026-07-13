"""Strict JSON serialization for public and authority-bearing artifacts."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


class FiniteJsonError(ValueError):
    """Raised when a JSON payload contains a non-finite numeric value."""


def _path_key(path: str, key: object) -> str:
    if isinstance(key, str) and key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def require_finite_json(value: Any, *, path: str = "$") -> None:
    """Reject NaN and infinities recursively, reporting their payload path.

    This function deliberately does not turn invalid measurements into ``null``.
    Producers must explicitly omit undefined optional fields or represent an
    unavailable optional value as ``None`` before reaching this boundary.
    """

    if isinstance(value, float):
        if not math.isfinite(value):
            raise FiniteJsonError(f"non-finite JSON number at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            require_finite_json(item, path=_path_key(path, key))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            require_finite_json(item, path=f"{path}[{index}]")


def dumps_finite_json(value: Any, **kwargs: Any) -> str:
    """Serialize standards-compliant JSON, rejecting non-finite numbers."""

    require_finite_json(value)
    kwargs["allow_nan"] = False
    return json.dumps(value, **kwargs)


def normalize_optional_nonfinite_json(value: Any) -> Any:
    """Return a deep JSON-like copy with unavailable optional numbers as null.

    This is intended for non-authoritative rendering and explicitly optional
    diagnostic values. Authority writers must still use :func:`dumps_finite_json`.
    """

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {
            key: normalize_optional_nonfinite_json(item) for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized = [normalize_optional_nonfinite_json(item) for item in value]
        return tuple(normalized) if isinstance(value, tuple) else normalized
    return copy.deepcopy(value)


__all__ = [
    "FiniteJsonError",
    "dumps_finite_json",
    "normalize_optional_nonfinite_json",
    "require_finite_json",
]
