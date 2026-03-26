from __future__ import annotations

from typing import Any

from .exceptions import InvarlockError


def encode_error(exc: Exception) -> dict[str, Any]:
    """Encode an exception as a structured machine-readable error payload."""

    try:
        category = type(exc).__name__
    except Exception:
        category = "Exception"

    payload: dict[str, Any] = {
        "code": "E_GENERIC",
        "category": category,
        "recoverable": False,
        "context": {},
    }

    try:
        if isinstance(exc, InvarlockError):
            payload["code"] = getattr(exc, "code", payload["code"]) or payload["code"]
            payload["recoverable"] = bool(getattr(exc, "recoverable", False))
            details = getattr(exc, "details", None)
            if isinstance(details, dict):
                payload["context"] = details
            return payload
    except Exception:
        pass

    if category in {"ValidationError", "ConfigError", "DataError"}:
        payload["code"] = "E_SCHEMA"

    return payload


__all__ = ["encode_error"]
