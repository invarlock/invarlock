from __future__ import annotations

from typing import Any

from .exceptions import InvarlockError

_ERROR_ENCODING_EXCEPTIONS = (AttributeError, RuntimeError, TypeError, ValueError)


def _safe_error_category(exc: Exception) -> str:
    try:
        return type(exc).__name__
    except _ERROR_ENCODING_EXCEPTIONS:
        return "Exception"


def _safe_is_invarlock_error(exc: Exception) -> bool:
    try:
        return isinstance(exc, InvarlockError)
    except _ERROR_ENCODING_EXCEPTIONS:
        return False


def encode_error(exc: Exception) -> dict[str, Any]:
    """Encode an exception as a structured machine-readable error payload."""
    category = _safe_error_category(exc)

    payload: dict[str, Any] = {
        "code": "E_GENERIC",
        "category": category,
        "recoverable": False,
        "context": {},
    }

    if _safe_is_invarlock_error(exc):
        payload["code"] = getattr(exc, "code", payload["code"]) or payload["code"]
        payload["recoverable"] = bool(getattr(exc, "recoverable", False))
        details = getattr(exc, "details", None)
        if isinstance(details, dict):
            payload["context"] = details
        return payload

    if category in {"ValidationError", "ConfigError", "DataError"}:
        payload["code"] = "E_SCHEMA"

    return payload


__all__ = ["encode_error"]
