from __future__ import annotations

from collections.abc import Callable
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from .exceptions import InvarlockError

T = TypeVar("T", bound=InvarlockError)

_ERROR_ENCODING_EXCEPTIONS = (AttributeError, RuntimeError, TypeError, ValueError)

ContextFn = Callable[[BaseException], dict[str, Any] | None]


@dataclass
class _WrapErrors(ContextDecorator, Generic[T]):  # noqa: UP046
    target_exc: type[T]
    code: str
    message: str
    context_fn: ContextFn | None = None

    # Context manager protocol
    def __enter__(self) -> _WrapErrors:  # pragma: no cover - trivial
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> Literal[False]:
        if exc is None:
            return False
        # If it's already a InvarlockError, do not double-wrap
        if isinstance(exc, InvarlockError):
            return False
        ctx = self.context_fn(exc) if self.context_fn is not None else None
        wrapped = self.target_exc(code=self.code, message=self.message, details=ctx)
        raise wrapped from exc


def wrap_errors(  # noqa: UP047
    target_exc: type[T],
    code: str,
    message: str,
    context_fn: ContextFn | None = None,
) -> _WrapErrors[T]:
    """Return a context manager/decorator that wraps arbitrary exceptions.

    Usage as context manager:
        with wrap_errors(AdapterError, "E202", "ADAPTER-LOAD-FAILED", ctx):
            risky()

    Usage as decorator:
        @wrap_errors(ValidationError, "E301", "VALIDATION-FAILED")
        def f(...): ...
    """
    return _WrapErrors(
        target_exc=target_exc, code=code, message=message, context_fn=context_fn
    )


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


__all__ = ["encode_error", "wrap_errors"]
