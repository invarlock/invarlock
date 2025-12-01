from __future__ import annotations

from collections.abc import Callable
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from .exceptions import InvarlockError

T = TypeVar("T", bound=InvarlockError)


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


__all__ = ["wrap_errors"]
