"""Explicit native process control for provider-owned SDK projections."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

_BACKEND: ContextVar[str | None] = ContextVar(
    "invarlock_execution_backend", default=None
)


@contextmanager
def using_execution_backend(backend: str) -> Iterator[None]:
    if backend not in {"python", "rust"}:
        raise ValueError("execution backend must be python or rust")
    token = _BACKEND.set(backend)
    try:
        yield
    finally:
        _BACKEND.reset(token)


def scope_execution_backend[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Scope a public transaction's keyword-only backend to host provider hooks."""

    @wraps(function)
    def scoped(*args: P.args, **kwargs: P.kwargs) -> R:
        backend = kwargs.get("backend", "python")
        if not isinstance(backend, str) or backend not in {"python", "rust"}:
            # Preserve the public transaction's own validation and error class.
            return function(*args, **kwargs)
        with using_execution_backend(backend):
            return function(*args, **kwargs)

    return scoped


def execution_backend() -> str:
    """Select only the backend explicitly propagated into the worker process."""
    backend = _BACKEND.get() or os.environ.get("INVARLOCK_EXECUTION_BACKEND", "python")
    if backend not in {"python", "rust"}:
        raise ValueError("INVARLOCK_EXECUTION_BACKEND must be python or rust")
    return backend


def run_native_process(
    *,
    program: str,
    arguments: Sequence[str],
    input_bytes: bytes,
    pass_fds: Sequence[int],
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    stdout_limit: int,
    stderr_limit: int,
    error_type: type[Exception],
    timeout_label: str,
    output_label: str,
) -> tuple[int, bytes, bytes]:
    """Transfer owned file capabilities to native drain/deadline supervision."""
    try:
        invarlock_native = importlib.import_module("invarlock_native")
    except ImportError as exc:
        raise error_type(
            "the explicit Rust execution backend requires invarlock_native"
        ) from exc
    cancellation = invarlock_native.CancellationToken()
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="invarlock-provider-control"
    ) as pool:
        future = pool.submit(
            invarlock_native.run_process,
            program,
            list(arguments),
            input_bytes,
            timeout_seconds,
            stdout_limit,
            stderr_limit,
            cancellation=cancellation,
            cwd=str(cwd),
            environment=dict(environment),
            pass_fds=list(pass_fds),
        )
        try:
            while True:
                try:
                    result = future.result(timeout=0.1)
                    if (
                        not isinstance(result, tuple)
                        or len(result) != 3
                        or type(result[0]) is not int
                        or type(result[1]) is not bytes
                        or type(result[2]) is not bytes
                    ):
                        raise ValueError("native process response is invalid")
                    return result[0], result[1], result[2]
                except TimeoutError:
                    continue
        except ValueError as exc:
            message = str(exc)
            if message == "process timed out":
                message = f"{timeout_label} timed out"
            elif message in {
                "process stdout limit exceeded",
                "process stderr limit exceeded",
            }:
                message = f"{output_label} {message.removeprefix('process ')}"
            raise error_type(message) from exc
        except BaseException:
            cancellation.cancel()
            raise
