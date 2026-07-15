"""Retry-controller construction for config-driven execution."""

from __future__ import annotations

from typing import Any


def init_retry_controller(
    *,
    until_pass: bool,
    max_attempts: int,
    timeout: int | None,
    baseline: str | None,
) -> Any:
    """Create the retry controller only when repeated execution is requested."""

    del baseline
    if not until_pass:
        return None
    from invarlock.core.retry import RetryController

    return RetryController(max_attempts=max_attempts, timeout=timeout, verbose=True)


__all__ = ["init_retry_controller"]
