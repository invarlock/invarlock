"""Process-local lookup of immutable installed-distribution versions."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
from functools import cache

_VERSION_ERRORS = (
    importlib_metadata.PackageNotFoundError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@cache
def _installed_distribution_version_for_process(
    process_id: int,
    distribution: str,
) -> str | None:
    """Resolve one distribution once for a specific process.

    ``process_id`` is intentionally part of the cache key. A forked child must
    perform its own metadata lookup instead of inheriting a result populated by
    its parent, while repeated calls inside one process remain inexpensive.
    """

    _ = process_id
    try:
        return importlib_metadata.version(distribution)
    except _VERSION_ERRORS:
        return None


def installed_distribution_version(distribution: str) -> str | None:
    """Return an installed package version, cached only within this process."""

    return _installed_distribution_version_for_process(os.getpid(), distribution)


def _clear_installed_distribution_version_cache() -> None:
    """Clear cached metadata results for isolated tests."""

    _installed_distribution_version_for_process.cache_clear()


__all__ = ["installed_distribution_version"]
