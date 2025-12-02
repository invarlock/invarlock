"""Plugin template namespace (`invarlock.plugins`)."""

from __future__ import annotations

from invarlock.core.abi import INVARLOCK_CORE_ABI as INVARLOCK_CORE_ABI

from .hello_guard import HelloGuard

__all__ = [
    "HelloGuard",
    "INVARLOCK_CORE_ABI",
]
