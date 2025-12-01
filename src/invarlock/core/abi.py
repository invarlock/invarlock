"""
Core ABI contract for InvarLock plugins.

Third-party plugins may declare a matching `INVARLOCK_CORE_ABI` attribute to signal
compatibility with this release line. Minor bumps may introduce breaking changes
in plugin contracts; match exactly for stability.
"""

from __future__ import annotations

# Plugin ABI for the core interfaces used by adapters/edits/guards.
# Increment when the plugin-facing contracts change.
INVARLOCK_CORE_ABI = "0.1"

__all__ = ["INVARLOCK_CORE_ABI"]
