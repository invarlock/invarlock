from __future__ import annotations

# Back-compat shim: re-export the core InvarlockError for CLI imports
from invarlock.core.exceptions import InvarlockError

__all__ = ["InvarlockError"]
