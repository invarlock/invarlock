from __future__ import annotations

import os


def guard_assert(cond: bool, msg: str) -> None:
    """Enable lightweight runtime contracts when INVARLOCK_ASSERT_GUARDS=1."""
    if os.getenv("INVARLOCK_ASSERT_GUARDS", "0") == "1":
        assert bool(cond), msg
