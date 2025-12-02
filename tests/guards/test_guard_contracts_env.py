from __future__ import annotations

import pytest

from invarlock.guards._contracts import guard_assert


def test_guard_assert_disabled_noop(monkeypatch):
    monkeypatch.delenv("INVARLOCK_ASSERT_GUARDS", raising=False)
    guard_assert(False, "msg")  # should not raise


def test_guard_assert_enabled_raises(monkeypatch):
    monkeypatch.setenv("INVARLOCK_ASSERT_GUARDS", "1")
    with pytest.raises(AssertionError):
        guard_assert(False, "boom")
