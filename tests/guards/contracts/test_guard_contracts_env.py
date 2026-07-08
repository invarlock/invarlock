from __future__ import annotations

import os
import subprocess
import sys

import pytest

from invarlock.guards.policies import guard_assert
from tests._repo_root import REPO_ROOT


def test_guard_assert_disabled_noop(monkeypatch):
    monkeypatch.delenv("INVARLOCK_ASSERT_GUARDS", raising=False)
    assert guard_assert(False, "msg") is None


def test_guard_assert_enabled_raises(monkeypatch):
    monkeypatch.setenv("INVARLOCK_ASSERT_GUARDS", "1")
    with pytest.raises(AssertionError):
        guard_assert(False, "boom")


def test_guard_assert_enabled_survives_optimized_python():
    env = os.environ.copy()
    env["INVARLOCK_ASSERT_GUARDS"] = "1"
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (src_path, env.get("PYTHONPATH", "")) if part
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            (
                "from invarlock.guards.policies import guard_assert\n"
                "try:\n"
                "    guard_assert(False, 'boom')\n"
                "except AssertionError as exc:\n"
                "    raise SystemExit(0 if str(exc) == 'boom' else 2)\n"
                "raise SystemExit(1)\n"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
