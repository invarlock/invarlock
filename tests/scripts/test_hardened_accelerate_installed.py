"""Real installed-wheel checks; opt in with an isolated candidate interpreter."""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path

import pytest

from tests.scripts.hardened_accelerate_checks import InstalledAccelerateChecks

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests/scripts/hardened_accelerate_checks.py"
CHECKS = unittest.defaultTestLoader.getTestCaseNames(InstalledAccelerateChecks)
# Prevent pytest from separately collecting this imported unittest class.
del InstalledAccelerateChecks


@pytest.mark.parametrize("check", CHECKS)
def test_installed_hardened_accelerate(check, tmp_path):
    _run_installed_check(check, tmp_path, "INVARLOCK_HARDENED_ACCELERATE_PYTHON")


def test_installed_hardened_accelerate_harness(tmp_path):
    _run_installed_check(None, tmp_path, "INVARLOCK_HARDENED_HARNESS_PYTHON")


def _run_installed_check(check, tmp_path, variable):
    interpreter = os.environ.get(variable)
    if not interpreter:
        pytest.skip(f"set {variable} to the wheel-only environment")
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HOME=str(tmp_path / "hf"),
    )
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            interpreter,
            "-I",
            str(SCRIPT),
            *(["--check", check] if check else ["--harness"]),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report["accelerate"] == "1.14.0+invarlock.1"
    assert report["tests_run"] == 1
    assert report["failures"] == report["errors"] == report["skipped"] == 0
