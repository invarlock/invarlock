"""Unit tests for the device drift checker utility."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _script_path() -> Path:
    return Path("scripts/check_device_drift.py")


def test_device_drift_checker_pass(tmp_path: Path) -> None:
    script = _script_path()
    cpu = Path("tests/fixtures/device_drift/cert_cpu.json")
    mps = Path("tests/fixtures/device_drift/cert_mps.json")
    result = subprocess.run(
        [sys.executable, str(script), str(cpu), str(mps), "--tolerance", "0.005"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Device drift OK" in result.stdout


def test_device_drift_checker_fail(tmp_path: Path) -> None:
    script = _script_path()
    cpu = Path("tests/fixtures/device_drift/cert_cpu.json")
    bad = Path("tests/fixtures/device_drift/cert_bad.json")
    result = subprocess.run(
        [sys.executable, str(script), str(cpu), str(bad), "--tolerance", "0.005"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Device drift exceeded tolerance" in result.stderr


pytestmark = pytest.mark.integration
