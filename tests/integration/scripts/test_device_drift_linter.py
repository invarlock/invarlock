"""Unit tests for the device drift checker utility."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _script_path() -> Path:
    return Path("scripts/smoke/check_device_drift.py")


def test_device_drift_checker_pass(tmp_path: Path) -> None:
    script = _script_path()
    cpu = Path("tests/fixtures/device_drift/report_cpu.json")
    mps = Path("tests/fixtures/device_drift/report_mps.json")
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
    cpu = Path("tests/fixtures/device_drift/report_cpu.json")
    bad = Path("tests/fixtures/device_drift/report_bad.json")
    result = subprocess.run(
        [sys.executable, str(script), str(cpu), str(bad), "--tolerance", "0.005"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Device drift exceeded tolerance" in result.stderr


def test_device_drift_checker_rejects_non_finite_ratio(tmp_path: Path) -> None:
    script = _script_path()
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(
        json.dumps({"primary_metric": {"ratio_vs_baseline": 1.0}}),
        encoding="utf-8",
    )
    candidate.write_text(
        '{"primary_metric": {"ratio_vs_baseline": NaN}}',
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script), str(reference), str(candidate)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "report ratio must be finite" in result.stderr


def test_device_drift_checker_rejects_invalid_tolerance(tmp_path: Path) -> None:
    script = _script_path()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"primary_metric": {"ratio_vs_baseline": 1.0}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script), str(report), str(report), "--tolerance", "-1"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--tolerance must be a finite non-negative number" in result.stderr


pytestmark = pytest.mark.integration
