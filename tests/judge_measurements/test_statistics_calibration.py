from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts/checks/judge_statistics_calibration.py"
FIXTURE = REPO_ROOT / "tests/fixtures/judge_measurements/statistics_calibration.json"


def _module():
    spec = importlib.util.spec_from_file_location(
        "judge_statistics_calibration", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_statistical_calibration_matches_implementation():
    result = _module().calibrate()
    assert result["passed"] is True
    assert result == json.loads(FIXTURE.read_text(encoding="utf-8"))
