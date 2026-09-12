from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_calibration_rejects_intervals_that_miss_the_mean_and_formula(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "EXPERIMENTS", 1)
    monkeypatch.setattr(
        module,
        "_scenarios",
        lambda: (module.Scenario("fixed", 1, Decimal(1), lambda rng: Decimal(0)),),
    )
    monkeypatch.setattr(
        module,
        "hoeffding_interval",
        lambda *args, **kwargs: SimpleNamespace(lower=Decimal(0), upper=Decimal(0)),
    )

    result = module.calibrate()

    assert result["passed"] is False
    scenario = result["scenarios"][0]
    assert scenario["misses"] == 1
    assert scenario["formula_mismatches"] == 1
    assert scenario["passed"] is False


def test_cli_prints_calibration_json(monkeypatch, capsys):
    module = _module()
    payload = {"passed": True, "scenarios": []}
    monkeypatch.setattr(module, "calibrate", lambda: payload)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    assert module.main() == 0
    assert json.loads(capsys.readouterr().out) == payload


def test_cli_checks_and_writes_exact_calibration_json(monkeypatch, tmp_path, capsys):
    module = _module()
    payload = {"passed": True, "scenarios": []}
    expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    checked = tmp_path / "checked.json"
    output = tmp_path / "result.json"
    checked.write_text(expected, encoding="utf-8")
    monkeypatch.setattr(module, "calibrate", lambda: payload)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--check", str(checked), "--output", str(output)]
    )

    assert module.main() == 0
    assert output.read_text(encoding="utf-8") == expected
    assert capsys.readouterr().out == ""


def test_cli_rejects_changed_calibration_without_writing_output(monkeypatch, tmp_path):
    module = _module()
    checked = tmp_path / "checked.json"
    output = tmp_path / "result.json"
    checked.write_text('{"passed": false}\n', encoding="utf-8")
    monkeypatch.setattr(module, "calibrate", lambda: {"passed": True})
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--check", str(checked), "--output", str(output)]
    )

    with pytest.raises(SystemExit, match="calibration result differs"):
        module.main()
    assert not output.exists()


def test_script_entry_point_checks_frozen_calibration(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--check", str(FIXTURE)])

    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert caught.value.code == 0
    assert json.loads(capsys.readouterr().out) == json.loads(
        FIXTURE.read_text(encoding="utf-8")
    )
