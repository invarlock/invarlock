"""Actual subprocess boundaries for the bounded campaign supervisor."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def supervisor(monkeypatch):
    directory = (
        Path(__file__).resolve().parents[2] / "examples/integrations/evaluator-live"
    )
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "live_supervisor_test", directory / "supervise.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supervisor_retains_success_and_failure(supervisor, tmp_path):
    for status in (0, 3):
        target = tmp_path / str(status)
        result = supervisor.run(
            [sys.executable, "-c", f"print('retained'); raise SystemExit({status})"],
            10,
            target,
        )
        assert result["exit_code"] == status and not result["timed_out"]
        assert (target / "output.log").read_text() == "retained\n"
        assert json.loads((target / "result.json").read_text()) == result


def test_supervisor_kills_stalled_command(supervisor, tmp_path):
    result = supervisor.run(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)",
        ],
        1,
        tmp_path / "deadline",
    )
    assert result["exit_code"] == 124 and result["timed_out"]
    assert result["elapsed_seconds"] < 10


@pytest.mark.parametrize("seconds", [0, -1, True, 86401])
def test_supervisor_refuses_invalid_cap(supervisor, tmp_path, seconds):
    with pytest.raises(ValueError):
        supervisor.run([sys.executable], seconds, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
