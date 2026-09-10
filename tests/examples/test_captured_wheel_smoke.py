"""The installed-wheel rehearsal must notice broken commands and artifacts."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/captured-results/wheel_smoke.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("captured_wheel_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _real_cli_transport(monkeypatch, module, *, fault=None):
    """Replace only process launching; execute each command against the real CLI."""
    runner = CliRunner()
    calls, roots = [], []
    executable = "/candidate/bin/invarlock"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--cli", executable])
    monkeypatch.setattr(module.shutil, "which", lambda name: name)
    monkeypatch.setenv("PYTHONPATH", "/untrusted/checkout")
    monkeypatch.setenv("INVARLOCK_SIGNING_KEY", "/untrusted/private.pem")

    def run(command, *, cwd, env, capture_output, text, check):
        assert command[0] == executable
        assert capture_output is True and text is True and check is False
        assert "PYTHONPATH" not in env and "INVARLOCK_SIGNING_KEY" not in env
        assert env["PYTHONSAFEPATH"] == "1"
        assert not cwd.is_relative_to(ROOT)
        roots.append(cwd)
        with monkeypatch.context() as process:
            process.chdir(cwd)
            process.delenv("PYTHONPATH", raising=False)
            process.delenv("INVARLOCK_SIGNING_KEY", raising=False)
            result = runner.invoke(app, command[1:], env=env)
        completed = subprocess.CompletedProcess(
            command, result.exit_code, result.stdout, result.stderr
        )
        calls.append(completed)
        if fault is not None:
            fault(completed, cwd)
        return completed

    monkeypatch.setattr(module.subprocess, "run", run)
    return calls, roots


def test_wheel_smoke_rehearses_real_signed_handoffs_and_all_gate_exits(
    monkeypatch, capsys
):
    module = _module()
    calls, roots = _real_cli_transport(monkeypatch, module)
    module.main()
    output = capsys.readouterr().out
    for example in ("classification", "extraction", "judge"):
        assert f"{example}: installed comparison" in output
    assert (
        "regression, integration error and insufficient-evidence exit codes pass"
        in output
    )
    verifications = [call for call in calls if call.args[1] == "verify"]
    assert [call.returncode for call in verifications] == [0, 0, 0, 6, 7, 7]
    assert all(json.loads(call.stdout)["integrity_ok"] for call in verifications[:3])
    evaluations = [
        call
        for call in calls
        if call.args[1] == "evaluate" and call.args[2].endswith("request.yaml")
    ]
    assert [call.returncode for call in evaluations] == [
        0,
        2,
        0,
        0,
        2,
        0,
        0,
        2,
        0,
        0,
        7,
        7,
    ]
    assert json.loads(evaluations[-2].stdout)["decision"] == "regression"
    assert json.loads(evaluations[-1].stdout)["decision"] == "insufficient_evidence"
    assert roots and all(root == roots[0] for root in roots)
    assert not roots[0].exists()
    assert os.environ["PYTHONPATH"] == "/untrusted/checkout"
    assert os.environ["INVARLOCK_SIGNING_KEY"] == "/untrusted/private.pem"


def test_wheel_smoke_requires_an_installed_cli(monkeypatch):
    module = _module()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="Install the candidate wheel"):
        module.main()


def test_wheel_smoke_reports_failed_command_diagnostics_and_cleans_up(monkeypatch):
    module = _module()

    def fail_keygen(completed, root):
        assert "--keygen" in completed.args
        completed.returncode = 2
        completed.stdout = "key generation rejected; "
        completed.stderr = "key directory is unavailable"

    calls, roots = _real_cli_transport(monkeypatch, module, fault=fail_keygen)
    with pytest.raises(
        RuntimeError,
        match="evaluate returned 2, expected 0: key generation rejected; key directory is unavailable",
    ):
        module.main()
    assert len(calls) == 1
    assert not roots[0].exists()


@pytest.mark.parametrize(
    ("fault_name", "error"),
    [
        ("malformed_status", json.JSONDecodeError),
        ("failed_decision", AssertionError),
        ("missing_report", FileNotFoundError),
        ("empty_report", AssertionError),
        ("unauthenticated", AssertionError),
        ("missed_regression", RuntimeError),
        ("render_claims_verified", AssertionError),
        ("render_changed_evidence", AssertionError),
        ("human_claims_verified", AssertionError),
    ],
)
def test_wheel_smoke_rejects_incomplete_or_contradictory_results(
    monkeypatch, fault_name, error
):
    module = _module()

    def corrupt(completed, root):
        command = completed.args
        if (
            command[1:3] == ["evaluate", "classification/request.yaml"]
            and completed.returncode == 0
        ):
            if fault_name == "human_claims_verified" and "--json" not in command:
                completed.stdout = completed.stdout.replace(
                    "Independent verification: not performed",
                    "Independent verification: passed",
                )
            elif "--json" in command and fault_name == "malformed_status":
                completed.stdout = "not JSON"
            elif "--json" in command and fault_name == "failed_decision":
                status = json.loads(completed.stdout)
                status["decision"] = "regression"
                completed.stdout = json.dumps(status)
        elif command[1] == "verify" and fault_name == "unauthenticated":
            status = json.loads(completed.stdout)
            status["integrity_ok"] = False
            completed.stdout = json.dumps(status)
        elif command[1] == "report" and completed.returncode == 0:
            if fault_name == "missing_report":
                (root / "classification.html").unlink()
            elif fault_name == "empty_report":
                (root / "classification.html").write_bytes(b"")
            elif fault_name == "render_claims_verified":
                status = json.loads(completed.stdout)
                status["independent_verification"] = "passed"
                completed.stdout = json.dumps(status)
            elif fault_name == "render_changed_evidence":
                manifest = root / "classification/artifacts/evidence/manifest.json"
                manifest.chmod(0o644)
                manifest.write_text("{}")
        elif (
            command[1:3] == ["evaluate", "regressed/request.yaml"]
            and fault_name == "missed_regression"
        ):
            completed.returncode = 0

    _, roots = _real_cli_transport(monkeypatch, module, fault=corrupt)
    with pytest.raises(error):
        module.main()
    assert roots and not roots[0].exists()


def test_wheel_smoke_uses_exact_core_root_without_a_namespace_switch(
    monkeypatch, capsys
):
    module = _module()
    calls, roots = _real_cli_transport(monkeypatch, module)
    module.main()
    assert calls and {call.args[1] for call in calls} == {
        "evaluate",
        "verify",
        "report",
    }
    assert "text summaries and rendering" in capsys.readouterr().out
    assert not roots[0].exists()
