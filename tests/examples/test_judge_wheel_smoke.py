"""Exercise the core-wheel judge rehearsal against the real command handlers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "examples/judge-measurements"
SCRIPT = FIXTURE / "wheel_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("judge_wheel_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transport(monkeypatch, module, fault=None):
    runner = CliRunner()
    calls, roots = [], []
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--fixture", str(FIXTURE)])
    monkeypatch.setattr(module.shutil, "which", lambda _: "/candidate/bin/invarlock")
    monkeypatch.setattr(module, "require_core_only", lambda: None)

    def run(command, *, cwd, env, capture_output, text, check):
        assert not cwd.is_relative_to(ROOT)
        assert (
            not {
                "PYTHONPATH",
                "INVARLOCK_SIGNING_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
                "OPENROUTER_API_KEY",
            }
            & env.keys()
        )
        assert capture_output and text and not check
        with monkeypatch.context() as context:
            context.chdir(cwd)
            result = runner.invoke(app, command[1:], env=env)
        completed = subprocess.CompletedProcess(
            command, result.exit_code, result.stdout, result.stderr
        )
        calls.append(completed)
        roots.append(cwd)
        if fault:
            fault(completed, cwd)
        return completed

    monkeypatch.setattr(module.subprocess, "run", run)
    return calls, roots


def test_core_judge_consumer_preserves_inconclusive_result_and_rejects_bad_pins(
    monkeypatch, capsys
):
    module = load_module()
    calls, roots = transport(monkeypatch, module)
    module.main()
    verifications = [c for c in calls if c.args[1] == "verify"]
    assert [c.returncode for c in verifications] == [7, 4, 4, 4, 4]
    assert json.loads(verifications[0].stdout)["verified"]
    assert "core-only signed replay" in capsys.readouterr().out
    assert not roots[0].exists()


@pytest.mark.parametrize("module_name", ["inspect_ai", "openai"])
def test_rehearsal_refuses_optional_sdk_install_or_import(monkeypatch, module_name):
    module = load_module()
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda name: object() if name == module_name else None,
    )
    with pytest.raises(RuntimeError, match="optional module"):
        module.require_core_only()


def test_rehearsal_requires_an_installed_candidate_cli(monkeypatch):
    module = load_module()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--fixture", str(FIXTURE)])
    monkeypatch.setattr(module, "require_core_only", lambda: None)
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        module.tempfile,
        "TemporaryDirectory",
        lambda **_kwargs: pytest.fail("missing CLI must fail before staging evidence"),
    )

    with pytest.raises(SystemExit, match="Install the candidate wheel"):
        module.main()


@pytest.mark.parametrize(
    "fault", ["accepts_inconclusive", "accepts_wrong_pin", "empty_report"]
)
def test_rehearsal_detects_broken_recipient_or_report(monkeypatch, fault):
    module = load_module()

    def corrupt(result, root):
        if fault == "empty_report" and result.args[1] == "report":
            (root / "report.html").write_bytes(b"")
        elif result.args[1] == "verify":
            if (fault == "accepts_inconclusive" and result.returncode == 7) or (
                fault == "accepts_wrong_pin" and result.returncode == 4
            ):
                result.returncode = 0

    _, roots = transport(monkeypatch, module, corrupt)
    with pytest.raises((RuntimeError, AssertionError)):
        module.main()
    assert not roots[0].exists()


def test_release_resolves_the_optional_judge_sdk_from_the_core_wheel():
    make = (ROOT / "Makefile").read_text()
    assert "inspect-judge-sdk-test: dist-check" in make
    assert (
        'core_wheels=("${ROOT_DIR}"/dist/invarlock-*.whl)'
        in (ROOT / "scripts/inspect_judge_sdk_gate.sh").read_text()
    )
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    assert "bash scripts/inspect_judge_sdk_gate.sh" in workflow
