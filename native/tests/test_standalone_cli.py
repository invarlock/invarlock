"""Public diagnostics, environment precedence, and native terminal content."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from test_standalone_workflows import full_fixtures
from typer.testing import CliRunner

from invarlock.cli.app import app

BIN = Path(os.environ["CARGO_TARGET_DIR"]) / "debug/invarlock-native-verify"


def run(args, env=None):
    return subprocess.run(
        [str(BIN), *map(str, args)],
        capture_output=True,
        text=True,
        env={"PATH": "/nonexistent", **(env or {})},
    )


@pytest.mark.parametrize("command", ["evaluate", "verify", "report"])
def test_help_is_execution_free(command):
    result = run([command, "--help"])
    assert result.returncode == 0
    assert "Usage:" in result.stdout and "--json" in result.stdout
    assert not result.stderr


@pytest.mark.parametrize(
    "mode", ["environment", "override", "terminal", "bad_backend", "missing_key"]
)
def test_captured_cli_controls(tmp_path, mode):
    root = tmp_path.resolve() / "fixture"
    full_fixtures.captured(root, 4, "pass")
    args = [
        "evaluate",
        str(root / "request.json"),
        "--output",
        str(root / "new"),
        "--preflight",
    ]
    env = {"INVARLOCK_SIGNING_KEY": str(root / "signer.pem")}
    if mode == "override":
        env["INVARLOCK_SIGNING_KEY"] = str(root / "missing")
        args += ["--signing-key", str(root / "signer.pem")]
    if mode == "bad_backend":
        args += ["--backend", "python"]
    if mode == "missing_key":
        env.clear()
    if mode != "terminal":
        args += ["--json"]
    result = run(args, env)
    if mode == "bad_backend":
        assert result.returncode == 2
        return
    reference = CliRunner().invoke(app, args, env=env)
    assert result.returncode == reference.exit_code
    assert not result.stderr
    if mode == "terminal":
        assert result.stdout == reference.stdout
    elif mode == "missing_key":
        payload = json.loads(result.stdout)
        assert (
            payload["format_version"] == json.loads(reference.stdout)["format_version"]
        )
        assert payload["ok"] is False and result.returncode == 2
    else:
        assert json.loads(result.stdout) == json.loads(reference.stdout)


def test_verification_error_uses_public_contract(tmp_path):
    root = tmp_path.resolve() / "fixture"
    full_fixtures.captured(root, 4, "pass")
    args = ["verify", str(root / "pack"), "--json"]
    expected = CliRunner().invoke(app, args)
    actual = run(args)
    assert actual.returncode == expected.exit_code == 2
    left, right = json.loads(actual.stdout), json.loads(expected.stdout)
    assert left.pop("errors") and right.pop("errors")
    assert left == right
    assert not actual.stderr


def test_failed_terminal_report_has_visible_diagnostic(tmp_path):
    result = run(["report", tmp_path / "missing"])
    assert result.returncode == 2
    assert "FAIL evidence must be a real directory" in result.stdout
