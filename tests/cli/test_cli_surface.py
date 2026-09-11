from __future__ import annotations

import json
from typing import cast

from click import Argument, Command, Group, Option
from typer.main import get_command
from typer.testing import CliRunner

from invarlock.cli.app import app

RUNNER = CliRunner()
ROOT_COMMAND = cast(Group, get_command(app))


def _command(name: str) -> Command:
    return ROOT_COMMAND.commands[name]


def _arguments(name: str) -> set[str]:
    return {
        param.name
        for param in _command(name).params
        if isinstance(param, Argument) and param.name is not None
    }


def _options(name: str) -> set[str]:
    return {
        option
        for param in _command(name).params
        if isinstance(param, Option)
        for option in param.opts
        if option.startswith("--")
    }


def test_core_help_renders() -> None:
    for args in ((), ("evaluate",), ("verify",), ("report",)):
        result = RUNNER.invoke(app, [*args, "--help"])
        assert result.exit_code == 0, result.output


def test_root_exposes_only_the_public_evaluation_journeys() -> None:
    assert set(ROOT_COMMAND.commands) == {"evaluate", "verify", "report"}


def test_evaluate_accepts_one_request_instead_of_model_flag_sprawl() -> None:
    options = _options("evaluate")

    assert _arguments("evaluate") == {"request"}
    assert "--baseline" not in options
    assert "--subject" not in options
    assert "--edit-config" not in options
    assert "--clean-selection" not in options
    assert "--allow-network" not in options
    assert "--allow-remote-code" not in options
    assert "--allow-installed-scorers" in options
    assert "--preflight" in options


def test_evaluate_setup_actions_are_request_free_and_machine_readable(
    tmp_path,
) -> None:
    options = _options("evaluate")
    assert {
        "--init",
        "--example",
        "--keygen",
        "--freeze-cases",
        "--case-set-output",
    } <= options

    initialized = RUNNER.invoke(
        app, ["evaluate", "--init", str(tmp_path / "example"), "--json"]
    )
    assert initialized.exit_code == 0, initialized.output
    init_result = json.loads(initialized.stdout)
    assert init_result["format_version"] == "invarlock/evaluation-setup-v1"
    assert init_result["action"] == "init"
    assert (tmp_path / "example/request.yaml").is_file()

    keygen = RUNNER.invoke(
        app, ["evaluate", "--keygen", str(tmp_path / "keys"), "--json"]
    )
    assert keygen.exit_code == 0, keygen.output
    assert (tmp_path / "keys/private.pem").is_file()
    assert (tmp_path / "keys/public.pem").is_file()


def test_evaluate_setup_rejects_request_and_conflicting_options(tmp_path) -> None:
    request = tmp_path / "request.yaml"
    request.write_text("not used", encoding="utf-8")
    result = RUNNER.invoke(
        app,
        ["evaluate", str(request), "--keygen", str(tmp_path / "keys"), "--json"],
    )
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload == {
        "action": None,
        "details": None,
        "errors": ["setup actions do not accept a positional request"],
        "format_version": "invarlock/evaluation-setup-v1",
        "ok": False,
    }
    assert not (tmp_path / "keys").exists()


def test_verify_uses_the_bundle_and_independent_trust_anchors() -> None:
    options = _options("verify")

    assert _arguments("verify") == {"evidence"}
    assert "--policy" in options
    assert "--expected-baseline-runtime" in options
    assert "--expected-subject-runtime" in options
    assert "--expected-baseline-artifact" in options
    assert "--expected-subject-artifact" in options
    assert "--expected-schedule" in options
    assert "--expected-signer" in options
    assert "--receipt" in options
    assert "--verifier-signing-key" in options
    assert "--verifier-identity" in options
    assert "--allow-installed-scorers" in options
    assert "--trust-profile" in options
    assert "--baseline" not in options


def test_report_renders_directly_from_the_bundle() -> None:
    options = _options("report")

    assert _arguments("report") == {"evidence"}
    assert "--html" in options
    assert "--run" not in options
