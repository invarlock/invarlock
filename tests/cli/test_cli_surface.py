from __future__ import annotations

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


def test_root_teaches_only_the_core_user_journey() -> None:
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
