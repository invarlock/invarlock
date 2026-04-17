import os

from click.termui import strip_ansi
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app


def test_evaluate_help_mentions_execution_mode_and_not_legacy_execution_flags():
    result = CliRunner().invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)

    assert "--execution-mode" in out
    assert "Execution mode" in out
    assert "for evaluation" in out
    assert "--allow-host-execution" not in out
    assert "--allow-third-party-plugins" not in out
    assert "--allow-remote-code" not in out


def test_evaluate_rejects_unknown_execution_mode():
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--baseline",
            "baseline",
            "--subject",
            "subject",
            "--execution-mode",
            "invalid",
        ],
    )
    assert result.exit_code == 2


def test_evaluate_accepts_trusted_local_without_usage_error():
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--baseline",
            "baseline",
            "--subject",
            "subject",
            "--execution-mode",
            "trusted-local",
            "--help",
        ],
    )
    assert result.exit_code == 0
