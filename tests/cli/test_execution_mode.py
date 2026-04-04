import os

from click.termui import strip_ansi
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app


def test_evaluate_help_mentions_assurance_and_not_legacy_execution_flags():
    result = CliRunner().invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)

    assert "--assurance" in out
    assert "Assurance level" in out
    assert "for evaluation" in out
    assert "--allow-host-execution" not in out
    assert "--allow-third-party-plugins" not in out
    assert "--allow-remote-code" not in out


def test_evaluate_rejects_unknown_assurance():
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--baseline",
            "baseline",
            "--subject",
            "subject",
            "--assurance",
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
            "--assurance",
            "trusted-local",
            "--help",
        ],
    )
    assert result.exit_code == 0
