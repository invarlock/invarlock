import os

from click.termui import strip_ansi
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app

runner = CliRunner()


def test_cli_evaluate_help():
    result = runner.invoke(app, ["evaluate", "--help"], env={"COLUMNS": "240"})
    assert result.exit_code == 0
    stdout = strip_ansi(result.stdout)
    assert "--baseline" in stdout and "--subject" in stdout
    assert "--baseline-report" in stdout
    assert "--baseline-revision" in stdout
    assert "--subject-revision" in stdout
    assert "--baseline-runtime-provider" in stdout
    assert "--subject-runtime-provider" in stdout
    assert "remote baseline" in stdout
    assert "remote subject" in stdout
    assert stdout.count("40-64 character lowercase hexadecimal revision") == 2
    assert "--edit-label" in stdout
    assert "--execution-mode" in stdout
    assert "--allow-remote-code" in stdout


def test_cli_verify_help_shows_runtime_provenance_choices():
    result = runner.invoke(app, ["verify", "--help"], env={"COLUMNS": "240"})
    assert result.exit_code == 0
    stdout = strip_ansi(result.stdout)
    compact = "".join(stdout.split())

    assert "--runtime-provenance" in stdout
    assert "container|host" in compact
    assert "--warning-policy" in stdout
