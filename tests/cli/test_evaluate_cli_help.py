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
    assert "--edit-label" in stdout
    assert "--allow-unattested-artifacts" in stdout
