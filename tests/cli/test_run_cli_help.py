import os

from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app

runner = CliRunner()


def test_cli_run_help_includes_edit_label_and_metric_kind():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--edit-label" in result.stdout
    assert "--metric-kind" in result.stdout


def test_cli_run_accepts_edit_label_flag():
    result = runner.invoke(app, ["run", "--edit-label", "noop", "--help"])
    assert result.exit_code == 0
