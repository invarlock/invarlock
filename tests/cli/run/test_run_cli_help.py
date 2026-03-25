import os

from click.termui import strip_ansi
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from tests.cli.run._internal_cli import internal_run_app

runner = CliRunner()


def test_cli_run_help_includes_edit_label_and_metric_kind():
    result = runner.invoke(internal_run_app, ["run", "--help"])
    assert result.exit_code == 0
    stdout = strip_ansi(result.stdout)
    assert "--edit-label" in stdout
    assert "--metric-kind" in stdout
    assert "quant_rtn" in stdout
    assert "quant|mixed" not in stdout


def test_cli_run_accepts_edit_label_flag():
    result = runner.invoke(internal_run_app, ["run", "--edit-label", "noop", "--help"])
    assert result.exit_code == 0
