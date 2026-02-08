import os
import re

from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app

runner = CliRunner()

_ANSI_ESCAPE_RE = re.compile(r"\x1b\\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def test_cli_run_help_includes_edit_label_and_metric_kind():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    stdout = _strip_ansi(result.stdout)
    assert "--edit-label" in stdout
    assert "--metric-kind" in stdout


def test_cli_run_accepts_edit_label_flag():
    result = runner.invoke(app, ["run", "--edit-label", "noop", "--help"])
    assert result.exit_code == 0
