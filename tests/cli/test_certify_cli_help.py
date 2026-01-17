import os

from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app

runner = CliRunner()


def test_cli_certify_help():
    result = runner.invoke(app, ["certify", "--help"])
    assert result.exit_code == 0
    assert "--baseline" in result.stdout and "--subject" in result.stdout
    assert "--edit-label" in result.stdout
