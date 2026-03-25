import os

from click.termui import strip_ansi
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app


def test_removed_commands_fail_with_migration_guidance():
    expected = {
        "run": "invarlock evaluate",
        "proof-pack": "advanced proof-pack",
        "policy": "advanced policy",
        "plugins": "advanced plugins",
        "calibrate": "advanced calibrate",
    }

    runner = CliRunner()
    for command, hint in expected.items():
        result = runner.invoke(app, [command])
        assert result.exit_code == 2
        out = strip_ansi(result.output)
        assert hint in out
