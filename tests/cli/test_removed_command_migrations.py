import os

from click.termui import strip_ansi
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app


def test_removed_commands_fail_with_migration_guidance():
    runner = CliRunner()
    for command in ("proof-pack", "policy", "plugins", "calibrate"):
        result = runner.invoke(app, [command])
        assert result.exit_code == 2
        out = strip_ansi(result.output)
        assert f"No such command '{command}'" in out
