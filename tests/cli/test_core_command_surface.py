import os
import re

from click.termui import strip_ansi
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app


def test_top_level_help_lists_only_core_and_advanced_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)

    for name in ("evaluate", "report", "verify", "doctor", "advanced", "version"):
        assert re.search(rf"^\s*│\s+{re.escape(name)}\s", out, re.MULTILINE)

    for removed in ("run", "evidence-pack", "policy", "plugins", "calibrate"):
        assert not re.search(rf"^\s*│\s+{re.escape(removed)}\s", out, re.MULTILINE)


def test_advanced_help_lists_advanced_commands():
    result = CliRunner().invoke(app, ["advanced", "--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)

    for name in ("evidence-pack", "policy", "plugins", "calibrate"):
        assert re.search(rf"^\s*│\s+{re.escape(name)}\s", out, re.MULTILINE)
