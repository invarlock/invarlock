import os
import re

from click.termui import strip_ansi
from click.testing import CliRunner as ClickRunner
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app
from invarlock.cli.commands.verify import runtime_verify_app


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

    for name in ("evidence-pack", "policy", "plugins", "calibrate", "runtime-verify"):
        assert re.search(rf"^\s*│\s+{re.escape(name)}\s", out, re.MULTILINE)


def test_report_help_lists_report_subcommands() -> None:
    result = CliRunner().invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)

    for name in ("generate", "explain", "html", "validate"):
        assert re.search(rf"^\s*│\s+{re.escape(name)}\s", out, re.MULTILINE)


def test_runtime_verify_help_lists_required_flags() -> None:
    result = ClickRunner().invoke(runtime_verify_app, ["--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "COMMAND [ARGS]..." not in out
    assert "runtime.manifest.json companion" in out
    assert "--report" in out
    assert "--manifest" in out
    assert "--json" in out
