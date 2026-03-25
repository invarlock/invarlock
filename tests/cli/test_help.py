import os
import re

from click.termui import strip_ansi
from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app


def test_invarlock_help_layout_and_exit_codes():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)

    # Core copy
    assert "evaluate model changes" in out.lower()
    assert "invarlock evaluate --baseline" in out and "--subject" in out

    # Exit codes surfaced (normalize whitespace to avoid wrapping issues)
    normalized = " ".join(out.split())
    assert "0=success" in normalized and "1=generic failure" in normalized
    assert "2=schema invalid" in normalized and "3=hard abort" in normalized

    # Command names presence (order may vary with Typer versions)
    for name in ("evaluate", "report", "verify", "doctor", "advanced", "version"):
        assert re.search(rf"^\s*│\s+{re.escape(name)}\s", out, re.MULTILINE)
    for removed in ("proof-pack", "run", "plugins", "policy", "calibrate"):
        assert not re.search(rf"^\s*│\s+{re.escape(removed)}\s", out, re.MULTILINE)


def test_invarlock_version_option():
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "InvarLock" in strip_ansi(result.stdout)


def test_report_group_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    for sub in ("verify", "explain", "html", "validate"):
        assert sub in out


def test_advanced_group_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(app, ["advanced", "--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    for sub in ("proof-pack", "policy", "plugins", "calibrate"):
        assert sub in out


def test_advanced_plugins_adapters_json_disabled_discovery():
    runner = CliRunner()
    # Keep third-party discovery off to enforce lightweight path and stable JSON
    import os as _os

    _os.environ["INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS"] = "0"
    res = runner.invoke(app, ["advanced", "plugins", "adapters", "--json"])
    assert res.exit_code == 0, res.output
    import json as _json

    payload = _json.loads(res.output)
    assert payload.get("category") == "adapters"
    assert "kind" not in payload
    names = {item.get("name") for item in payload.get("items", [])}
    assert "hf_causal" in names
