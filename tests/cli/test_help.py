import os

from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
from invarlock.cli.app import app


def test_invarlock_help_layout_and_exit_codes():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout

    # Core copy
    assert "certify model changes" in out.lower()
    # Updated help mentions built-in quant_rtn demo via --edit-config
    assert "built-in quant_rtn" in out.lower()
    assert "invarlock certify --baseline" in out and "--subject" in out

    # Exit codes surfaced (normalize whitespace to avoid wrapping issues)
    normalized = " ".join(out.split())
    assert "0=success" in normalized and "1=generic failure" in normalized
    assert "2=schema invalid" in normalized and "3=hard abort" in normalized

    # Command names presence (order may vary with Typer versions)
    for name in ("certify", "report", "run", "plugins", "doctor", "version"):
        assert name in out


def test_report_group_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    for sub in ("verify", "explain", "html", "validate"):
        assert sub in out


def test_plugins_group_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(app, ["plugins", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    for sub in ("list", "guards", "edits", "install", "uninstall"):
        assert sub in out


def test_plugins_adapters_json_disabled_discovery():
    runner = CliRunner()
    # Disable plugin discovery to enforce lightweight path and stable JSON
    import os as _os

    _os.environ["INVARLOCK_DISABLE_PLUGIN_DISCOVERY"] = "1"
    res = runner.invoke(app, ["plugins", "adapters", "--json"])
    assert res.exit_code == 0, res.output
    import json as _json

    payload = _json.loads(res.output)
    assert payload.get("kind") == "adapters"
    assert payload.get("items") == []
    assert payload.get("discovery") == "disabled"
