from __future__ import annotations

import json

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_plugins_discovery_disabled_minimal_json_adapters():
    r = CliRunner().invoke(
        app,
        ["plugins", "adapters", "--json"],
        env={"INVARLOCK_DISABLE_PLUGIN_DISCOVERY": "1"},
    )
    assert r.exit_code == 0
    obj = json.loads(r.stdout.strip().splitlines()[-1])
    assert obj.get("kind") == "adapters"
    assert obj.get("discovery") == "disabled"
    assert isinstance(obj.get("items"), list) and len(obj["items"]) == 0


def test_plugins_explain_unknown_adapter_exits():
    r = CliRunner().invoke(
        app, ["plugins", "adapters", "--explain", "__unknown_adapter__"]
    )
    assert r.exit_code == 1
