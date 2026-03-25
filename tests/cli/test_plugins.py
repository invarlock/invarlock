from __future__ import annotations

import json

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_plugins_discovery_disabled_minimal_json_adapters():
    r = CliRunner().invoke(
        app,
        ["advanced", "plugins", "adapters", "--json"],
        env={"INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS": "0"},
    )
    assert r.exit_code == 0
    obj = json.loads(r.stdout.strip().splitlines()[-1])
    assert obj.get("category") == "adapters"
    assert "kind" not in obj
    assert isinstance(obj.get("items"), list)
    assert any(item.get("name") == "hf_causal" for item in obj["items"])


def test_plugins_explain_unknown_adapter_exits():
    r = CliRunner().invoke(
        app, ["advanced", "plugins", "adapters", "--explain", "__unknown_adapter__"]
    )
    assert r.exit_code == 1
