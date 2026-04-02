from __future__ import annotations

import json
from contextlib import contextmanager

import pytest
import typer
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.cli.commands.plugins import plugins_command


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


def test_plugins_command_resets_runtime_security_on_exit(monkeypatch):
    import invarlock.cli.security_helpers as security_helpers

    state = {"active": False, "configured": None, "enter": 0, "exit": 0}

    @contextmanager
    def fake_configure(**kwargs):
        state["configured"] = kwargs
        state["active"] = True
        state["enter"] += 1
        try:
            yield
        finally:
            state["active"] = False
            state["exit"] += 1

    monkeypatch.setattr(security_helpers, "configure_runtime_security", fake_configure)

    with pytest.raises(typer.Exit) as excinfo:
        plugins_command(category="__unknown__", allow_third_party_plugins=True)

    assert excinfo.value.exit_code == 2
    assert state["configured"]["allow_third_party_plugins"] is True
    assert state["active"] is False
    assert state["enter"] == 1
    assert state["exit"] == 1
