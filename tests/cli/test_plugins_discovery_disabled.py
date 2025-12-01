from __future__ import annotations

import importlib
import json


def test_plugins_discovery_disabled_json(monkeypatch, capsys):
    monkeypatch.setenv("INVARLOCK_DISABLE_PLUGIN_DISCOVERY", "1")
    mod = importlib.import_module("invarlock.cli.commands.plugins")
    # JSON output path returns a stable payload
    mod.plugins_command(
        category="adapters",
        only=None,
        verbose=False,
        json_out=True,
        explain=None,
        hide_unsupported=True,
    )
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["discovery"] == "disabled"
    assert payload["kind"] == "adapters"


def test_plugins_discovery_disabled_message(monkeypatch, capsys):
    monkeypatch.setenv("INVARLOCK_DISABLE_PLUGIN_DISCOVERY", "1")
    mod = importlib.import_module("invarlock.cli.commands.plugins")
    # Table/text path emits a terse message when discovery is disabled
    mod.plugins_command(
        category=None,
        only=None,
        verbose=False,
        json_out=False,
        explain=None,
        hide_unsupported=True,
    )
    out = capsys.readouterr().out
    assert "Plugin discovery disabled" in out
