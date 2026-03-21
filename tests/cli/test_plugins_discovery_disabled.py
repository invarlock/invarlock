from __future__ import annotations

import importlib
import json


def test_plugins_discovery_disabled_json(monkeypatch, capsys):
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "0")
    mod = importlib.import_module("invarlock.cli.commands.plugins")
    # JSON output still includes built-ins while keeping third-party discovery off.
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
    assert payload["category"] == "adapters"
    assert "kind" not in payload
    assert any(item["name"] == "hf_causal" for item in payload["items"])


def test_plugins_discovery_disabled_message(monkeypatch, capsys):
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "0")
    mod = importlib.import_module("invarlock.cli.commands.plugins")
    # Table/text path still renders built-ins while keeping third-party discovery off.
    mod.plugins_command(
        category=None,
        only=None,
        verbose=False,
        json_out=False,
        explain=None,
        hide_unsupported=True,
    )
    out = capsys.readouterr().out
    assert "hf_causal" in out
