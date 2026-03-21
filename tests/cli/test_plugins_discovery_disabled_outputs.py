from __future__ import annotations

import json

from typer.testing import CliRunner

from invarlock.cli.app import app


def _invoke_json(args, env=None):
    r = CliRunner().invoke(app, args, env=env or {})
    assert r.exit_code == 0, r.output
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_plugins_discovery_disabled_for_all_categories_json():
    env = {"INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS": "0"}
    for cat in ("adapters", "guards", "edits", "plugins"):
        p = _invoke_json(["plugins", "list", cat, "--json"], env=env)
        assert p.get("category") == cat
        assert "kind" not in p
        assert isinstance(p.get("items"), list)
        assert p.get("items")


def test_plugins_datasets_json_shape():
    p = _invoke_json(["plugins", "list", "datasets", "--json"])
    assert p.get("category") == "datasets"
    assert p.get("format_version") == "plugins-v1"
    assert isinstance(p.get("items"), list)
