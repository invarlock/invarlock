from __future__ import annotations

import json

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_plugins_adapter_alias_json():
    r1 = CliRunner().invoke(app, ["plugins", "adapters", "--json"])
    r2 = CliRunner().invoke(app, ["plugins", "adapter", "--json"])
    assert r1.exit_code == 0 and r2.exit_code == 0
    p1 = json.loads(r1.stdout.strip().splitlines()[-1])
    p2 = json.loads(r2.stdout.strip().splitlines()[-1])
    assert p1.get("items") == p2.get("items")


def test_plugins_guards_and_edits_text_and_json():
    # Compact text table
    r = CliRunner().invoke(app, ["plugins", "guards"])
    assert r.exit_code == 0 and "Guard Plugins" in r.stdout
    # Verbose table
    rv = CliRunner().invoke(app, ["plugins", "edits", "--verbose"])
    assert rv.exit_code == 0 and "Edit Plugins" in rv.stdout
    # JSON shapes
    rj = CliRunner().invoke(app, ["plugins", "guards", "--json"])
    assert rj.exit_code == 0
    pj = json.loads(rj.stdout.strip().splitlines()[-1])
    assert pj.get("format_version") == "plugins-v1" and pj.get("category") == "guards"


def test_plugins_edits_explain_unknown():
    r = CliRunner().invoke(app, ["plugins", "edits", "--explain", "__does_not_exist__"])
    assert r.exit_code in (1, 2)
