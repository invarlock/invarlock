from __future__ import annotations

import json

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_plugins_adapters_json():
    result = CliRunner().invoke(app, ["advanced", "plugins", "adapters", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert isinstance(payload.get("items"), list)


def test_plugins_guards_and_edits_text_and_json():
    # Compact text table
    r = CliRunner().invoke(app, ["advanced", "plugins", "guards"])
    assert r.exit_code == 0 and "Guard Plugins" in r.stdout
    r_edits = CliRunner().invoke(app, ["advanced", "plugins", "edits"])
    assert r_edits.exit_code == 0 and "Edit Plugins" in r_edits.stdout
    # Verbose table
    rv = CliRunner().invoke(app, ["advanced", "plugins", "edits", "--verbose"])
    assert rv.exit_code == 0 and "Edit Plugins" in rv.stdout
    # JSON shapes
    rj = CliRunner().invoke(app, ["advanced", "plugins", "guards", "--json"])
    assert rj.exit_code == 0
    pj = json.loads(rj.stdout.strip().splitlines()[-1])
    assert pj.get("format_version") == "plugins-v2" and pj.get("category") == "guards"


def test_plugins_datasets_text_table():
    result = CliRunner().invoke(app, ["advanced", "plugins", "list", "datasets"])
    assert result.exit_code == 0
    assert "Dataset Providers" in result.stdout


def test_plugins_edits_explain_unknown():
    r = CliRunner().invoke(
        app, ["advanced", "plugins", "edits", "--explain", "__does_not_exist__"]
    )
    assert r.exit_code in (1, 2)
