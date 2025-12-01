from __future__ import annotations

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_plugins_guards_and_edits_list_text():
    r = CliRunner().invoke(app, ["plugins", "guards"])
    assert r.exit_code == 0 and "Guard Plugins" in r.stdout
    r2 = CliRunner().invoke(app, ["plugins", "edits"])
    assert r2.exit_code == 0 and "Edit Plugins" in r2.stdout
