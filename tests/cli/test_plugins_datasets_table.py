from __future__ import annotations

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_plugins_datasets_text_table():
    r = CliRunner().invoke(app, ["plugins", "list", "datasets"])
    assert r.exit_code == 0
    # Should show a title and at least one provider row
    assert "Dataset Providers" in r.stdout
