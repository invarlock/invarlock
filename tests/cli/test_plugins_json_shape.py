import json

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app


@pytest.mark.parametrize("cat", ["adapters", "guards", "edits", "plugins"])
def test_plugins_json_shape_and_order(cat):
    r = CliRunner().invoke(app, ["plugins", "list", cat, "--json"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    assert payload["format_version"] == "plugins-v1"
    assert payload["category"] == cat
    items = payload["items"]
    assert isinstance(items, list) and items
    required = {"name", "kind", "module", "entry_point"}
    for row in items:
        assert required <= set(row.keys())
        assert row["kind"] in {"adapter", "guard", "edit", "plugin"}
    names = [(row["name"].lower(), row["kind"].lower()) for row in items]
    assert names == sorted(names)


def test_plugins_unknown_category_exit_code():
    r = CliRunner().invoke(app, ["plugins", "list", "unknown-category"])
    assert r.exit_code == 2


def test_plugins_json_sorting_tie_breakers():
    # Ensure deterministic sort by (name, kind, module, entry_point)
    r = CliRunner().invoke(app, ["plugins", "list", "plugins", "--json"])
    assert r.exit_code == 0
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    items = payload.get("items", [])
    ordered = sorted(
        items,
        key=lambda r: (
            str(r.get("name", "")).lower(),
            str(r.get("kind", "")).lower(),
            str(r.get("module", "")).lower(),
            str(r.get("entry_point", "")).lower(),
        ),
    )
    assert items == ordered
