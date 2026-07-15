import json

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app


def test_plugins_json_without_category_emits_one_envelope():
    result = CliRunner().invoke(app, ["advanced", "plugins", "list", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["format_version"] == "plugins-v2"
    assert payload["category"] == "all"
    assert isinstance(payload["items"], list)
    assert {item["kind"] for item in payload["items"]} == {
        "adapter",
        "dataset",
        "edit",
        "guard",
        "runtime_provider",
    }
    required = {"name", "kind", "module", "entry_point"}
    assert all(required <= set(item) for item in payload["items"])


@pytest.mark.parametrize(
    "cat", ["adapters", "guards", "edits", "runtime-providers", "plugins"]
)
def test_plugins_json_shape_and_order(cat):
    r = CliRunner().invoke(app, ["advanced", "plugins", "list", cat, "--json"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    assert payload["format_version"] == "plugins-v2"
    assert payload["category"] == cat
    items = payload["items"]
    assert isinstance(items, list) and items
    required = {"name", "kind", "module", "entry_point"}
    for row in items:
        assert required <= set(row.keys())
        assert row["kind"] in {
            "adapter",
            "guard",
            "edit",
            "plugin",
            "runtime_provider",
        }
    names = [(row["name"].lower(), row["kind"].lower()) for row in items]
    assert names == sorted(names)


def test_plugins_unknown_category_exit_code():
    r = CliRunner().invoke(app, ["advanced", "plugins", "list", "unknown-category"])
    assert r.exit_code == 2


def test_plugins_json_sorting_tie_breakers():
    # Ensure deterministic sort by (name, kind, module, entry_point)
    r = CliRunner().invoke(app, ["advanced", "plugins", "list", "plugins", "--json"])
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


def test_plugins_json_embeds_expanded_contract_catalog():
    r = CliRunner().invoke(app, ["advanced", "plugins", "list", "plugins", "--json"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    contracts = payload["contracts"]
    for key, filename in {
        "validation_keys": "validation_keys.json",
        "console_labels": "console_labels.json",
        "metric_kinds": "metric_kinds.json",
    }.items():
        assert contracts[key]["path"] == f"contracts/{filename}"
        assert contracts[key]["kind"] == "array"


def test_runtime_provider_json_inventory_is_metadata_only() -> None:
    result = CliRunner().invoke(
        app, ["advanced", "plugins", "runtime-providers", "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["format_version"] == "plugins-v2"
    assert payload["category"] == "runtime-providers"
    assert payload["items"] == [
        {
            "name": "hf_transformers",
            "kind": "runtime_provider",
            "module": "invarlock.runtime_providers.hf_transformers",
            "entry_point": None,
            "origin": "builtin",
            "status": "ready",
            "required_extra": "invarlock[hf]",
            "support_tier": "core_supported",
            "strict_assurance_allowed": True,
            "maintained_catalog": False,
            "deployment_claim": False,
        }
    ]


def test_runtime_provider_list_alias_and_text_surface() -> None:
    listed = CliRunner().invoke(
        app,
        ["advanced", "plugins", "list", "runtime-providers", "--json"],
    )
    rendered = CliRunner().invoke(app, ["advanced", "plugins", "runtime-providers"])

    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.stdout)["items"][0]["name"] == "hf_transformers"
    assert rendered.exit_code == 0, rendered.output
    assert "Runtime Providers" in rendered.stdout
    assert "hf_transformers" in rendered.stdout
