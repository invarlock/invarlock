from __future__ import annotations

import json

from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.public_contracts import load_adapter_capabilities, load_support_matrix


def test_adapter_capabilities_contract_is_exposed_through_plugins_json() -> None:
    contract = load_adapter_capabilities()
    expected = {item["adapter"]: item for item in contract["adapters"]}

    res = CliRunner().invoke(
        app, ["advanced", "plugins", "adapters", "--json", "--show-unsupported"]
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout.strip().splitlines()[-1])

    assert (
        payload["contracts"]["adapter_capabilities"]["format_version"]
        == contract["format_version"]
    )
    items = {item["name"]: item for item in payload["items"]}

    for adapter_name in (
        "hf_causal",
        "hf_mlm",
        "hf_multimodal",
        "hf_seq2seq",
        "hf_bnb",
        "hf_awq",
        "hf_gptq",
    ):
        assert adapter_name in expected
        assert items[adapter_name]["capability"] == expected[adapter_name]


def test_support_matrix_adapters_have_capability_entries() -> None:
    capabilities = {
        item["adapter"]
        for item in load_adapter_capabilities()["adapters"]
        if isinstance(item.get("adapter"), str)
    }
    support_adapters = {
        lane["adapter"]
        for lane in load_support_matrix()["lanes"]
        if isinstance(lane.get("adapter"), str)
    }

    assert support_adapters <= capabilities
    assert "hf_auto" in capabilities
