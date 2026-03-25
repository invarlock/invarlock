from __future__ import annotations

import json

from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.public_contracts import load_adapter_capabilities


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
        "hf_seq2seq",
        "hf_bnb",
        "hf_awq",
        "hf_gptq",
    ):
        assert adapter_name in expected
        assert items[adapter_name]["capability"] == expected[adapter_name]
