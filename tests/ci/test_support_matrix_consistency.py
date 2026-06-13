from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.public_contracts import load_support_matrix


def _parse_docs_support_labels() -> dict[str, str]:
    text = Path("docs/README.md").read_text(encoding="utf-8")
    marker = "## Support Matrix"
    assert marker in text
    section = text.split(marker, 1)[1]
    rows: dict[str, str] = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 5 or parts[0] in {"Surface", "-------"}:
            continue
        rows[parts[0]] = parts[4]
    return rows


def _lane_tiers(payload: dict) -> dict[str, str]:
    lanes = payload.get("support_matrix", {}).get("lanes", [])
    return {
        lane["lane_id"]: lane["support_tier"]
        for lane in lanes
        if isinstance(lane, dict)
        and isinstance(lane.get("lane_id"), str)
        and isinstance(lane.get("support_tier"), str)
    }


def test_support_matrix_contract_matches_docs_and_cli_json_surfaces() -> None:
    contract = load_support_matrix()
    docs_labels = _parse_docs_support_labels()

    runner = CliRunner()
    plugins = runner.invoke(app, ["advanced", "plugins", "adapters", "--json"])
    assert plugins.exit_code == 0, plugins.output
    plugins_payload = json.loads(plugins.stdout.strip().splitlines()[-1])

    doctor = runner.invoke(app, ["doctor", "--json"])
    assert doctor.exit_code in (0, 1), doctor.output
    doctor_payload = json.loads(doctor.stdout.strip().splitlines()[-1])

    assert (
        plugins_payload["support_matrix"]["format_version"]
        == contract["format_version"]
    )
    assert (
        doctor_payload["support_matrix"]["format_version"] == contract["format_version"]
    )

    contract_tiers = {
        lane["lane_id"]: lane["support_tier"] for lane in contract["lanes"]
    }
    assert _lane_tiers(plugins_payload) == contract_tiers
    assert _lane_tiers(doctor_payload) == contract_tiers

    families = {lane["family"]: lane for lane in contract["lanes"]}
    assert set(families) == {
        "GPT-2 causal LM",
        "BERT / RoBERTa MLM",
        "Mistral 7B causal LM",
        "Ministral 3 8B causal LM (text-only eval)",
        "Ministral 3 14B causal LM (text-only eval)",
        "Qwen2 7B causal LM",
        "Qwen2.5 7B causal LM",
        "Qwen2.5 14B causal LM",
        "Qwen3 causal LM",
        "DeepSeek-R1-Distill-Qwen causal LM",
        "Phi-4 causal LM (text-only eval)",
        "Gemma 4 E2B causal LM (text-only eval)",
        "TinyLlama 1.1B causal LM",
        "OLMo 2 7B causal LM",
        "OLMo 2 13B causal LM",
        "OpenLLaMA 7B causal LM",
        "Falcon 7B causal LM",
        "Qwen3.5 causal LM",
        "Gemma 4 12B any-to-any LM",
        "Ministral 3 3B causal LM (text-only eval)",
        "Granite 4.1 8B causal LM",
        "Granite 4.1 3B causal LM",
        "SmolLM3 3B causal LM",
        "Phi-4 mini causal LM",
        "DeepSeek-R1-Distill-Qwen 14B causal LM",
        "DeepSeek-R1-0528-Qwen3 8B causal LM",
        "Falcon-H1R 7B causal LM",
        "Seq2Seq / local pairs",
    }
    assert families["GPT-2 causal LM"]["support_tier"] == "published_basis"
    assert families["BERT / RoBERTa MLM"]["support_tier"] == "published_basis"
    assert families["Mistral 7B causal LM"]["support_tier"] == "published_basis"
    assert (
        families["Ministral 3 8B causal LM (text-only eval)"]["support_tier"]
        == "published_basis"
    )
    assert (
        families["Ministral 3 14B causal LM (text-only eval)"]["support_tier"]
        == "published_basis"
    )
    assert families["Qwen2 7B causal LM"]["support_tier"] == "published_basis"
    assert families["Qwen2.5 7B causal LM"]["support_tier"] == "published_basis"
    assert families["Qwen2.5 14B causal LM"]["support_tier"] == "published_basis"
    assert families["Qwen3 causal LM"]["support_tier"] == "published_basis"
    assert (
        families["DeepSeek-R1-Distill-Qwen causal LM"]["support_tier"]
        == "published_basis"
    )
    assert (
        families["Phi-4 causal LM (text-only eval)"]["support_tier"]
        == "published_basis"
    )
    assert (
        families["Gemma 4 E2B causal LM (text-only eval)"]["support_tier"]
        == "published_basis"
    )
    assert families["TinyLlama 1.1B causal LM"]["support_tier"] == "published_basis"
    assert families["OLMo 2 7B causal LM"]["support_tier"] == "published_basis"
    assert families["OLMo 2 13B causal LM"]["support_tier"] == "published_basis"
    assert families["OpenLLaMA 7B causal LM"]["support_tier"] == "published_basis"
    assert families["Falcon 7B causal LM"]["support_tier"] == "published_basis"
    assert families["Qwen3.5 causal LM"]["support_tier"] == "published_basis"
    assert families["Gemma 4 12B any-to-any LM"]["support_tier"] == (
        "published_basis"
    )
    assert families["Granite 4.1 3B causal LM"]["support_tier"] == "published_basis"
    assert families["Granite 4.1 8B causal LM"]["support_tier"] == "published_basis"
    assert (
        families["Ministral 3 3B causal LM (text-only eval)"]["support_tier"]
        == "published_basis"
    )
    assert (
        families["DeepSeek-R1-0528-Qwen3 8B causal LM"]["support_tier"]
        == "published_basis"
    )
    assert (
        families["DeepSeek-R1-Distill-Qwen 14B causal LM"]["support_tier"]
        == "published_basis"
    )
    assert families["SmolLM3 3B causal LM"]["support_tier"] == "published_basis"
    assert families["SmolLM3 3B causal LM"]["evidence_status"] == (
        "published_release_strict"
    )
    assert families["Phi-4 mini causal LM"]["support_tier"] == "published_basis"
    assert families["Phi-4 mini causal LM"]["evidence_status"] == (
        "published_release_strict"
    )
    assert families["Falcon-H1R 7B causal LM"]["support_tier"] == "published_basis"
    assert families["Falcon-H1R 7B causal LM"]["evidence_status"] == (
        "published_release_strict"
    )

    for family, lane in families.items():
        assert docs_labels[family] == lane["docs_label"]
        assert lane["support_groups"], family
