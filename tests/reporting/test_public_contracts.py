from __future__ import annotations

from pathlib import Path

import invarlock.public_contracts as contracts


def test_public_contract_loaders_and_catalog_round_trip() -> None:
    support_matrix = contracts.load_support_matrix()
    assert support_matrix["format_version"] == "support-matrix-v1"
    assert {lane["lane_id"] for lane in contracts.published_basis_lanes()} == {
        "gpt2-causal-hf",
        "bert-mlm-hf",
    }

    family_catalog = contracts.load_model_family_catalog()
    assert family_catalog["format_version"] == "model-family-catalog-v1"
    assert family_catalog["as_of"] == "2026-03-18"
    assert family_catalog["declared_support"][0]["display_name"] == "GPT-2 causal LM"
    assert (
        family_catalog["recommended_additions"][0]["display_name"]
        == "Full multimodal evaluation pipeline"
    )
    assert {item["display_name"] for item in family_catalog["declared_support"]} >= {
        "Llama 3.1 / 3.3 causal LM",
        "Qwen3 causal LM",
        "Gemma 3 causal LM (text-only eval)",
        "DeepSeek-R1-Distill-Qwen causal LM",
        "Phi-4 causal LM (text-only eval)",
        "OLMo 2 causal LM",
        "Qwen3.5 causal LM",
        "DeepSeek-V3 causal LM",
    }

    gpt2_lane = contracts.support_lane_by_id("gpt2-causal-hf")
    assert gpt2_lane is not None
    assert gpt2_lane["support_tier"] == "published_basis"

    onnx_capability = contracts.adapter_capability("hf_causal_onnx")
    assert onnx_capability is not None
    assert onnx_capability["guard_coverage"] == "eval_only"

    catalog = contracts.contract_catalog()
    assert catalog["support_matrix"]["format_version"] == "support-matrix-v1"
    assert (
        catalog["model_family_catalog"]["format_version"] == "model-family-catalog-v1"
    )
    assert catalog["plugin_compatibility"]["core_abi"] == "0.1"
    assert catalog["plugin_compatibility"]["match_policy"] == "exact_match"
    assert catalog["policy_pack"]["path"] == "contracts/policy_pack.schema.json"

    schema = contracts.load_policy_pack_schema()
    assert schema["title"] == "InvarLock Policy Pack"
    assert (
        contracts.load_proof_pack_manifest_schema()["title"]
        == "InvarLock Proof Pack Manifest"
    )


def test_public_contract_paths_are_repo_relative() -> None:
    path = contracts.contract_path("support_matrix.json")
    assert path == contracts.CONTRACTS_ROOT / "support_matrix.json"
    assert path.is_file()
    assert (
        contracts.contract_relpath("support_matrix.json")
        == "contracts/support_matrix.json"
    )
    assert Path(
        contracts.contract_reference("support_matrix.json")["path"]
    ).as_posix() == ("contracts/support_matrix.json")


def test_public_contract_helpers_fall_back_when_contracts_are_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        contracts,
        "load_json_contract",
        lambda _filename: (_ for _ in ()).throw(OSError("missing")),
    )

    assert contracts.load_support_matrix() == {
        "format_version": "support-matrix-v1",
        "lanes": [],
    }
    assert contracts.load_adapter_capabilities() == {
        "format_version": "adapter-capabilities-v1",
        "adapters": [],
    }
    assert contracts.load_model_family_catalog() == {
        "format_version": "model-family-catalog-v1",
        "declared_support": [],
        "implemented_coverage": [],
        "usage_only": [],
        "recommended_additions": [],
    }
    assert contracts.load_plugin_compatibility() == {
        "format_version": "plugin-compatibility-v1"
    }
    assert contracts.load_policy_pack_schema() == {}
    assert contracts.load_proof_pack_manifest_schema() == {}
    assert contracts.support_lane_by_id("missing") is None
    assert contracts.adapter_capability("missing") is None
    assert contracts.contract_reference("support_matrix.json") == {
        "path": "contracts/support_matrix.json"
    }


def test_public_contract_helpers_reject_non_mapping_payloads(monkeypatch) -> None:
    payloads = {
        "support_matrix.json": ["unexpected"],
        "model_family_catalog.json": "unexpected",
        "adapter_capabilities.json": "unexpected",
        "plugin_compatibility.json": ["unexpected"],
        "policy_pack.schema.json": ["unexpected"],
        "proof_pack_manifest.schema.json": ["unexpected"],
    }
    monkeypatch.setattr(
        contracts,
        "_safe_load",
        lambda filename, default: payloads.get(filename, default),
    )

    assert contracts.load_support_matrix() == {
        "format_version": "support-matrix-v1",
        "lanes": [],
    }
    assert contracts.load_adapter_capabilities() == {
        "format_version": "adapter-capabilities-v1",
        "adapters": [],
    }
    assert contracts.load_model_family_catalog() == {
        "format_version": "model-family-catalog-v1",
        "declared_support": [],
        "implemented_coverage": [],
        "usage_only": [],
        "recommended_additions": [],
    }
    assert contracts.load_plugin_compatibility() == {
        "format_version": "plugin-compatibility-v1"
    }
    assert contracts.load_policy_pack_schema() == {}
    assert contracts.load_proof_pack_manifest_schema() == {}


def test_public_contract_lane_and_adapter_helpers_cover_non_matching_entries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        contracts,
        "_safe_load",
        lambda filename, default: {
            "support_matrix.json": {
                "format_version": "support-matrix-v1",
                "lanes": [
                    {"lane_id": "first", "support_tier": "community_experimental"},
                    {"lane_id": "second", "support_tier": "published_basis"},
                    "bad",
                ],
            },
            "adapter_capabilities.json": {
                "format_version": "adapter-capabilities-v1",
                "adapters": [
                    {"adapter": "", "guard_coverage": "none"},
                    {"adapter": "good", "guard_coverage": "full"},
                    "bad",
                ],
            },
            "model_family_catalog.json": {
                "format_version": "model-family-catalog-v1",
                "declared_support": [
                    {"family_id": "gpt2-causal-lm", "display_name": "GPT-2 causal LM"},
                    "bad",
                ],
                "implemented_coverage": [],
                "usage_only": [],
                "recommended_additions": [],
            },
            "plugin_compatibility.json": {
                "format_version": "plugin-compatibility-v1",
                "format": "compatibility-doc",
                "core_abi": "0.1",
                "match_policy": "exact_match",
            },
        }.get(filename, default),
    )

    assert contracts.support_lane_by_id("missing") is None
    assert contracts.support_lane_by_id("second") == {
        "lane_id": "second",
        "support_tier": "published_basis",
    }
    assert contracts.adapter_capability_map() == {
        "good": {"adapter": "good", "guard_coverage": "full"}
    }
    assert contracts.load_model_family_catalog()["declared_support"][0] == {
        "family_id": "gpt2-causal-lm",
        "display_name": "GPT-2 causal LM",
    }
    assert contracts.contract_reference("plugin_compatibility.json") == {
        "path": "contracts/plugin_compatibility.json",
        "format_version": "plugin-compatibility-v1",
        "format": "compatibility-doc",
        "core_abi": "0.1",
        "match_policy": "exact_match",
    }
