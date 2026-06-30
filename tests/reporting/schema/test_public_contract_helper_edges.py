from __future__ import annotations

import invarlock.public_contracts as contracts


def test_public_contract_lane_and_adapter_helpers_cover_non_matching_entries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        contracts,
        "_load_contract_or_raise",
        lambda filename: {
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
                "published_basis_candidates_text_le_14b": {"candidates": []},
                "recommended_additions": [],
            },
            "plugin_compatibility.json": {
                "format_version": "plugin-compatibility-v1",
                "format": "compatibility-doc",
                "core_abi": "0.1",
                "match_policy": "exact_match",
            },
        }[filename],
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
    assert contracts.contract_reference("validation_keys.json") == {
        "path": "contracts/validation_keys.json",
        "kind": "array",
        "item_count": len(contracts.load_json_contract("validation_keys.json")),
    }
