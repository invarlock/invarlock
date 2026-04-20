from __future__ import annotations

import json
from pathlib import Path


def _load_tuned_edit_params() -> dict:
    return json.loads(
        Path("scripts/evidence_packs/tuned_edit_params.json").read_text(
            encoding="utf-8"
        )
    )


def _supported_experimental_models() -> list[str]:
    payload = json.loads(
        Path("contracts/model_family_catalog.json").read_text(encoding="utf-8")
    )
    models: list[str] = []
    for family in payload["declared_support"]:
        if family.get("state") != "supported_experimental":
            continue
        models.extend(family.get("representative_models", []))
    return models


def _deferred_candidate_models() -> list[str]:
    return [
        "openlm-research/open_llama_7b",
        "facebook/opt-1.3b",
        "tiiuae/falcon-7b",
        "THUDM/glm-4-9b-chat",
        "distilbert-base-uncased",
    ]


def test_supported_experimental_models_have_selected_clean_tuned_edit_params() -> None:
    payload = _load_tuned_edit_params()
    models = payload["models"]

    for model_id in _supported_experimental_models():
        assert model_id in models, model_id
        entry = models[model_id]
        assert set(entry) == {
            "fp8_quant",
            "lowrank_svd",
            "magnitude_prune",
            "quant_rtn",
        }

        assert entry["fp8_quant"]["edit_dir_name"] == "fp8_e5m2_clean"
        assert entry["fp8_quant"]["format"] == "e5m2"
        assert entry["fp8_quant"]["status"] == "selected"

        assert entry["lowrank_svd"]["edit_dir_name"] == "svd_rank32_clean"
        assert entry["lowrank_svd"]["rank"] == 32
        assert entry["lowrank_svd"]["status"] == "selected"
        assert entry["lowrank_svd"]["scope"].startswith("ffn@layer=")

        assert entry["magnitude_prune"]["edit_dir_name"] == "prune_clean"
        assert entry["magnitude_prune"]["status"] == "selected"
        assert 0 < entry["magnitude_prune"]["sparsity"] <= 0.12

        assert entry["quant_rtn"]["bits"] == 4
        assert entry["quant_rtn"]["edit_dir_name"] == "quant_4bit_clean"
        assert entry["quant_rtn"]["group_size"] == 32
        assert entry["quant_rtn"]["status"] == "selected"


def test_deferred_candidate_models_have_queue_ready_clean_tuned_edit_params() -> None:
    payload = _load_tuned_edit_params()
    models = payload["models"]

    for model_id in _deferred_candidate_models():
        assert model_id in models, model_id
        entry = models[model_id]
        assert set(entry) == {
            "fp8_quant",
            "lowrank_svd",
            "magnitude_prune",
            "quant_rtn",
        }
        assert entry["fp8_quant"]["status"] == "selected"
        assert entry["lowrank_svd"]["status"] == "selected"
        assert entry["magnitude_prune"]["status"] == "selected"
        assert entry["quant_rtn"]["status"] == "selected"


def test_qwen25_7b_tuned_edit_params_cover_clean_edit_matrix() -> None:
    payload = _load_tuned_edit_params()

    qwen25_7b = payload["models"]["Qwen/Qwen2.5-7B"]
    assert set(qwen25_7b) == {
        "fp8_quant",
        "lowrank_svd",
        "magnitude_prune",
        "quant_rtn",
    }
    assert qwen25_7b["fp8_quant"] == {
        "edit_dir_name": "fp8_e5m2_clean",
        "format": "e5m2",
        "reason": "selected_by_evaluate_pass:e5m2_ffn",
        "scope": "ffn",
        "status": "selected",
    }
    lowrank_svd = qwen25_7b["lowrank_svd"]
    assert lowrank_svd["edit_dir_name"] == "svd_rank32_clean"
    assert lowrank_svd["rank"] == 32
    assert lowrank_svd["reason"] == "selected_by_evaluate_pass:rank32_ffn_layer15"
    assert lowrank_svd["scope"] == "ffn@layer=15"
    assert lowrank_svd["status"] == "selected"
    assert qwen25_7b["magnitude_prune"] == {
        "edit_dir_name": "prune_clean",
        "reason": "selected_by_evaluate_pass:sparsity120_ffn",
        "scope": "ffn",
        "sparsity": 0.12,
        "status": "selected",
    }
    assert qwen25_7b["quant_rtn"] == {
        "bits": 4,
        "edit_dir_name": "quant_4bit_clean",
        "group_size": 32,
        "reason": "selected_by_evaluate_pass:bits4_g32_ffn",
        "scope": "ffn",
        "status": "selected",
    }


def test_mistral_7b_tuned_prune_clean_is_model_specific_and_stable() -> None:
    payload = _load_tuned_edit_params()

    mistral_7b = payload["models"]["mistralai/Mistral-7B-v0.1"]
    assert mistral_7b["magnitude_prune"] == {
        "edit_dir_name": "prune_clean",
        "reason": "selected_by_evaluate_pass:sparsity100_ffn",
        "scope": "ffn",
        "sparsity": 0.1,
        "status": "selected",
    }
