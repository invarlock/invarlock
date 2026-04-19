from __future__ import annotations

import json
from pathlib import Path


def test_qwen25_7b_tuned_edit_params_cover_clean_edit_matrix() -> None:
    payload = json.loads(
        Path("scripts/evidence_packs/tuned_edit_params.json").read_text(
            encoding="utf-8"
        )
    )

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
    payload = json.loads(
        Path("scripts/evidence_packs/tuned_edit_params.json").read_text(
            encoding="utf-8"
        )
    )

    mistral_7b = payload["models"]["mistralai/Mistral-7B-v0.1"]
    assert mistral_7b["magnitude_prune"] == {
        "edit_dir_name": "prune_clean",
        "reason": "selected_by_evaluate_pass:sparsity100_ffn",
        "scope": "ffn",
        "sparsity": 0.1,
        "status": "selected",
    }
