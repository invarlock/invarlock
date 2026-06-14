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
        "tiiuae/falcon-7b",
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
        assert entry["quant_rtn"]["group_size"] in {32, 64}
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


def test_olmo13_tuned_lowrank_clean_is_retuned_and_exact() -> None:
    payload = _load_tuned_edit_params()

    olmo13 = payload["models"]["allenai/OLMo-2-1124-13B-Instruct"]
    assert olmo13["lowrank_svd"] == {
        "edit_dir_name": "svd_rank32_clean",
        "rank": 32,
        "reason": "selected_by_supported_experimental_recheck_retune:rank32_ffn_layer31",
        "scope": "ffn@layer=31",
        "status": "selected",
    }


def test_qwen2_7b_tuned_edit_params_are_exact() -> None:
    payload = _load_tuned_edit_params()

    assert payload["models"]["Qwen/Qwen2-7B"] == {
        "fp8_quant": {
            "edit_dir_name": "fp8_e5m2_clean",
            "format": "e5m2",
            "reason": "selected_by_supported_experimental_recheck_seed:e5m2_ffn",
            "scope": "ffn",
            "status": "selected",
        },
        "lowrank_svd": {
            "edit_dir_name": "svd_rank32_clean",
            "rank": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:rank32_ffn_layer15",
            "scope": "ffn@layer=15",
            "status": "selected",
        },
        "magnitude_prune": {
            "edit_dir_name": "prune_clean",
            "reason": "selected_by_supported_experimental_recheck_seed:sparsity120_ffn",
            "scope": "ffn",
            "sparsity": 0.12,
            "status": "selected",
        },
        "quant_rtn": {
            "bits": 4,
            "edit_dir_name": "quant_4bit_clean",
            "group_size": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:bits4_g32_ffn",
            "scope": "ffn",
            "status": "selected",
        },
    }


def test_qwen25_14b_tuned_edit_params_are_exact() -> None:
    payload = _load_tuned_edit_params()

    assert payload["models"]["Qwen/Qwen2.5-14B"] == {
        "fp8_quant": {
            "edit_dir_name": "fp8_e5m2_clean",
            "format": "e5m2",
            "reason": "selected_by_evaluate_pass:e5m2_ffn",
            "scope": "ffn",
            "status": "selected",
        },
        "lowrank_svd": {
            "edit_dir_name": "svd_rank32_clean",
            "rank": 32,
            "reason": "selected_by_evaluate_pass:rank32_ffn_layer31",
            "scope": "ffn@layer=31",
            "status": "selected",
        },
        "magnitude_prune": {
            "edit_dir_name": "prune_clean",
            "reason": "selected_by_evaluate_pass:sparsity120_attn",
            "scope": "attn",
            "sparsity": 0.12,
            "status": "selected",
        },
        "quant_rtn": {
            "bits": 4,
            "edit_dir_name": "quant_4bit_clean",
            "group_size": 32,
            "reason": "selected_by_evaluate_pass:bits4_g32_attn",
            "scope": "attn",
            "status": "selected",
        },
    }


def test_qwen3_8b_tuned_lowrank_clean_is_retuned_and_exact() -> None:
    payload = _load_tuned_edit_params()

    qwen3 = payload["models"]["Qwen/Qwen3-8B"]
    assert qwen3["lowrank_svd"] == {
        "edit_dir_name": "svd_rank32_clean",
        "rank": 32,
        "reason": "selected_by_supported_experimental_recheck_retune:rank32_ffn_layer15",
        "scope": "ffn@layer=15",
        "status": "selected",
    }


def test_deepseek_r1_distill_qwen_7b_tuned_lowrank_clean_is_retuned_and_exact() -> None:
    payload = _load_tuned_edit_params()

    deepseek = payload["models"]["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]
    assert deepseek["lowrank_svd"] == {
        "edit_dir_name": "svd_rank32_clean",
        "rank": 32,
        "reason": "selected_by_supported_experimental_recheck_retune:rank32_ffn_layer17",
        "scope": "ffn@layer=17",
        "status": "selected",
    }


def test_ministral3_8b_tuned_clean_quant_and_prune_are_retuned_and_exact() -> None:
    payload = _load_tuned_edit_params()

    ministral = payload["models"]["mistralai/Ministral-3-8B-Instruct-2512-BF16"]
    assert ministral["magnitude_prune"] == {
        "edit_dir_name": "prune_clean",
        "reason": "selected_by_supported_experimental_recheck_retune:sparsity080_ffn",
        "scope": "ffn",
        "sparsity": 0.08,
        "status": "selected",
    }
    assert ministral["quant_rtn"] == {
        "bits": 4,
        "edit_dir_name": "quant_4bit_clean",
        "group_size": 64,
        "reason": "selected_by_supported_experimental_recheck_retune:bits4_g64_ffn",
        "scope": "ffn",
        "status": "selected",
    }


def test_qwen3_5_9b_tuned_edit_params_are_exact() -> None:
    payload = _load_tuned_edit_params()

    assert payload["models"]["Qwen/Qwen3.5-9B"] == {
        "fp8_quant": {
            "edit_dir_name": "fp8_e5m2_clean",
            "format": "e5m2",
            "reason": "selected_by_supported_experimental_recheck_seed:e5m2_ffn",
            "scope": "ffn",
            "status": "selected",
        },
        "lowrank_svd": {
            "edit_dir_name": "svd_rank32_clean",
            "rank": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:rank32_ffn_layer15",
            "scope": "ffn@layer=15",
            "status": "selected",
        },
        "magnitude_prune": {
            "edit_dir_name": "prune_clean",
            "reason": "selected_by_supported_experimental_recheck_seed:sparsity120_ffn",
            "scope": "ffn",
            "sparsity": 0.12,
            "status": "selected",
        },
        "quant_rtn": {
            "bits": 4,
            "edit_dir_name": "quant_4bit_clean",
            "group_size": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:bits4_g32_ffn",
            "scope": "ffn",
            "status": "selected",
        },
    }


def test_ministral3_14b_tuned_edit_params_are_exact() -> None:
    payload = _load_tuned_edit_params()

    assert payload["models"]["mistralai/Ministral-3-14B-Instruct-2512-BF16"] == {
        "fp8_quant": {
            "edit_dir_name": "fp8_e5m2_clean",
            "format": "e5m2",
            "reason": "selected_by_supported_experimental_recheck_seed:e5m2_ffn",
            "scope": "ffn",
            "status": "selected",
        },
        "lowrank_svd": {
            "edit_dir_name": "svd_rank32_clean",
            "rank": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:rank32_ffn_layer19",
            "scope": "ffn@layer=19",
            "status": "selected",
        },
        "magnitude_prune": {
            "edit_dir_name": "prune_clean",
            "reason": "selected_by_supported_experimental_recheck_seed:sparsity100_ffn",
            "scope": "ffn",
            "sparsity": 0.1,
            "status": "selected",
        },
        "quant_rtn": {
            "bits": 4,
            "edit_dir_name": "quant_4bit_clean",
            "group_size": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:bits4_g32_ffn",
            "scope": "ffn",
            "status": "selected",
        },
    }


def test_phi4_tuned_edit_params_are_exact() -> None:
    payload = _load_tuned_edit_params()

    assert payload["models"]["microsoft/Phi-4-reasoning-plus"] == {
        "fp8_quant": {
            "edit_dir_name": "fp8_e5m2_clean",
            "format": "e5m2",
            "reason": "selected_by_supported_experimental_recheck_seed:e5m2_ffn",
            "scope": "ffn",
            "status": "selected",
        },
        "lowrank_svd": {
            "edit_dir_name": "svd_rank32_clean",
            "rank": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:rank32_ffn_layer19",
            "scope": "ffn@layer=19",
            "status": "selected",
        },
        "magnitude_prune": {
            "edit_dir_name": "prune_clean",
            "reason": "selected_by_supported_experimental_recheck_seed:sparsity080_ffn",
            "scope": "ffn",
            "sparsity": 0.08,
            "status": "selected",
        },
        "quant_rtn": {
            "bits": 4,
            "edit_dir_name": "quant_4bit_clean",
            "group_size": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:bits4_g32_ffn",
            "scope": "ffn",
            "status": "selected",
        },
    }


def test_gemma4_e2b_tuned_edit_params_are_exact() -> None:
    payload = _load_tuned_edit_params()

    assert payload["models"]["google/gemma-4-E2B-it"] == {
        "fp8_quant": {
            "edit_dir_name": "fp8_e5m2_clean",
            "format": "e5m2",
            "reason": "selected_by_supported_experimental_recheck_seed:e5m2_ffn",
            "scope": "ffn",
            "status": "selected",
        },
        "lowrank_svd": {
            "edit_dir_name": "svd_rank32_clean",
            "rank": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:rank32_ffn_layer17",
            "scope": "ffn@layer=17",
            "status": "selected",
        },
        "magnitude_prune": {
            "edit_dir_name": "prune_clean",
            "reason": "selected_by_supported_experimental_recheck_seed:sparsity080_ffn",
            "scope": "ffn",
            "sparsity": 0.08,
            "status": "selected",
        },
        "quant_rtn": {
            "bits": 4,
            "edit_dir_name": "quant_4bit_clean",
            "group_size": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:bits4_g32_ffn",
            "scope": "ffn",
            "status": "selected",
        },
    }


def test_tinyllama_tuned_edit_params_are_exact() -> None:
    payload = _load_tuned_edit_params()

    assert payload["models"]["TinyLlama/TinyLlama-1.1B-Chat-v1.0"] == {
        "fp8_quant": {
            "edit_dir_name": "fp8_e5m2_clean",
            "format": "e5m2",
            "reason": "selected_by_supported_experimental_recheck_seed:e5m2_ffn",
            "scope": "ffn",
            "status": "selected",
        },
        "lowrank_svd": {
            "edit_dir_name": "svd_rank32_clean",
            "rank": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:rank32_ffn_layer11",
            "scope": "ffn@layer=11",
            "status": "selected",
        },
        "magnitude_prune": {
            "edit_dir_name": "prune_clean",
            "reason": "selected_by_supported_experimental_recheck_seed:sparsity080_ffn",
            "scope": "ffn",
            "sparsity": 0.08,
            "status": "selected",
        },
        "quant_rtn": {
            "bits": 4,
            "edit_dir_name": "quant_4bit_clean",
            "group_size": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:bits4_g32_ffn",
            "scope": "ffn",
            "status": "selected",
        },
    }


def test_olmo2_7b_tuned_edit_params_are_exact() -> None:
    payload = _load_tuned_edit_params()

    assert payload["models"]["allenai/OLMo-2-1124-7B"] == {
        "fp8_quant": {
            "edit_dir_name": "fp8_e5m2_clean",
            "format": "e5m2",
            "reason": "selected_by_supported_experimental_recheck_seed:e5m2_ffn",
            "scope": "ffn",
            "status": "selected",
        },
        "lowrank_svd": {
            "edit_dir_name": "svd_rank32_clean",
            "rank": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:rank32_ffn_layer15",
            "scope": "ffn@layer=15",
            "status": "selected",
        },
        "magnitude_prune": {
            "edit_dir_name": "prune_clean",
            "reason": "selected_by_supported_experimental_recheck_seed:sparsity100_ffn",
            "scope": "ffn",
            "sparsity": 0.1,
            "status": "selected",
        },
        "quant_rtn": {
            "bits": 4,
            "edit_dir_name": "quant_4bit_clean",
            "group_size": 32,
            "reason": "selected_by_supported_experimental_recheck_seed:bits4_g32_ffn",
            "scope": "ffn",
            "status": "selected",
        },
    }
