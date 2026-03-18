from __future__ import annotations

from pathlib import Path

import yaml

from invarlock.cli.config import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_causal_lm_family_presets_load() -> None:
    root = _repo_root()
    presets = {
        "wikitext2_512.yaml": "sshleifer/tiny-gpt2",
        "llama3_1_8b_512.yaml": "meta-llama/Llama-3.1-8B-Instruct",
        "mistral_7b_512.yaml": "mistralai/Mistral-7B-v0.1",
        "qwen2_7b_512.yaml": "Qwen/Qwen2-7B",
        "qwen3_8b_512.yaml": "Qwen/Qwen3-8B",
        "gemma3_4b_it_512.yaml": "google/gemma-3-4b-it",
        "deepseek_r1_distill_qwen_7b_512.yaml": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "phi4_reasoning_plus_512.yaml": "microsoft/Phi-4-reasoning-plus",
        "olmo2_7b_512.yaml": "allenai/OLMo-2-1124-7B",
        "qwen3_5_9b_512.yaml": "Qwen/Qwen3.5-9B",
        "deepseek_v3_0324_512.yaml": "deepseek-ai/DeepSeek-V3-0324",
    }
    for name, model_id in presets.items():
        cfg = load_config(root / "configs/presets/causal_lm" / name)
        assert cfg.model.id == model_id
        assert cfg.model.adapter == "hf_causal"


def test_null_sweep_calibration_configs_reference_models() -> None:
    root = _repo_root()
    configs = {
        "null_sweep_llama3_1_8b.yaml": "meta-llama/Llama-3.1-8B-Instruct",
        "null_sweep_mistral_7b.yaml": "mistralai/Mistral-7B-v0.1",
        "null_sweep_qwen2_7b.yaml": "Qwen/Qwen2-7B",
        "null_sweep_qwen3_8b.yaml": "Qwen/Qwen3-8B",
        "null_sweep_gemma3_4b_it.yaml": "google/gemma-3-4b-it",
        "null_sweep_deepseek_r1_distill_qwen_7b.yaml": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "null_sweep_phi4_reasoning_plus.yaml": "microsoft/Phi-4-reasoning-plus",
        "null_sweep_olmo2_7b.yaml": "allenai/OLMo-2-1124-7B",
        "null_sweep_qwen3_5_9b.yaml": "Qwen/Qwen3.5-9B",
        "null_sweep_deepseek_v3_0324.yaml": "deepseek-ai/DeepSeek-V3-0324",
    }
    for name, model_id in configs.items():
        data = yaml.safe_load(
            (root / "configs/calibration" / name).read_text(encoding="utf-8")
        )
        assert data["model"]["id"] == model_id
