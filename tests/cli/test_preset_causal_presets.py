from __future__ import annotations

from pathlib import Path

import yaml

from invarlock.cli.config import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_causal_lm_family_presets_load() -> None:
    root = _repo_root()
    expected_drift_band = {"min": 0.9, "max": 1.2}
    presets = {
        "wikitext2_512.yaml": "sshleifer/tiny-gpt2",
        "mistral_7b_512.yaml": "mistralai/Mistral-7B-v0.1",
        "qwen2_7b_512.yaml": "Qwen/Qwen2-7B",
        "qwen3_8b_512.yaml": "Qwen/Qwen3-8B",
        "qwq_32b_512.yaml": "Qwen/QwQ-32B",
        "deepseek_r1_distill_qwen_7b_512.yaml": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "phi4_reasoning_plus_512.yaml": "microsoft/Phi-4-reasoning-plus",
        "tinyllama_1_1b_512.yaml": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "olmo2_7b_512.yaml": "allenai/OLMo-2-1124-7B",
        "olmo2_13b_512.yaml": "allenai/OLMo-2-1124-13B-Instruct",
        "qwen3_5_9b_512.yaml": "Qwen/Qwen3.5-9B",
    }
    expected_provider_kinds = {
        "phi4_reasoning_plus_512.yaml": "hf_text",
    }
    expected_skip_overhead = {
        "phi4_reasoning_plus_512.yaml",
    }
    for name, model_id in presets.items():
        cfg = load_config(root / "configs/presets/causal_lm" / name)
        assert cfg.model.id == model_id
        assert cfg.model.adapter == "hf_causal"
        provider = cfg.data["dataset"]["provider"]
        if name in expected_provider_kinds:
            assert provider["kind"] == expected_provider_kinds[name]
        else:
            assert provider == "wikitext2"
        if name in expected_skip_overhead:
            assert cfg.data["context"]["run"]["skip_overhead_check"] is True
        if name != "wikitext2_512.yaml":
            assert cfg.data["primary_metric"]["drift_band"] == expected_drift_band


def test_null_sweep_calibration_configs_reference_models() -> None:
    root = _repo_root()
    expected_drift_band = {"min": 0.9, "max": 1.2}
    configs = {
        "null_sweep_mistral_7b.yaml": "mistralai/Mistral-7B-v0.1",
        "null_sweep_qwen2_7b.yaml": "Qwen/Qwen2-7B",
        "null_sweep_qwen3_8b.yaml": "Qwen/Qwen3-8B",
        "null_sweep_qwq_32b.yaml": "Qwen/QwQ-32B",
        "null_sweep_deepseek_r1_distill_qwen_7b.yaml": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "null_sweep_phi4_reasoning_plus.yaml": "microsoft/Phi-4-reasoning-plus",
        "null_sweep_tinyllama_1_1b.yaml": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "null_sweep_olmo2_7b.yaml": "allenai/OLMo-2-1124-7B",
        "null_sweep_olmo2_13b.yaml": "allenai/OLMo-2-1124-13B-Instruct",
        "null_sweep_qwen3_5_9b.yaml": "Qwen/Qwen3.5-9B",
    }
    for name, model_id in configs.items():
        data = yaml.safe_load(
            (root / "configs/calibration" / name).read_text(encoding="utf-8")
        )
        assert data["model"]["id"] == model_id
        if name == "null_sweep_phi4_reasoning_plus.yaml":
            assert data["dataset"]["provider"]["kind"] == "hf_text"
        assert data["primary_metric"]["drift_band"] == expected_drift_band
