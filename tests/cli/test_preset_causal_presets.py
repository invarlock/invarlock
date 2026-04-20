from __future__ import annotations

from pathlib import Path

import yaml

from invarlock.core.config_loader import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_causal_lm_family_presets_load() -> None:
    root = _repo_root()
    expected_drift_band = {"min": 0.9, "max": 1.2}
    presets = {
        "wikitext2_512.yaml": "sshleifer/tiny-gpt2",
        "mistral_7b_512.yaml": "mistralai/Mistral-7B-v0.1",
        "ministral3_8b_512.yaml": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "ministral3_14b_512.yaml": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "qwen2_7b_512.yaml": "Qwen/Qwen2-7B",
        "qwen2_5_7b_512.yaml": "Qwen/Qwen2.5-7B",
        "qwen2_5_14b_512.yaml": "Qwen/Qwen2.5-14B",
        "qwen3_8b_512.yaml": "Qwen/Qwen3-8B",
        "deepseek_r1_distill_qwen_7b_512.yaml": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "phi4_reasoning_plus_512.yaml": "microsoft/Phi-4-reasoning-plus",
        "gemma4_e2b_512.yaml": "google/gemma-4-E2B-it",
        "tinyllama_1_1b_512.yaml": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "olmo2_7b_512.yaml": "allenai/OLMo-2-1124-7B",
        "olmo2_13b_512.yaml": "allenai/OLMo-2-1124-13B-Instruct",
        "qwen3_5_9b_512.yaml": "Qwen/Qwen3.5-9B",
    }
    expected_provider_kinds = {
        "deepseek_r1_distill_qwen_7b_512.yaml": "hf_text",
        "gemma4_e2b_512.yaml": "hf_text",
        "ministral3_8b_512.yaml": "hf_text",
        "ministral3_14b_512.yaml": "hf_text",
        "olmo2_13b_512.yaml": "hf_text",
        "olmo2_7b_512.yaml": "hf_text",
        "phi4_reasoning_plus_512.yaml": "hf_text",
        "qwen2_5_7b_512.yaml": "hf_text",
        "qwen2_5_14b_512.yaml": "hf_text",
        "qwen2_7b_512.yaml": "hf_text",
        "qwen3_5_9b_512.yaml": "hf_text",
        "qwen3_8b_512.yaml": "hf_text",
    }
    expected_skip_overhead = {
        "gemma4_e2b_512.yaml",
        "phi4_reasoning_plus_512.yaml",
    }
    for name, model_id in presets.items():
        cfg = load_config(root / "configs/presets/causal_lm" / name)
        assert cfg.require_section("model")["id"] == model_id
        assert cfg.require_section("model")["adapter"] == "hf_causal"
        if name == "gemma4_e2b_512.yaml":
            assert cfg.require_section("model")["attn_implementation"] == "sdpa"
        if name == "phi4_reasoning_plus_512.yaml":
            assert cfg.require_section("model")["trust_remote_code"] is True
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
        "null_sweep_ministral3_8b.yaml": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "null_sweep_ministral3_14b.yaml": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "null_sweep_qwen2_7b.yaml": "Qwen/Qwen2-7B",
        "null_sweep_qwen2_5_7b.yaml": "Qwen/Qwen2.5-7B",
        "null_sweep_qwen2_5_14b.yaml": "Qwen/Qwen2.5-14B",
        "null_sweep_qwen3_8b.yaml": "Qwen/Qwen3-8B",
        "null_sweep_deepseek_r1_distill_qwen_7b.yaml": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "null_sweep_phi4_reasoning_plus.yaml": "microsoft/Phi-4-reasoning-plus",
        "null_sweep_gemma4_e2b.yaml": "google/gemma-4-E2B-it",
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
            assert data["model"]["trust_remote_code"] is True
        if name in {
            "null_sweep_deepseek_r1_distill_qwen_7b.yaml",
            "null_sweep_gemma4_e2b.yaml",
            "null_sweep_ministral3_8b.yaml",
            "null_sweep_ministral3_14b.yaml",
            "null_sweep_olmo2_13b.yaml",
            "null_sweep_olmo2_7b.yaml",
            "null_sweep_phi4_reasoning_plus.yaml",
            "null_sweep_qwen2_5_7b.yaml",
            "null_sweep_qwen2_5_14b.yaml",
            "null_sweep_qwen2_7b.yaml",
            "null_sweep_qwen3_5_9b.yaml",
            "null_sweep_qwen3_8b.yaml",
        }:
            assert data["dataset"]["provider"]["kind"] == "hf_text"
        if name == "null_sweep_gemma4_e2b.yaml":
            assert data["model"]["attn_implementation"] == "sdpa"
        assert data["primary_metric"]["drift_band"] == expected_drift_band


def test_candidate_causal_lm_presets_load() -> None:
    root = _repo_root()
    expected_drift_band = {"min": 0.9, "max": 1.2}
    presets = {
        "openllama_7b_512.yaml": (
            "openlm-research/open_llama_7b",
            "hf_causal",
        ),
        "opt_1_3b_512.yaml": (
            "facebook/opt-1.3b",
            "hf_causal",
        ),
        "falcon_7b_512.yaml": (
            "tiiuae/falcon-7b",
            "auto",
        ),
        "glm4_9b_chat_512.yaml": (
            "THUDM/glm-4-9b-chat",
            "auto",
        ),
    }
    for name, (model_id, adapter) in presets.items():
        cfg = load_config(root / "configs/presets/causal_lm" / name)
        assert cfg.require_section("model")["id"] == model_id
        assert cfg.require_section("model")["adapter"] == adapter
        assert cfg.data["dataset"]["provider"]["kind"] == "hf_text"
        assert cfg.data["primary_metric"]["drift_band"] == expected_drift_band

    glm_cfg = load_config(root / "configs/presets/causal_lm" / "glm4_9b_chat_512.yaml")
    assert glm_cfg.require_section("model")["trust_remote_code"] is True


def test_candidate_null_sweep_calibration_configs_reference_models() -> None:
    root = _repo_root()
    expected_drift_band = {"min": 0.9, "max": 1.2}
    configs = {
        "null_sweep_openllama_7b.yaml": (
            "openlm-research/open_llama_7b",
            "hf_causal",
        ),
        "null_sweep_opt_1_3b.yaml": (
            "facebook/opt-1.3b",
            "hf_causal",
        ),
        "null_sweep_falcon_7b.yaml": (
            "tiiuae/falcon-7b",
            "auto",
        ),
        "null_sweep_glm4_9b_chat.yaml": (
            "THUDM/glm-4-9b-chat",
            "auto",
        ),
    }
    for name, (model_id, adapter) in configs.items():
        data = yaml.safe_load(
            (root / "configs/calibration" / name).read_text(encoding="utf-8")
        )
        assert data["model"]["id"] == model_id
        assert data["model"]["adapter"] == adapter
        assert data["dataset"]["provider"]["kind"] == "hf_text"
        assert data["primary_metric"]["drift_band"] == expected_drift_band

    glm_data = yaml.safe_load(
        (root / "configs/calibration" / "null_sweep_glm4_9b_chat.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert glm_data["model"]["trust_remote_code"] is True
