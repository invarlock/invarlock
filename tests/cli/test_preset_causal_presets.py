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
        "open_llama_7b_512.yaml": "openlm-research/open_llama_7b",
        "opt_1_3b_512.yaml": "facebook/opt-1.3b",
        "falcon_7b_512.yaml": "tiiuae/falcon-7b",
        "glm4_9b_chat_512.yaml": "THUDM/glm-4-9b-chat",
        "ministral3_8b_512.yaml": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "ministral3_14b_512.yaml": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "qwen2_7b_512.yaml": "Qwen/Qwen2-7B",
        "qwen2_5_7b_512.yaml": "Qwen/Qwen2.5-7B",
        "qwen2_5_14b_512.yaml": "Qwen/Qwen2.5-14B",
        "qwen3_8b_512.yaml": "Qwen/Qwen3-8B",
        "qwen3_30b_a3b_instruct_2507_512.yaml": (
            "Qwen/Qwen3-30B-A3B-Instruct-2507"
        ),
        "olmoe_1b_7b_0924_512.yaml": "allenai/OLMoE-1B-7B-0924",
        "deepseek_r1_distill_qwen_7b_512.yaml": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "phi4_reasoning_plus_512.yaml": "microsoft/Phi-4-reasoning-plus",
        "gemma4_e2b_512.yaml": "google/gemma-4-E2B-it",
        "tinyllama_1_1b_512.yaml": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "olmo2_7b_512.yaml": "allenai/OLMo-2-1124-7B",
        "olmo2_13b_512.yaml": "allenai/OLMo-2-1124-13B-Instruct",
        "qwen3_5_9b_512.yaml": "Qwen/Qwen3.5-9B",
        "ministral3_3b_512.yaml": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        "granite4_1_8b_512.yaml": "ibm-granite/granite-4.1-8b",
        "granite4_1_3b_512.yaml": "ibm-granite/granite-4.1-3b",
        "smollm3_3b_512.yaml": "HuggingFaceTB/SmolLM3-3B",
        "phi4_mini_512.yaml": "microsoft/Phi-4-mini-instruct",
        "deepseek_r1_distill_qwen_14b_512.yaml": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        ),
        "deepseek_r1_0528_qwen3_8b_512.yaml": ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"),
        "falcon_h1r_7b_512.yaml": "tiiuae/Falcon-H1R-7B",
    }
    expected_provider_kinds = {
        "deepseek_r1_distill_qwen_7b_512.yaml": "hf_text",
        "deepseek_r1_distill_qwen_14b_512.yaml": "hf_text",
        "deepseek_r1_0528_qwen3_8b_512.yaml": "hf_text",
        "falcon_h1r_7b_512.yaml": "hf_text",
        "gemma4_e2b_512.yaml": "hf_text",
        "granite4_1_3b_512.yaml": "hf_text",
        "granite4_1_8b_512.yaml": "hf_text",
        "ministral3_3b_512.yaml": "hf_text",
        "ministral3_8b_512.yaml": "hf_text",
        "ministral3_14b_512.yaml": "hf_text",
        "olmo2_13b_512.yaml": "hf_text",
        "olmo2_7b_512.yaml": "hf_text",
        "olmoe_1b_7b_0924_512.yaml": "hf_text",
        "phi4_mini_512.yaml": "hf_text",
        "phi4_reasoning_plus_512.yaml": "hf_text",
        "qwen2_5_7b_512.yaml": "hf_text",
        "qwen2_5_14b_512.yaml": "hf_text",
        "qwen2_7b_512.yaml": "hf_text",
        "qwen3_30b_a3b_instruct_2507_512.yaml": "hf_text",
        "qwen3_5_9b_512.yaml": "hf_text",
        "qwen3_8b_512.yaml": "hf_text",
        "smollm3_3b_512.yaml": "hf_text",
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
        if name == "qwen3_30b_a3b_instruct_2507_512.yaml":
            model = cfg.require_section("model")
            assert model["dtype"] == "bfloat16"
            assert model["device_map"] == "auto"
            assert model["low_cpu_mem_usage"] is True
            assert model["collect_loading_info"] is False
            guards = cfg.require_section("guards")
            for guard_name in ("spectral", "rmt"):
                guard_cfg = guards[guard_name]
                assert "model.layers.*.self_attn.*_proj" in guard_cfg[
                    "module_include_patterns"
                ]
                assert "model.layers.*.mlp.gate" in guard_cfg[
                    "module_include_patterns"
                ]
                assert "model.layers.*.mlp.shared_expert*" in guard_cfg[
                    "module_include_patterns"
                ]
                assert guard_cfg["module_exclude_patterns"] == [
                    "model.layers.*.mlp.experts.*"
                ]
        if name == "olmoe_1b_7b_0924_512.yaml":
            model = cfg.require_section("model")
            assert model["dtype"] == "bfloat16"
            assert model["low_cpu_mem_usage"] is True
            assert model["collect_loading_info"] is False
            guards = cfg.require_section("guards")
            assert "spectral" not in guards
            assert "rmt" not in guards
        if name in {
            "glm4_9b_chat_512.yaml",
            "phi4_reasoning_plus_512.yaml",
        }:
            assert cfg.require_section("model")["trust_remote_code"] is True
        if name == "phi4_mini_512.yaml":
            assert "trust_remote_code" not in cfg.require_section("model")
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
        "null_sweep_open_llama_7b.yaml": "openlm-research/open_llama_7b",
        "null_sweep_opt_1_3b.yaml": "facebook/opt-1.3b",
        "null_sweep_falcon_7b.yaml": "tiiuae/falcon-7b",
        "null_sweep_glm4_9b_chat.yaml": "THUDM/glm-4-9b-chat",
        "null_sweep_ministral3_8b.yaml": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "null_sweep_ministral3_14b.yaml": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "null_sweep_qwen2_7b.yaml": "Qwen/Qwen2-7B",
        "null_sweep_qwen2_5_7b.yaml": "Qwen/Qwen2.5-7B",
        "null_sweep_qwen2_5_14b.yaml": "Qwen/Qwen2.5-14B",
        "null_sweep_qwen3_8b.yaml": "Qwen/Qwen3-8B",
        "null_sweep_qwen3_30b_a3b_instruct_2507.yaml": (
            "Qwen/Qwen3-30B-A3B-Instruct-2507"
        ),
        "null_sweep_olmoe_1b_7b_0924.yaml": "allenai/OLMoE-1B-7B-0924",
        "null_sweep_deepseek_r1_distill_qwen_7b.yaml": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "null_sweep_phi4_reasoning_plus.yaml": "microsoft/Phi-4-reasoning-plus",
        "null_sweep_gemma4_e2b.yaml": "google/gemma-4-E2B-it",
        "null_sweep_tinyllama_1_1b.yaml": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "null_sweep_olmo2_7b.yaml": "allenai/OLMo-2-1124-7B",
        "null_sweep_olmo2_13b.yaml": "allenai/OLMo-2-1124-13B-Instruct",
        "null_sweep_qwen3_5_9b.yaml": "Qwen/Qwen3.5-9B",
        "null_sweep_ministral3_3b.yaml": (
            "mistralai/Ministral-3-3B-Instruct-2512-BF16"
        ),
        "null_sweep_granite4_1_8b.yaml": "ibm-granite/granite-4.1-8b",
        "null_sweep_granite4_1_3b.yaml": "ibm-granite/granite-4.1-3b",
        "null_sweep_smollm3_3b.yaml": "HuggingFaceTB/SmolLM3-3B",
        "null_sweep_phi4_mini.yaml": "microsoft/Phi-4-mini-instruct",
        "null_sweep_deepseek_r1_distill_qwen_14b.yaml": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        ),
        "null_sweep_deepseek_r1_0528_qwen3_8b.yaml": (
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        ),
        "null_sweep_falcon_h1r_7b.yaml": "tiiuae/Falcon-H1R-7B",
    }
    for name, model_id in configs.items():
        data = yaml.safe_load(
            (root / "configs/calibration" / name).read_text(encoding="utf-8")
        )
        assert data["model"]["id"] == model_id
        if name in {
            "null_sweep_glm4_9b_chat.yaml",
            "null_sweep_phi4_reasoning_plus.yaml",
        }:
            assert data["model"]["trust_remote_code"] is True
        if name == "null_sweep_phi4_mini.yaml":
            assert "trust_remote_code" not in data["model"]
        if name in {
            "null_sweep_deepseek_r1_distill_qwen_7b.yaml",
            "null_sweep_deepseek_r1_distill_qwen_14b.yaml",
            "null_sweep_deepseek_r1_0528_qwen3_8b.yaml",
            "null_sweep_falcon_h1r_7b.yaml",
            "null_sweep_gemma4_e2b.yaml",
            "null_sweep_granite4_1_3b.yaml",
            "null_sweep_granite4_1_8b.yaml",
            "null_sweep_ministral3_3b.yaml",
            "null_sweep_ministral3_8b.yaml",
            "null_sweep_ministral3_14b.yaml",
            "null_sweep_olmo2_13b.yaml",
            "null_sweep_olmo2_7b.yaml",
            "null_sweep_olmoe_1b_7b_0924.yaml",
            "null_sweep_phi4_mini.yaml",
            "null_sweep_phi4_reasoning_plus.yaml",
            "null_sweep_qwen2_5_7b.yaml",
            "null_sweep_qwen2_5_14b.yaml",
            "null_sweep_qwen2_7b.yaml",
            "null_sweep_qwen3_30b_a3b_instruct_2507.yaml",
            "null_sweep_qwen3_5_9b.yaml",
            "null_sweep_qwen3_8b.yaml",
            "null_sweep_smollm3_3b.yaml",
        }:
            assert data["dataset"]["provider"]["kind"] == "hf_text"
        if name == "null_sweep_gemma4_e2b.yaml":
            assert data["model"]["attn_implementation"] == "sdpa"
        if name == "null_sweep_qwen3_30b_a3b_instruct_2507.yaml":
            assert data["model"]["dtype"] == "bfloat16"
            assert data["model"]["device_map"] == "auto"
            assert data["model"]["low_cpu_mem_usage"] is True
            assert data["model"]["collect_loading_info"] is False
            for guard_name in ("spectral", "rmt"):
                guard_cfg = data["guards"][guard_name]
                assert "model.layers.*.self_attn.*_proj" in guard_cfg[
                    "module_include_patterns"
                ]
                assert "model.layers.*.mlp.gate" in guard_cfg[
                    "module_include_patterns"
                ]
                assert "model.layers.*.mlp.shared_expert*" in guard_cfg[
                    "module_include_patterns"
                ]
                assert guard_cfg["module_exclude_patterns"] == [
                    "model.layers.*.mlp.experts.*"
                ]
        if name == "null_sweep_olmoe_1b_7b_0924.yaml":
            assert data["model"]["dtype"] == "bfloat16"
            assert data["model"]["low_cpu_mem_usage"] is True
            assert data["model"]["collect_loading_info"] is False
            assert "spectral" not in data["guards"]
            assert "rmt" not in data["guards"]
        assert data["primary_metric"]["drift_band"] == expected_drift_band


def test_candidate_causal_lm_presets_load() -> None:
    root = _repo_root()
    expected_drift_band = {"min": 0.9, "max": 1.2}
    presets = {
        "open_llama_7b_512.yaml": (
            "openlm-research/open_llama_7b",
            "hf_causal",
        ),
        "opt_1_3b_512.yaml": (
            "facebook/opt-1.3b",
            "hf_causal",
        ),
        "falcon_7b_512.yaml": (
            "tiiuae/falcon-7b",
            "hf_causal",
        ),
        "glm4_9b_chat_512.yaml": (
            "THUDM/glm-4-9b-chat",
            "hf_causal",
        ),
    }
    for name, (model_id, adapter) in presets.items():
        cfg = load_config(root / "configs/presets/causal_lm" / name)
        assert cfg.require_section("model")["id"] == model_id
        assert cfg.require_section("model")["adapter"] == adapter
        if name == "glm4_9b_chat_512.yaml":
            assert cfg.require_section("model")["trust_remote_code"] is True
        assert cfg.data["dataset"]["provider"] == "wikitext2"
        assert cfg.data["primary_metric"]["drift_band"] == expected_drift_band


def test_candidate_null_sweep_calibration_configs_reference_models() -> None:
    root = _repo_root()
    expected_drift_band = {"min": 0.9, "max": 1.2}
    configs = {
        "null_sweep_open_llama_7b.yaml": (
            "openlm-research/open_llama_7b",
            "hf_causal",
        ),
        "null_sweep_opt_1_3b.yaml": (
            "facebook/opt-1.3b",
            "hf_causal",
        ),
        "null_sweep_falcon_7b.yaml": (
            "tiiuae/falcon-7b",
            "hf_causal",
        ),
        "null_sweep_glm4_9b_chat.yaml": (
            "THUDM/glm-4-9b-chat",
            "hf_causal",
        ),
    }
    for name, (model_id, adapter) in configs.items():
        data = yaml.safe_load(
            (root / "configs/calibration" / name).read_text(encoding="utf-8")
        )
        assert data["model"]["id"] == model_id
        assert data["model"]["adapter"] == adapter
        if name == "null_sweep_glm4_9b_chat.yaml":
            assert data["model"]["trust_remote_code"] is True
        assert data["dataset"]["provider"] == "wikitext2"
        assert data["primary_metric"]["drift_band"] == expected_drift_band
