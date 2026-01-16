from __future__ import annotations

from pathlib import Path

import yaml

from invarlock.cli.config import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_causal_lm_family_presets_load() -> None:
    root = _repo_root()
    presets = {
        "llama3_8b_512.yaml": "meta-llama/Meta-Llama-3-8B",
        "mistral_7b_512.yaml": "mistralai/Mistral-7B-v0.1",
        "qwen2_7b_512.yaml": "Qwen/Qwen2-7B",
    }
    for name, model_id in presets.items():
        cfg = load_config(root / "configs/presets/causal_lm" / name)
        assert cfg.model.id == model_id
        assert cfg.model.adapter == "hf_llama"


def test_null_sweep_calibration_configs_reference_models() -> None:
    root = _repo_root()
    configs = {
        "null_sweep_llama3_8b.yaml": "meta-llama/Meta-Llama-3-8B",
        "null_sweep_mistral_7b.yaml": "mistralai/Mistral-7B-v0.1",
        "null_sweep_qwen2_7b.yaml": "Qwen/Qwen2-7B",
    }
    for name, model_id in configs.items():
        data = yaml.safe_load(
            (root / "configs/calibration" / name).read_text(encoding="utf-8")
        )
        assert data["model"]["id"] == model_id
