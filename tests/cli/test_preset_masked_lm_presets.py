from __future__ import annotations

from pathlib import Path

import yaml

from invarlock.core.config_loader import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_candidate_masked_lm_preset_loads() -> None:
    root = _repo_root()
    cfg = load_config(
        root / "configs/presets/masked_lm/distilbert_base_uncased_128.yaml"
    )

    assert cfg.require_section("model")["id"] == "distilbert-base-uncased"
    assert cfg.require_section("model")["adapter"] == "hf_mlm"
    assert cfg.data["dataset"]["provider"] == "wikitext2"
    assert cfg.data["primary_metric"]["drift_band"] == {"min": 0.9, "max": 1.2}


def test_candidate_masked_lm_null_sweep_config_references_model() -> None:
    root = _repo_root()
    data = yaml.safe_load(
        (
            root / "configs/calibration" / "null_sweep_distilbert_base_uncased.yaml"
        ).read_text(encoding="utf-8")
    )

    assert data["model"]["id"] == "distilbert-base-uncased"
    assert data["model"]["adapter"] == "hf_mlm"
    assert data["dataset"]["provider"] == "wikitext2"
    assert data["primary_metric"]["drift_band"] == {"min": 0.9, "max": 1.2}
