from __future__ import annotations

from pathlib import Path

import yaml

from invarlock.core.config_loader import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_masked_lm_family_presets_load() -> None:
    root = _repo_root()
    expected_drift_band = {"min": 0.9, "max": 1.2}
    presets = {
        "wikitext2_128.yaml": "prajjwal1/bert-tiny",
        "distilbert_base_uncased_128.yaml": "distilbert-base-uncased",
    }
    for name, model_id in presets.items():
        cfg = load_config(root / "configs/presets/masked_lm" / name)
        assert cfg.require_section("model")["id"] == model_id
        assert cfg.require_section("model")["adapter"] == "hf_mlm"
        assert cfg.require_section("eval")["metric"]["kind"] == "ppl_mlm"
        assert cfg.require_section("eval")["loss"]["type"] == "mlm"
        if name != "wikitext2_128.yaml":
            assert cfg.data["primary_metric"]["drift_band"] == expected_drift_band


def test_null_sweep_masked_lm_calibration_configs_reference_models() -> None:
    root = _repo_root()
    data = yaml.safe_load(
        (
            root / "configs/calibration/null_sweep_distilbert_base_uncased.yaml"
        ).read_text(encoding="utf-8")
    )
    assert data["model"]["id"] == "distilbert-base-uncased"
    assert data["model"]["adapter"] == "hf_mlm"
    assert data["eval"]["metric"]["kind"] == "ppl_mlm"
    assert data["eval"]["loss"]["type"] == "mlm"
    assert data["primary_metric"]["drift_band"] == {"min": 0.9, "max": 1.2}
