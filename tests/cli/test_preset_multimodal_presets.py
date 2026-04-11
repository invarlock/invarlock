from __future__ import annotations

from pathlib import Path

from invarlock.core.config_loader import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_multimodal_preset_loads_and_points_at_demo_fixture() -> None:
    root = _repo_root()
    cfg = load_config(
        root / "configs/presets/multimodal/gemma4_e2b_vision_text_256.yaml"
    )

    model = cfg.require_section("model")
    dataset = cfg.require_section("dataset")
    eval_section = cfg.require_section("eval")

    assert model["id"] == "google/gemma-4-E2B-it"
    assert model["adapter"] == "hf_multimodal"
    assert model["attn_implementation"] == "sdpa"
    assert dataset["provider"]["kind"] == "vision_text"
    assert (
        dataset["provider"]["path"] == "tests/fixtures/vision_text/demo_manifest.jsonl"
    )
    assert eval_section["metric"]["kind"] == "accuracy"
    assert eval_section["loss"]["type"] == "classification"

    fixture = root / dataset["provider"]["path"]
    assert fixture.is_file()
