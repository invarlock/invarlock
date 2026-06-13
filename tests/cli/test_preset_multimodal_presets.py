from __future__ import annotations

from pathlib import Path

from invarlock.core.config_loader import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_multimodal_preset_loads_and_points_at_demo_fixture() -> None:
    root = _repo_root()
    preset_path = root / "configs/presets/multimodal/gemma4_e2b_vision_text_256.yaml"
    cfg = load_config(preset_path)

    model = cfg.require_section("model")
    dataset = cfg.require_section("dataset")
    eval_section = cfg.require_section("eval")
    guards = cfg.require_section("guards")

    assert model["id"] == "google/gemma-4-E2B-it"
    assert model["adapter"] == "hf_multimodal"
    assert model["attn_implementation"] == "sdpa"
    assert dataset["provider"]["kind"] == "vision_text"
    assert (
        dataset["provider"]["path"] == "tests/fixtures/vision_text/demo_manifest.jsonl"
    )
    assert eval_section["metric"]["kind"] == "accuracy"
    assert eval_section["loss"]["type"] == "classification"
    assert guards["order"] == [
        "invariants",
        "spectral",
        "rmt",
        "variance",
        "invariants",
    ]
    assert guards["spectral"]["module_include_patterns"]
    assert guards["rmt"]["module_include_patterns"]

    fixture = root / dataset["provider"]["path"]
    assert fixture.is_file()


def test_gemma4_12b_multimodal_preset_declares_unified_candidate() -> None:
    root = _repo_root()
    preset_path = root / "configs/presets/multimodal/gemma4_12b_vision_text_256.yaml"
    cfg = load_config(preset_path)

    model = cfg.require_section("model")
    dataset = cfg.require_section("dataset")
    eval_section = cfg.require_section("eval")
    guards = cfg.require_section("guards")

    assert model["id"] == "google/gemma-4-12B-it"
    assert model["adapter"] == "hf_multimodal"
    assert model["attn_implementation"] == "sdpa"
    assert dataset["provider"]["kind"] == "vision_text"
    assert (
        dataset["provider"]["path"] == "tests/fixtures/vision_text/demo_manifest.jsonl"
    )
    assert eval_section["metric"]["kind"] == "accuracy"
    assert eval_section["loss"]["type"] == "classification"
    assert guards["order"] == [
        "invariants",
        "spectral",
        "rmt",
        "variance",
        "invariants",
    ]
    assert guards["spectral"]["module_include_patterns"]
    assert guards["rmt"]["module_include_patterns"]

    fixture = root / dataset["provider"]["path"]
    assert fixture.is_file()
