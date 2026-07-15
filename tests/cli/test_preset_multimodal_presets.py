from __future__ import annotations

from pathlib import Path

from invarlock.core.config_loader import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assert_large_gemma_memory_controls(model: dict[str, object]) -> None:
    assert model["dtype"] == "bfloat16"
    assert model["device_map"] == "auto"
    assert model["low_cpu_mem_usage"] is True
    assert model["collect_loading_info"] is False


def _assert_public_vqav2_image_text_config(
    *,
    rel_path: str,
    model_id: str,
    output_dir: str | None = None,
    requires_sdpa: bool = False,
    expected_windows: int = 400,
    expect_metric_impact_skip: bool = False,
) -> None:
    root = _repo_root()
    cfg = load_config(root / rel_path)

    model = cfg.require_section("model")
    dataset = cfg.require_section("dataset")
    eval_section = cfg.require_section("eval")
    guards = cfg.require_section("guards")

    assert model["id"] == model_id
    assert model["adapter"] == "hf_multimodal"
    _assert_large_gemma_memory_controls(model)
    if requires_sdpa:
        assert model["attn_implementation"] == "sdpa"
    assert dataset["provider"]["kind"] == "vision_text"
    assert dataset["provider"]["path"].endswith(
        "public_datasets/vqav2_sample_validation_800/manifest.jsonl"
    )
    assert dataset["preview_n"] == expected_windows
    assert dataset["final_n"] == expected_windows
    assert eval_section["metric"]["kind"] == "accuracy"
    assert eval_section["loss"]["type"] == "classification"
    assert cfg.require_section("primary_metric")["drift_band"] == {
        "min": 0.8,
        "max": 1.2,
    }
    context = cfg.data.get("context", {})
    skip_guard_metric_impact = context.get("run", {}).get(
        "skip_guard_metric_impact_check", False
    )
    assert skip_guard_metric_impact is expect_metric_impact_skip
    assert guards["order"] == [
        "invariants",
        "spectral",
        "rmt",
        "variance",
        "invariants",
    ]
    assert guards["spectral"]["module_include_patterns"]
    assert guards["rmt"]["module_include_patterns"]
    if output_dir is not None:
        assert str(cfg.require_section("output")["dir"]) == output_dir


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
    _assert_large_gemma_memory_controls(model)
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


def test_gemma4_12b_public_vqav2_preset_uses_materialized_manifest_path() -> None:
    root = _repo_root()
    preset_path = root / "configs/presets/multimodal/gemma4_12b_public_vqav2_256.yaml"
    cfg = load_config(preset_path)

    model = cfg.require_section("model")
    dataset = cfg.require_section("dataset")
    eval_section = cfg.require_section("eval")
    guards = cfg.require_section("guards")

    assert model["id"] == "google/gemma-4-12B-it"
    assert model["adapter"] == "hf_multimodal"
    assert model["attn_implementation"] == "sdpa"
    _assert_large_gemma_memory_controls(model)
    assert dataset["provider"]["kind"] == "vision_text"
    assert dataset["provider"]["path"].endswith(
        "public_datasets/vqav2_sample_validation_800/manifest.jsonl"
    )
    assert dataset["preview_n"] == 400
    assert dataset["final_n"] == 400
    assert eval_section["metric"]["kind"] == "accuracy"
    assert eval_section["loss"]["type"] == "classification"
    assert guards["order"] == [
        "invariants",
        "spectral",
        "rmt",
        "variance",
        "invariants",
    ]


def test_gemma4_26b_a4b_public_vqav2_preset_declares_moe_candidate() -> None:
    root = _repo_root()
    preset_path = (
        root / "configs/presets/multimodal/gemma4_26b_a4b_public_vqav2_256.yaml"
    )
    cfg = load_config(preset_path)

    model = cfg.require_section("model")
    dataset = cfg.require_section("dataset")
    eval_section = cfg.require_section("eval")
    guards = cfg.require_section("guards")

    assert model["id"] == "google/gemma-4-26B-A4B-it"
    assert model["adapter"] == "hf_multimodal"
    assert model["attn_implementation"] == "sdpa"
    _assert_large_gemma_memory_controls(model)
    assert dataset["provider"]["kind"] == "vision_text"
    assert dataset["provider"]["path"].endswith(
        "public_datasets/vqav2_sample_validation_800/manifest.jsonl"
    )
    assert dataset["preview_n"] == 400
    assert dataset["final_n"] == 400
    assert eval_section["metric"]["kind"] == "accuracy"
    assert eval_section["loss"]["type"] == "classification"
    assert cfg.data.get("context", {}) == {}
    assert guards["spectral"]["module_include_patterns"]
    assert guards["spectral"]["family_caps"]["router"] == 5.0
    assert guards["rmt"]["module_include_patterns"]


def test_small_multimodal_candidate_public_vqav2_presets_load() -> None:
    for rel_path, model_id, requires_sdpa in (
        (
            "configs/presets/multimodal/gemma4_e4b_public_vqav2_256.yaml",
            "google/gemma-4-E4B-it",
            True,
        ),
        (
            "configs/presets/multimodal/gemma4_e2b_public_vqav2_256.yaml",
            "google/gemma-4-E2B-it",
            True,
        ),
        (
            "configs/presets/multimodal/qwen3_5_4b_public_vqav2_256.yaml",
            "Qwen/Qwen3.5-4B",
            False,
        ),
        (
            "configs/presets/multimodal/qwen3_5_2b_public_vqav2_256.yaml",
            "Qwen/Qwen3.5-2B",
            False,
        ),
    ):
        _assert_public_vqav2_image_text_config(
            rel_path=rel_path,
            model_id=model_id,
            requires_sdpa=requires_sdpa,
        )


def test_gemma4_12b_null_sweep_calibration_config_uses_public_manifest() -> None:
    root = _repo_root()
    cfg = load_config(root / "configs/calibration/null_sweep_gemma4_12b.yaml")

    model = cfg.require_section("model")
    dataset = cfg.require_section("dataset")
    eval_section = cfg.require_section("eval")
    guards = cfg.require_section("guards")

    assert model["id"] == "google/gemma-4-12B-it"
    assert model["adapter"] == "hf_multimodal"
    assert model["attn_implementation"] == "sdpa"
    _assert_large_gemma_memory_controls(model)
    assert dataset["provider"]["kind"] == "vision_text"
    assert dataset["provider"]["path"].endswith(
        "public_datasets/vqav2_sample_validation_800/manifest.jsonl"
    )
    assert dataset["preview_n"] == 16
    assert dataset["final_n"] == 16
    assert eval_section["metric"]["kind"] == "accuracy"
    assert eval_section["loss"]["type"] == "classification"
    assert cfg.require_section("primary_metric")["drift_band"] == {
        "min": 0.8,
        "max": 1.2,
    }
    assert (
        cfg.require_section("context")["run"]["skip_guard_metric_impact_check"] is True
    )
    assert guards["spectral"]["module_include_patterns"]
    assert guards["rmt"]["module_include_patterns"]


def test_gemma4_26b_a4b_null_sweep_calibration_config_uses_public_manifest() -> None:
    root = _repo_root()
    cfg = load_config(root / "configs/calibration/null_sweep_gemma4_26b_a4b.yaml")

    model = cfg.require_section("model")
    dataset = cfg.require_section("dataset")
    eval_section = cfg.require_section("eval")
    guards = cfg.require_section("guards")

    assert model["id"] == "google/gemma-4-26B-A4B-it"
    assert model["adapter"] == "hf_multimodal"
    assert model["attn_implementation"] == "sdpa"
    _assert_large_gemma_memory_controls(model)
    assert dataset["provider"]["kind"] == "vision_text"
    assert dataset["provider"]["path"].endswith(
        "public_datasets/vqav2_sample_validation_800/manifest.jsonl"
    )
    assert dataset["preview_n"] == 16
    assert dataset["final_n"] == 16
    assert eval_section["metric"]["kind"] == "accuracy"
    assert eval_section["loss"]["type"] == "classification"
    assert cfg.require_section("primary_metric")["drift_band"] == {
        "min": 0.8,
        "max": 1.2,
    }
    assert (
        cfg.require_section("context")["run"]["skip_guard_metric_impact_check"] is True
    )
    assert guards["spectral"]["module_include_patterns"]
    assert guards["spectral"]["family_caps"]["router"] == 5.0
    assert guards["rmt"]["module_include_patterns"]


def test_small_multimodal_candidate_null_sweeps_use_public_manifest() -> None:
    for rel_path, model_id, output_dir, requires_sdpa in (
        (
            "configs/calibration/null_sweep_gemma4_e4b.yaml",
            "google/gemma-4-E4B-it",
            "runs/null_sweeps/gemma4_e4b",
            True,
        ),
        (
            "configs/calibration/null_sweep_gemma4_e2b_image_text.yaml",
            "google/gemma-4-E2B-it",
            "runs/null_sweeps/gemma4_e2b_image_text",
            True,
        ),
        (
            "configs/calibration/null_sweep_qwen3_5_4b.yaml",
            "Qwen/Qwen3.5-4B",
            "runs/null_sweeps/qwen3_5_4b",
            False,
        ),
        (
            "configs/calibration/null_sweep_qwen3_5_2b.yaml",
            "Qwen/Qwen3.5-2B",
            "runs/null_sweeps/qwen3_5_2b",
            False,
        ),
    ):
        _assert_public_vqav2_image_text_config(
            rel_path=rel_path,
            model_id=model_id,
            output_dir=output_dir,
            requires_sdpa=requires_sdpa,
            expected_windows=16,
            expect_metric_impact_skip=True,
        )
