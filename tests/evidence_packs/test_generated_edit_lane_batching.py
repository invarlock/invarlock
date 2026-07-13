from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.evidence_packs.python import create_edits_batch as batch_edit_mod
from scripts.evidence_packs.python.editing.implementations import (
    build_validation_edit_metadata,
    validate_edit_metadata,
)
from scripts.evidence_packs.python.editing.streaming_transform import (
    replay_transformation_tensor,
)


def test_batch_edit_dir_name_covers_known_edit_families() -> None:
    quant = batch_edit_mod._get_edit_dir_name(
        {
            "type": "quant_rtn",
            "bits": 4,
            "group_size": 32,
            "scope": "ffn",
        },
        "clean",
    )
    assert quant.startswith("generated--quant_rtn--bits-4--group_size-32--scope-ffn")
    assert "--version-clean--sha256-" in quant
    assert (
        batch_edit_mod._get_edit_dir_name(
            {"type": "magnitude_prune", "ratio": 0.25},
            "stress",
        )
        == "prune_25pct_stress"
    )
    for unsupported in ("fp8_quant", "lowrank_svd"):
        with pytest.raises(ValueError, match="dedicated storage and replay contract"):
            batch_edit_mod._get_edit_dir_name({"type": unsupported}, "clean")
    lowrank = batch_edit_mod._get_edit_dir_name(
        {
            "type": "synthetic_lowrank_delta",
            "rank": 4,
            "scale": 8.0,
            "scope": "attn",
        },
        "clean",
    )
    assert lowrank.startswith(
        "generated--synthetic_lowrank_delta--rank-4--scale-8--scope-attn"
    )
    dense = batch_edit_mod._get_edit_dir_name(
        {
            "type": "synthetic_dense_update",
            "step_size": 0.0001,
            "iterations": 3,
            "scope": "ffn",
        },
        "stress",
    )
    assert dense.startswith(
        "generated--synthetic_dense_update--iterations-3--step_size-0.0001--scope-ffn"
    )
    assert (
        batch_edit_mod._get_edit_dir_name(
            {"type": "custom_family"},
            "clean",
        )
        == "custom_family_clean"
    )
    with pytest.raises(ValueError, match="canonical directory identity"):
        batch_edit_mod._get_edit_dir_name(
            {
                "type": "quant_rtn",
                "bits": 4,
                "group_size": 32,
                "scope": "ffn",
                "edit_dir_name": "quant_4bit_clean",
            },
            "clean",
        )


def test_batch_directory_fallback_rejects_raw_transform_path_collisions() -> None:
    common = {"type": "quant_rtn", "bits": 4, "scope": "ffn"}
    group_32 = batch_edit_mod._get_edit_dir_name({**common, "group_size": 32}, "stress")
    group_64 = batch_edit_mod._get_edit_dir_name({**common, "group_size": 64}, "stress")
    attn = batch_edit_mod._get_edit_dir_name(
        {**common, "group_size": 32, "scope": "attn"}, "stress"
    )

    assert len({group_32, group_64, attn}) == 3


def test_batch_determinism_modes(monkeypatch) -> None:
    previous_grad = torch.is_grad_enabled()
    previous_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        monkeypatch.setenv("PACK_DETERMINISM", "strict")
        batch_edit_mod._configure_determinism()
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.is_grad_enabled() is False

        monkeypatch.setenv("PACK_DETERMINISM", "throughput")
        batch_edit_mod._configure_determinism()
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.set_grad_enabled(previous_grad)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32_matmul
        torch.backends.cudnn.allow_tf32 = previous_tf32_cudnn


def test_batch_resolve_pending_spec_entry_paths(monkeypatch, tmp_path: Path) -> None:
    model_output_dir = tmp_path / "model"
    model_output_dir.mkdir()

    class Resolved:
        def __init__(
            self,
            *,
            skip: bool = False,
            selected: bool = True,
            status: str = "selected",
        ) -> None:
            self.skip = skip
            self.selected = selected
            self.status = status

        def to_batch_payload(self) -> dict[str, object]:
            return {
                "type": "synthetic_lowrank_delta",
                "rank": 4,
                "scale": 8,
                "scope": "attn",
            }

    assert batch_edit_mod._resolve_pending_spec_entry(
        spec_entry="not-a-dict",
        model_output_dir=model_output_dir,
    ) == (None, 0, 0)

    monkeypatch.setattr(batch_edit_mod, "resolve_batch_entry", lambda **_kwargs: None)
    assert batch_edit_mod._resolve_pending_spec_entry(
        spec_entry={"spec": "synthetic_lowrank_delta:clean:attn"},
        model_output_dir=model_output_dir,
    ) == (None, 0, 0)

    monkeypatch.setattr(
        batch_edit_mod,
        "resolve_batch_entry",
        lambda **_kwargs: Resolved(skip=True),
    )
    assert batch_edit_mod._resolve_pending_spec_entry(
        spec_entry={"spec": "synthetic_lowrank_delta:clean:attn"},
        model_output_dir=model_output_dir,
    ) == (None, 0, 0)

    monkeypatch.setattr(
        batch_edit_mod,
        "resolve_batch_entry",
        lambda **_kwargs: Resolved(selected=False, status="missing"),
    )
    with pytest.raises(ValueError, match="Tuned edit preset missing"):
        batch_edit_mod._resolve_pending_spec_entry(
            spec_entry={"spec": "synthetic_lowrank_delta:clean:attn"},
            model_output_dir=model_output_dir,
        )

    monkeypatch.setattr(
        batch_edit_mod,
        "resolve_batch_entry",
        lambda **_kwargs: Resolved(),
    )
    with pytest.raises(ValueError, match="invalid clean-selection artifact directory"):
        batch_edit_mod._resolve_pending_spec_entry(
            spec_entry={
                "spec": "synthetic_lowrank_delta:4:8:attn",
                "version": "clean",
                "selection_edit_dir_name": "clean_synthetic_lowrank_delta",
            },
            model_output_dir=model_output_dir,
        )

    with pytest.raises(ValueError, match="retired v2 clean-selection artifact"):
        batch_edit_mod._resolve_pending_spec_entry(
            spec_entry={
                "spec": "synthetic_lowrank_delta:4:8:attn",
                "version": "clean",
                "v2_selection_edit_dir_name": "clean_synthetic_lowrank_delta",
            },
            model_output_dir=model_output_dir,
        )

    expected_dir = batch_edit_mod._get_edit_dir_name(
        {
            "type": "synthetic_lowrank_delta",
            "rank": 4,
            "scale": 8,
            "scope": "attn",
        },
        "clean",
    )
    occupied_path = model_output_dir / "models" / expected_dir
    occupied_path.mkdir(parents=True)
    # A copied baseline can carry plausible-looking sidecars.  Batch creation
    # must not count or reuse it before exact replay validation.
    for name in (
        "config.json",
        "edit_metadata.json",
        "transformation_materialization.json",
    ):
        (occupied_path / name).write_text("{}\n", encoding="utf-8")
    (occupied_path / "model.safetensors").write_bytes(b"forged")
    with pytest.raises(ValueError, match="refusing final artifact reuse"):
        batch_edit_mod._resolve_pending_spec_entry(
            spec_entry={"spec": "synthetic_lowrank_delta:clean:attn"},
            model_output_dir=model_output_dir,
        )

    for child in occupied_path.iterdir():
        child.unlink()
    occupied_path.rmdir()
    pending, created, failed = batch_edit_mod._resolve_pending_spec_entry(
        spec_entry={"spec": "synthetic_lowrank_delta:clean:attn", "version": "clean"},
        model_output_dir=model_output_dir,
    )
    assert created == 0
    assert failed == 0
    assert pending is not None
    parsed, edit_path = pending
    assert parsed["type"] == "synthetic_lowrank_delta"
    assert edit_path == model_output_dir / "models" / expected_dir


def test_batch_process_spec_counts_successes_and_failures(
    monkeypatch, tmp_path: Path
) -> None:
    model_output_dir = tmp_path / "model"
    model_output_dir.mkdir()
    calls: list[Path] = []
    expected_dir = batch_edit_mod._get_edit_dir_name(
        {
            "type": "synthetic_lowrank_delta",
            "rank": 4,
            "scale": 8,
            "scope": "attn",
        },
        "clean",
    )

    pending = (
        {"type": "synthetic_lowrank_delta", "rank": 4, "scale": 8, "scope": "attn"},
        model_output_dir / "models" / expected_dir,
    )
    monkeypatch.setattr(
        batch_edit_mod,
        "_resolve_pending_spec_entry",
        lambda **_kwargs: (pending, 0, 0),
    )
    monkeypatch.setattr(
        batch_edit_mod,
        "_create_streaming_transformation_artifact",
        lambda **kwargs: calls.append(kwargs["edit_path"]),
    )

    assert batch_edit_mod._process_spec_entry(
        spec_entry={"spec": "synthetic_lowrank_delta:clean:attn"},
        model_output_dir=model_output_dir,
        baseline_path=tmp_path / "baseline",
    ) == (1, 0)
    assert calls == [model_output_dir / "models" / expected_dir]

    monkeypatch.setattr(
        batch_edit_mod,
        "_create_streaming_transformation_artifact",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert batch_edit_mod._process_spec_entry(
        spec_entry={"spec": "synthetic_lowrank_delta:clean:attn"},
        model_output_dir=model_output_dir,
        baseline_path=tmp_path / "baseline",
    ) == (0, 1)

    sequence = iter(
        [
            (None, 1, 0),
            (pending, 0, 0),
            (pending, 0, 0),
        ]
    )
    monkeypatch.setattr(
        batch_edit_mod,
        "_resolve_pending_spec_entry",
        lambda **_kwargs: next(sequence),
    )
    outcomes = iter([None, RuntimeError("failed")])

    def _create_or_fail(**_kwargs: object) -> None:
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome

    monkeypatch.setattr(
        batch_edit_mod,
        "_create_streaming_transformation_artifact",
        _create_or_fail,
    )
    monkeypatch.setattr(batch_edit_mod, "_clear_memory", lambda: None)

    assert batch_edit_mod._process_edit_specs(
        edit_specs=[{"spec": "a"}, {"spec": "b"}, {"spec": "c"}],
        baseline_path=tmp_path / "baseline",
        model_output_dir=model_output_dir,
    ) == (2, 1)


def test_streaming_replay_is_parameter_bound_and_pure() -> None:
    source = torch.ones(8, 8, dtype=torch.float32)
    low_scale = replay_transformation_tensor(
        source,
        edit_type="synthetic_lowrank_delta",
        parameters={"rank": 2, "scale": 2.0},
    )
    high_scale = replay_transformation_tensor(
        source,
        edit_type="synthetic_lowrank_delta",
        parameters={"rank": 2, "scale": 8.0},
    )
    one_step = replay_transformation_tensor(
        source,
        edit_type="synthetic_dense_update",
        parameters={"step_size": 0.0001, "iterations": 1},
    )
    three_steps = replay_transformation_tensor(
        source,
        edit_type="synthetic_dense_update",
        parameters={"step_size": 0.0001, "iterations": 3},
    )

    assert torch.equal(source, torch.ones_like(source))
    assert (high_scale - source).norm() > (low_scale - source).norm()
    assert (three_steps - source).norm() > (one_step - source).norm()


def test_synthetic_lowrank_and_dense_update_metadata_helpers_validate() -> None:
    stats = type(
        "Stats",
        (),
        {
            "edited_tensors": 1,
            "edited_params": 4,
            "total_params": 8,
            "details": {},
            "coverage_payload": lambda self: {
                "edited_tensors": 1,
                "edited_params": 4,
                "total_params": 8,
            },
        },
    )()

    lora_metadata = build_validation_edit_metadata(
        edit_type="synthetic_lowrank_delta",
        scope="attn",
        parameters={"rank": 4, "scale": 8.0},
        coverage=stats.coverage_payload(),
        edit_provenance={
            "edit_family": "synthetic_lowrank_delta",
            "edit_method": "deterministic_synthetic_lowrank_delta",
            "edit_count": 1,
            "dynamic_runtime_required": False,
            "synthetic": True,
            "trained_adapter": False,
            "adapter_merge_performed": False,
        },
        edit_impact={
            "scenario_types": [
                "target_success",
                "near_neighbor",
                "unrelated_locality",
                "general_ability_sentinel",
            ]
        },
        extra={
            "edit_topology": {
                "artifact_kind": "checkpoint",
                "runtime_activation_policy": "static_subject_checkpoint",
                "training_or_edit_data_ref": "none-synthetic-generator",
            },
            "delta_privacy": {
                "delta_available": "hash_only",
                "privacy_sensitivity": "public",
                "public_raw_delta_approved": False,
            },
        },
    )
    synthetic_dense_update_metadata = build_validation_edit_metadata(
        edit_type="synthetic_dense_update",
        scope="ffn",
        parameters={"step_size": 0.0001, "iterations": 1},
        coverage=stats.coverage_payload(),
        edit_provenance={
            "edit_family": "synthetic_dense_update",
            "edit_method": "deterministic_synthetic_dense_update",
            "edit_count": 1,
            "dynamic_runtime_required": False,
            "synthetic": True,
            "optimization_performed": False,
            "training_data_used": False,
        },
        edit_impact={
            "scenario_types": [
                "target_success",
                "unrelated_locality",
                "general_ability_sentinel",
            ]
        },
        extra={
            "edit_topology": {
                "artifact_kind": "checkpoint",
                "runtime_activation_policy": "static_checkpoint",
                "training_or_edit_data_ref": "none-synthetic-generator",
            },
            "delta_privacy": {
                "delta_available": "hash_only",
                "privacy_sensitivity": "public",
                "public_raw_delta_approved": False,
            },
        },
    )

    assert (
        lora_metadata["actual_storage_format"]
        == "dense_float_with_synthetic_lowrank_delta"
    )
    assert (
        synthetic_dense_update_metadata["actual_storage_format"]
        == "dense_float_with_synthetic_update"
    )
    assert validate_edit_metadata(lora_metadata) == []
    assert validate_edit_metadata(synthetic_dense_update_metadata) == []
