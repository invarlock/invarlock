from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.evidence_packs.python.editing import implementations as edits
from scripts.evidence_packs.python.editing import tensor_ops
from scripts.evidence_packs.python.editing.streaming_transform import (
    replay_transformation_tensor,
)


def test_legacy_mutable_edit_apis_are_absent() -> None:
    retired_names = (
        "apply_fp8_dequantized_simulation",
        "apply_rtn_dequantized_simulation",
        "apply_synthetic_dense_update",
        "apply_synthetic_lowrank_delta",
        "extract_layer_index",
        "fp8_dtype",
        "matches_edit_scope",
        "parse_scope_layers",
        "round_to_nearest_dequantized",
        "total_model_params",
        "truncated_svd",
    )

    for module in (tensor_ops, edits):
        for name in retired_names:
            assert not hasattr(module, name), f"{module.__name__}.{name} survived"

    assert tensor_ops.__all__ == ["magnitude_prune_tensor"]
    assert callable(tensor_ops.magnitude_prune_tensor)


def test_streaming_replay_replaces_mutable_tensor_edits_without_mutation() -> None:
    source = torch.tensor(
        [[-2.0, -1.2, -0.1, 0.7, 1.9], [1.5, 0.9, 0.2, -0.8, -1.7]],
        dtype=torch.float32,
    )
    original = source.clone()

    quantized = replay_transformation_tensor(
        source,
        edit_type="quant_rtn",
        parameters={"bits": 4, "group_size": 3},
    )
    lowrank = replay_transformation_tensor(
        source,
        edit_type="synthetic_lowrank_delta",
        parameters={"rank": 2, "scale": 8.0},
    )
    dense = replay_transformation_tensor(
        source,
        edit_type="synthetic_dense_update",
        parameters={"step_size": 0.001, "iterations": 2},
    )

    assert torch.equal(source, original)
    for transformed in (quantized, lowrank, dense):
        assert transformed.shape == source.shape
        assert transformed.dtype == source.dtype
        assert torch.isfinite(transformed).all()
        assert not torch.equal(transformed, source)


def test_magnitude_pruning_tie_breaking_is_exact_and_deterministic() -> None:
    weight = torch.ones(2, 3)

    pruned = tensor_ops.magnitude_prune_tensor(weight, 0.5)

    assert torch.equal(
        pruned,
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )


def test_magnitude_pruning_preserves_zero_and_full_sparsity_boundaries() -> None:
    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]])

    unchanged = tensor_ops.magnitude_prune_tensor(weight, 0.0)
    fully_pruned = tensor_ops.magnitude_prune_tensor(weight, 1.0)

    assert unchanged is weight
    assert torch.equal(fully_pruned, torch.zeros_like(weight))

    unique_threshold = tensor_ops.magnitude_prune_tensor(weight, 0.25)
    assert torch.equal(unique_threshold, torch.tensor([[0.0, -2.0], [3.0, -4.0]]))


def test_magnitude_pruning_rejects_impossible_threshold_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
    original = torch.kthvalue

    def invalid_threshold(values: torch.Tensor, k: int):
        result = original(values, k)
        return SimpleNamespace(values=torch.tensor(5.0), indices=result.indices)

    monkeypatch.setattr(torch, "kthvalue", invalid_threshold)
    with pytest.raises(RuntimeError, match="threshold accounting underflowed"):
        tensor_ops.magnitude_prune_tensor(weight, 0.5)


@pytest.mark.parametrize("edit_type", ("fp8_quant", "lowrank_svd"))
def test_retired_generated_metadata_cannot_be_constructed(edit_type: str) -> None:
    with pytest.raises(ValueError, match="dedicated storage and replay contract"):
        edits.build_validation_edit_metadata(
            edit_type=edit_type,
            scope="ffn",
        )


def test_clean_specs_fail_closed_without_a_selection_receipt(tmp_path: Path) -> None:
    resolved = edits.resolve_edit_spec(
        model_output_dir=tmp_path / "model",
        edit_spec="synthetic_lowrank_delta:clean",
        version_hint="clean",
    )

    assert resolved.status == "invalid"
    assert resolved.reason == "clean_selection_requires_receipt"
    assert resolved.edit_dir_name == ""


def test_direct_verifier_grade_specs_validate_against_the_streaming_contract(
    tmp_path: Path,
) -> None:
    lowrank = edits.resolve_edit_spec(
        model_output_dir=tmp_path / "model",
        edit_spec="synthetic_lowrank_delta:4:8:attn",
        version_hint="stress",
    )
    dense = edits.resolve_edit_spec(
        model_output_dir=tmp_path / "model",
        edit_spec="synthetic_dense_update:0.0001:1:ffn",
        version_hint="stress",
    )

    assert lowrank.selected
    assert lowrank.edit_dir_name == edits.generated_transformation_edit_dir_name(
        edit_type="synthetic_lowrank_delta",
        parameters={"rank": 4, "scale": 8.0},
        scope="attn",
        version="stress",
    )
    assert "rank-4" in lowrank.edit_dir_name
    assert "scale-8" in lowrank.edit_dir_name
    assert "scope-attn" in lowrank.edit_dir_name
    assert "sha256-" in lowrank.edit_dir_name
    assert lowrank.to_batch_payload()["rank"] == 4
    assert lowrank.to_batch_payload()["scale"] == 8.0
    assert dense.selected
    assert dense.edit_dir_name == edits.generated_transformation_edit_dir_name(
        edit_type="synthetic_dense_update",
        parameters={"step_size": 0.0001, "iterations": 1},
        scope="ffn",
        version="stress",
    )
    assert "iterations-1" in dense.edit_dir_name
    assert "step_size-0.0001" in dense.edit_dir_name
    assert "scope-ffn" in dense.edit_dir_name
    assert dense.to_batch_payload()["step_size"] == 0.0001
    assert dense.to_batch_payload()["iterations"] == 1


@pytest.mark.parametrize(
    ("first_spec", "second_spec"),
    [
        ("quant_rtn:4:32:ffn", "quant_rtn:4:64:ffn"),
        ("quant_rtn:4:32:ffn", "quant_rtn:4:32:attn"),
        (
            "synthetic_lowrank_delta:4:2:ffn",
            "synthetic_lowrank_delta:4:8:ffn",
        ),
        (
            "synthetic_lowrank_delta:4:8:ffn",
            "synthetic_lowrank_delta:4:8:attn",
        ),
        (
            "synthetic_dense_update:0.0001:2:ffn",
            "synthetic_dense_update:0.01:2:ffn",
        ),
        (
            "synthetic_dense_update:0.01:2:ffn",
            "synthetic_dense_update:0.01:2:attn",
        ),
    ],
)
def test_raw_transform_directory_identity_covers_every_effective_parameter_and_scope(
    tmp_path: Path,
    first_spec: str,
    second_spec: str,
) -> None:
    first = edits.resolve_edit_spec(
        model_output_dir=tmp_path / "model",
        edit_spec=first_spec,
        version_hint="stress",
    )
    second = edits.resolve_edit_spec(
        model_output_dir=tmp_path / "model",
        edit_spec=second_spec,
        version_hint="stress",
    )
    repeated = edits.resolve_edit_spec(
        model_output_dir=tmp_path / "model",
        edit_spec=first_spec,
        version_hint="stress",
    )

    assert first.selected and second.selected and repeated.selected
    assert first.edit_dir_name != second.edit_dir_name
    assert first.edit_dir_name == repeated.edit_dir_name
    assert first.edit_dir_name.startswith("generated--")
    assert "--sha256-" in first.edit_dir_name


def test_real_training_edit_specs_fail_with_migration_guidance(tmp_path: Path) -> None:
    lowrank = edits.resolve_edit_spec(
        model_output_dir=tmp_path / "model",
        edit_spec="lora_merge:4:8:attn",
        version_hint="stress",
    )
    dense = edits.resolve_edit_spec(
        model_output_dir=tmp_path / "model",
        edit_spec="fine_tune:0.0001:1:ffn",
        version_hint="stress",
    )

    assert lowrank.status == "invalid"
    assert "verifier-grade synthetic fixture" in lowrank.reason
    assert "real PEFT/LoRA integration or training campaign" in lowrank.reason
    assert dense.status == "invalid"
    assert "verifier-grade synthetic fixture" in dense.reason
    assert "real fine-tune integration or training campaign" in dense.reason
