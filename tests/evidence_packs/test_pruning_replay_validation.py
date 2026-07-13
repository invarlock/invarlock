from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.evidence_packs.python.editing import validate_pruning as pruning_validator
from scripts.evidence_packs.python.editing.artifact_tensor_validation import (
    _verifier_exact_magnitude_prune_reference,
)
from scripts.evidence_packs.python.editing.implementations import (
    build_validation_edit_metadata,
)
from scripts.evidence_packs.python.editing.streaming_pruning import (
    materialize_magnitude_pruned_artifact,
)
from scripts.evidence_packs.python.editing.validate_artifact import (
    validate_pruning_artifact,
)
from tests.evidence_packs._support_pruning_replay_validation import (
    _metadata,
    _write_checkpoint,
    _write_indexed_checkpoint,
    _write_json,
)


def test_pruning_verifier_reference_uses_stable_flattened_tie_breaking() -> None:
    source = torch.ones(2, 3)

    replayed = _verifier_exact_magnitude_prune_reference(source, 0.5)

    assert torch.equal(
        replayed,
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )


def test_pruning_verifier_reference_preserves_signed_zero_bytes() -> None:
    source = torch.tensor([-0.0, 0.0, 1.0, 2.0], dtype=torch.float32)

    replayed = _verifier_exact_magnitude_prune_reference(source, 0.75)

    expected = torch.tensor([-0.0, 0.0, 0.0, 2.0], dtype=torch.float32)
    assert torch.equal(
        replayed.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf"), float("-inf")))
def test_pruning_verifier_reference_rejects_nonfinite_input(nonfinite: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _verifier_exact_magnitude_prune_reference(
            torch.tensor([1.0, nonfinite], dtype=torch.float32), 0.5
        )


def test_pruning_replay_accepts_exact_prune_and_out_of_scope_equality(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_checkpoint(
        baseline,
        {
            "model.layers.0.mlp.up_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "model.layers.0.self_attn.q_proj.weight": torch.arange(
                4, dtype=torch.float32
            ).reshape(2, 2),
        },
    )
    materialize_magnitude_pruned_artifact(
        baseline_path=baseline,
        output_path=artifact,
        sparsity=0.5,
        scope="ffn",
        device="cpu",
    )

    result = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )

    assert result["ok"] is True
    assert result["selected_tensors"] == 1
    assert result["selected_params"] == 4
    assert result["total_params"] == 8
    assert result["expected_pruned_params"] == 2
    assert result["observed_changed_params"] == 2
    assert result["out_of_scope_tensors_checked"] == 1


def test_parallel_pruning_replay_matches_serial_result(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_checkpoint(
        baseline,
        {
            "model.layers.0.mlp.up_proj.weight": torch.arange(
                16, dtype=torch.float32
            ).reshape(4, 4),
            "model.layers.0.mlp.down_proj.weight": torch.arange(
                16, 32, dtype=torch.float32
            ).reshape(4, 4),
            "model.layers.0.self_attn.q_proj.weight": torch.arange(
                4, dtype=torch.float32
            ).reshape(2, 2),
        },
    )
    materialize_magnitude_pruned_artifact(
        baseline_path=baseline,
        output_path=artifact,
        sparsity=0.5,
        scope="ffn",
        device="cpu",
    )

    serial = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )
    parallel = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
        workers=2,
        worker_threads=1,
    )

    assert parallel == serial


@pytest.mark.parametrize(
    ("workers", "worker_threads", "message"),
    (
        (0, 1, "workers must be between"),
        (1, -1, "worker threads must be between"),
        (9, 1, "workers must be between"),
        (1, 9, "worker threads must be between"),
    ),
)
def test_pruning_replay_rejects_unsafe_worker_settings(
    tmp_path: Path,
    workers: int,
    worker_threads: int,
    message: str,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    _write_checkpoint(baseline, {"model.layers.0.mlp.up_proj.weight": tensor})
    materialize_magnitude_pruned_artifact(
        baseline_path=baseline,
        output_path=artifact,
        sparsity=0.5,
        scope="ffn",
        device="cpu",
    )

    result = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
        workers=workers,
        worker_threads=worker_threads,
    )

    assert result["ok"] is False
    assert any(message in issue for issue in result["issues"])


def test_parallel_pruning_replay_worker_error_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    _write_checkpoint(baseline, {"model.layers.0.mlp.up_proj.weight": tensor})
    materialize_magnitude_pruned_artifact(
        baseline_path=baseline,
        output_path=artifact,
        sparsity=0.5,
        scope="ffn",
        device="cpu",
    )

    def fail_worker(**_kwargs: object) -> None:
        raise RuntimeError("injected replay failure")

    monkeypatch.setattr(pruning_validator, "_pruning_replay_one_tensor", fail_worker)
    result = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
        workers=2,
        worker_threads=1,
    )

    assert result["ok"] is False
    assert any(
        "pruning replay worker failed: RuntimeError: injected replay failure" in issue
        for issue in result["issues"]
    )


def test_pruning_replay_does_not_delegate_to_materializer_tensor_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_checkpoint(
        baseline,
        {
            "model.layers.0.mlp.up_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "model.layers.0.self_attn.q_proj.weight": torch.arange(
                4, dtype=torch.float32
            ).reshape(2, 2),
        },
    )
    materialize_magnitude_pruned_artifact(
        baseline_path=baseline,
        output_path=artifact,
        sparsity=0.5,
        scope="ffn",
        device="cpu",
    )

    from scripts.evidence_packs.python.editing import tensor_ops

    def materializer_helper_must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("validator delegated to the materializer helper")

    monkeypatch.setattr(
        tensor_ops, "magnitude_prune_tensor", materializer_helper_must_not_run
    )

    result = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )

    assert result["ok"] is True


def test_pruning_replay_rejects_unchanged_checkpoint_with_copied_metadata(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    tensors = {
        "model.layers.0.mlp.up_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "model.layers.0.self_attn.q_proj.weight": torch.arange(
            4, dtype=torch.float32
        ).reshape(2, 2),
    }
    _write_checkpoint(baseline, tensors)
    _write_checkpoint(artifact, tensors, metadata=_metadata())

    result = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )

    assert result["ok"] is False
    assert any("exact prune replay" in issue for issue in result["issues"])


def test_pruning_replay_rejects_out_of_scope_tensor_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_checkpoint(
        baseline,
        {
            "model.layers.0.mlp.up_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        },
    )
    _write_checkpoint(
        artifact,
        {
            "model.layers.0.mlp.up_proj.weight": torch.tensor([[0.0, 0.0], [3.0, 4.0]]),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
        },
        metadata=_metadata(),
    )

    result = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )

    assert result["ok"] is False
    assert any("out-of-scope tensor changed" in issue for issue in result["issues"])


def test_pruning_replay_rejects_support_file_or_signed_zero_drift(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_checkpoint(
        baseline,
        {
            "model.layers.0.mlp.up_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "model.layers.0.self_attn.q_proj.weight": torch.tensor(
                [[-0.0, 1.0], [2.0, 3.0]]
            ),
        },
    )
    _write_checkpoint(
        artifact,
        {
            "model.layers.0.mlp.up_proj.weight": torch.tensor([[0.0, 0.0], [3.0, 4.0]]),
            "model.layers.0.self_attn.q_proj.weight": torch.tensor(
                [[0.0, 1.0], [2.0, 3.0]]
            ),
        },
        metadata=_metadata(),
    )
    result = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )

    assert result["ok"] is False
    assert any("out-of-scope tensor changed" in issue for issue in result["issues"])

    _write_json(artifact / "tokenizer_config.json", {"model_max_length": 256})
    support_result = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )
    assert any("support file changed" in issue for issue in support_result["issues"])


def test_pruning_replay_rejects_noop_sparsity(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_checkpoint(
        baseline,
        {"model.layers.0.mlp.up_proj.weight": torch.eye(2)},
    )
    _write_checkpoint(
        artifact,
        {"model.layers.0.mlp.up_proj.weight": torch.eye(2)},
        metadata=build_validation_edit_metadata(
            edit_type="magnitude_prune",
            scope="ffn",
            parameters={"target_sparsity": 0.0},
            coverage={
                "edited_tensors": 1,
                "edited_params": 4,
                "total_params": 4,
                "coverage_ratio": 1.0,
            },
        ),
    )

    result = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.0,
    )

    assert result["ok"] is False
    assert any("sparsity must be in (0, 1)" in issue for issue in result["issues"])


def test_pruning_replay_rejects_index_shard_path_escape(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    tensor_name = "model.layers.0.mlp.up_proj.weight"
    baseline_external = tmp_path / "baseline_external.safetensors"
    artifact_external = tmp_path / "artifact_external.safetensors"
    save_file({tensor_name: torch.tensor([[1.0, 2.0], [3.0, 4.0]])}, baseline_external)
    save_file({tensor_name: torch.tensor([[0.0, 0.0], [3.0, 4.0]])}, artifact_external)
    _write_indexed_checkpoint(
        baseline,
        shard_name="../baseline_external.safetensors",
        tensors={tensor_name: torch.empty(0)},
    )
    _write_indexed_checkpoint(
        artifact,
        shard_name="../artifact_external.safetensors",
        tensors={tensor_name: torch.empty(0)},
        metadata=_metadata(),
    )

    result = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )

    assert result["ok"] is False
    assert any("checkpoint-relative" in issue for issue in result["issues"])


def test_pruning_replay_rejects_gpt_oss_and_non_float_targets(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    tensor_name = "model.layers.0.mlp.up_proj.weight"
    _write_checkpoint(
        baseline,
        {tensor_name: torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)},
        config={"model_type": "gpt_oss", "quantization_config": {"format": "mxfp4"}},
    )
    _write_checkpoint(
        artifact,
        {tensor_name: torch.tensor([[0, 0], [3, 4]], dtype=torch.int8)},
        metadata=_metadata(),
        config={"model_type": "gpt_oss", "quantization_config": {"format": "mxfp4"}},
    )

    result = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )

    assert result["ok"] is False
    assert any("GPT-OSS/MXFP4" in issue for issue in result["issues"])
