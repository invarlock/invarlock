from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.evidence_packs.python.editing import streaming_transform
from scripts.evidence_packs.python.editing.streaming_transform import (
    TRANSFORMATION_MATERIALIZATION_RECEIPT,
    TRANSFORMATION_MATERIALIZATION_SCHEMA,
    TRANSFORMATION_PROGRESS_FILE,
    materialize_transformation_artifact,
    replay_transformation_tensor,
)
from scripts.evidence_packs.python.editing.transformation_contract import (
    TransformationContractError,
)
from scripts.evidence_packs.python.editing.validate_artifact import (
    validate_edit_artifact,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_baseline(path: Path, *, target_value: float = 0.37) -> None:
    path.mkdir(parents=True)
    _write_json(path / "config.json", {"model_type": "qwen2", "num_hidden_layers": 1})
    _write_json(path / "tokenizer_config.json", {"model_max_length": 128})
    _write_json(path / "tokenizer.json", {"version": "1.0"})
    (path / "vocab.json").write_bytes(b'{"a": 0}\n')
    save_file(
        {
            "model.layers.0.mlp.up_proj.weight": torch.tensor(
                [
                    [target_value, 0.72, -1.11, 2.43],
                    [-0.31, 1.79, 0.16, -2.07],
                    [1.31, -0.54, 3.27, 0.91],
                    [-1.77, 0.43, -0.81, 2.14],
                ],
                dtype=torch.float32,
            ),
            "model.layers.0.self_attn.q_proj.weight": torch.tensor(
                [[-0.0, 1.0], [2.0, 3.0]], dtype=torch.float32
            ),
            "model.visual.blocks.0.mlp.up_proj.weight": torch.tensor(
                [[4.0, 5.0], [6.0, 7.0]], dtype=torch.float32
            ),
        },
        path / "model.safetensors",
        metadata={"format": "pt"},
    )


def _tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(torch.uint8)


def _artifact_tensors(artifact: Path) -> dict[str, torch.Tensor]:
    index = json.loads(
        (artifact / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    result: dict[str, torch.Tensor] = {}
    for name, shard_name in index["weight_map"].items():
        with safe_open(
            str(artifact / shard_name), framework="pt", device="cpu"
        ) as handle:
            result[name] = handle.get_tensor(name)
    return result


def _baseline_tensors(baseline: Path) -> dict[str, torch.Tensor]:
    with safe_open(
        str(baseline / "model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}


@pytest.mark.parametrize(
    ("edit_type", "parameters"),
    [
        ("quant_rtn", {"bits": 4, "group_size": 2}),
        ("synthetic_lowrank_delta", {"rank": 2, "scale": 2.0}),
        ("synthetic_dense_update", {"step_size": 0.001, "iterations": 2}),
    ],
)
def test_streaming_materializer_changes_only_architecture_selected_tensors(
    tmp_path: Path,
    edit_type: str,
    parameters: dict[str, object],
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_baseline(baseline)

    result = materialize_transformation_artifact(
        baseline_path=baseline,
        output_path=artifact,
        edit_type=edit_type,
        parameters=parameters,
        scope="ffn",
    )

    assert result["schema"] == TRANSFORMATION_MATERIALIZATION_SCHEMA
    assert result["ok"] is True
    assert result["selected_tensors"] == 1
    assert result["actual_changes"]["value_changed_tensors"] == 1
    assert result["actual_changes"]["value_changed_params"] > 0
    assert result["source_shard_plan_sha256"].startswith("sha256:")
    assert result["output_shard_plan_sha256"].startswith("sha256:")
    assert validate_edit_artifact(
        artifact,
        require_metadata=True,
        expected_edit_type=edit_type,
        expected_artifact_class="validation_subject_checkpoint",
    ).ok

    receipt = json.loads(
        (artifact / TRANSFORMATION_MATERIALIZATION_RECEIPT).read_text(encoding="utf-8")
    )
    assert receipt["target_manifest_sha256"] == result["target_manifest_sha256"]
    assert receipt["source_shard_plan_sha256"] == result["source_shard_plan_sha256"]
    assert receipt["transformation"]["parameters"] == parameters
    assert receipt["actual_changes"] == result["actual_changes"]

    source = _baseline_tensors(baseline)
    output = _artifact_tensors(artifact)
    target_name = "model.layers.0.mlp.up_proj.weight"
    assert not torch.equal(
        _tensor_bytes(output[target_name]), _tensor_bytes(source[target_name])
    )
    for name in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.visual.blocks.0.mlp.up_proj.weight",
    ):
        assert torch.equal(_tensor_bytes(output[name]), _tensor_bytes(source[name]))
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    ):
        assert (artifact / name).read_bytes() == (baseline / name).read_bytes()


def test_streaming_materializer_rejects_copy_baseline_and_no_target_subjects(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    _write_baseline(baseline, target_value=1.0)
    # Every target element is exactly a representable 4-bit RTN endpoint, so
    # grouping cannot introduce an effective value change.
    save_file(
        {
            "model.layers.0.mlp.up_proj.weight": torch.ones(4, 4),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
        },
        baseline / "model.safetensors",
    )
    with pytest.raises(RuntimeError, match="no effective parameter changes"):
        materialize_transformation_artifact(
            baseline_path=baseline,
            output_path=tmp_path / "noop",
            edit_type="quant_rtn",
            parameters={"bits": 4, "group_size": 2},
            scope="ffn",
        )
    assert not (tmp_path / "noop").exists()

    no_target = tmp_path / "no-target"
    no_target.mkdir()
    _write_json(
        no_target / "config.json",
        {"model_type": "qwen2", "num_hidden_layers": 1},
    )
    _write_json(no_target / "tokenizer_config.json", {"model_max_length": 128})
    save_file(
        {"model.embed_tokens.weight": torch.ones(4, 4)},
        no_target / "model.safetensors",
    )
    with pytest.raises(TransformationContractError, match="no selected tensors"):
        materialize_transformation_artifact(
            baseline_path=no_target,
            output_path=tmp_path / "no-target-output",
            edit_type="quant_rtn",
            parameters={"bits": 4, "group_size": 2},
            scope="ffn",
        )


@pytest.mark.parametrize(
    ("model_type", "name"),
    (
        ("qwen3", "model.layers.0.auxiliary.mlp.up_proj.weight"),
        ("qwen3", "model.layers.0.multi_token_prediction.mlp.up_proj.weight"),
    ),
)
def test_streaming_materializer_rejects_noncanonical_or_auxiliary_targets(
    tmp_path: Path, model_type: str, name: str
) -> None:
    """Raw materialization must share the final pack's target exclusions."""

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _write_json(
        baseline / "config.json",
        {"model_type": model_type, "num_hidden_layers": 1},
    )
    _write_json(baseline / "tokenizer_config.json", {"model_max_length": 128})
    save_file(
        {name: torch.ones(4, 4)},
        baseline / "model.safetensors",
        metadata={"format": "pt"},
    )

    with pytest.raises(TransformationContractError, match="no selected tensors"):
        materialize_transformation_artifact(
            baseline_path=baseline,
            output_path=tmp_path / "artifact",
            edit_type="quant_rtn",
            parameters={"bits": 4, "group_size": 2},
            scope="ffn",
        )


def test_streaming_materializer_binds_and_rewrites_a_sharded_safetensors_topology(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    baseline.mkdir()
    _write_json(
        baseline / "config.json", {"model_type": "qwen2", "num_hidden_layers": 1}
    )
    _write_json(baseline / "tokenizer_config.json", {"model_max_length": 128})
    save_file(
        {"model.layers.0.mlp.up_proj.weight": torch.arange(16).reshape(4, 4).float()},
        baseline / "model-00001-of-00002.safetensors",
    )
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.arange(4)
            .reshape(2, 2)
            .float()
        },
        baseline / "model-00002-of-00002.safetensors",
    )
    _write_json(
        baseline / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 80},
            "weight_map": {
                "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            },
        },
    )

    result = materialize_transformation_artifact(
        baseline_path=baseline,
        output_path=artifact,
        edit_type="synthetic_lowrank_delta",
        parameters={"rank": 2, "scale": 2.0},
        scope="ffn",
    )

    receipt = json.loads(
        (artifact / TRANSFORMATION_MATERIALIZATION_RECEIPT).read_text(encoding="utf-8")
    )
    metadata = json.loads((artifact / "edit_metadata.json").read_text(encoding="utf-8"))
    assert result["output_shards"] == 2
    for field in (
        "max_output_shard_bytes",
        "source_shard_plan",
        "source_shard_plan_sha256",
        "output_shard_plan",
        "output_shard_plan_sha256",
    ):
        assert metadata[field] == receipt[field] == result[field]
    assert receipt["max_output_shard_bytes"] == 1024 * 1024 * 1024
    source_plan = receipt["source_shard_plan"]
    output_plan = receipt["output_shard_plan"]
    assert set(source_plan) == {"source_shards"}
    assert len(source_plan["source_shards"]) == 2
    assert all(
        set(shard) == {"path", "sha256", "tensor_names", "byte_count"}
        for shard in source_plan["source_shards"]
    )
    assert set(output_plan) == {
        "source_shard_plan_sha256",
        "target_manifest_sha256",
        "chunks",
    }
    assert all(
        set(chunk)
        == {"name", "source_path", "source_sha256", "tensor_names", "byte_count"}
        for chunk in output_plan["chunks"]
    )
    assert _canonical_sha256(source_plan) == receipt["source_shard_plan_sha256"]
    assert _canonical_sha256(output_plan) == receipt["output_shard_plan_sha256"]
    assert (
        output_plan["source_shard_plan_sha256"] == receipt["source_shard_plan_sha256"]
    )
    assert output_plan["target_manifest_sha256"] == receipt["target_manifest_sha256"]
    assert receipt["output_weights"]["index_sha256"].startswith("sha256:")
    assert len(receipt["output_weights"]["shards"]) == 2
    output = _artifact_tensors(artifact)
    source_attn = torch.arange(4).reshape(2, 2).float()
    assert torch.equal(
        _tensor_bytes(output["model.layers.0.self_attn.q_proj.weight"]),
        _tensor_bytes(source_attn),
    )


@pytest.mark.parametrize(
    ("first_shard", "second_shard"),
    (
        ("model-00001.safetensors", "model-00001.safetensors"),
        ("model-00001.safetensors", "model-00002.safetensors"),
    ),
    ids=("same-shard", "different-shards"),
)
def test_streaming_materializer_rejects_duplicate_source_weight_map_keys(
    tmp_path: Path,
    first_shard: str,
    second_shard: str,
) -> None:
    """Do not let JSON's last-key-wins behavior choose a source topology."""

    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    baseline.mkdir()
    _write_json(
        baseline / "config.json", {"model_type": "qwen2", "num_hidden_layers": 1}
    )
    save_file(
        {"model.layers.0.mlp.up_proj.weight": torch.arange(16).reshape(4, 4).float()},
        baseline / "model-00001.safetensors",
    )
    if second_shard != first_shard:
        save_file(
            {"model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2)},
            baseline / "model-00002.safetensors",
        )
    tensor_name = "model.layers.0.mlp.up_proj.weight"
    (baseline / "model.safetensors.index.json").write_text(
        '{"metadata":{},"weight_map":{'
        f'"{tensor_name}":"{first_shard}",'
        f'"{tensor_name}":"{second_shard}"}}}}',
        encoding="utf-8",
    )

    with pytest.raises(TransformationContractError, match="duplicate key"):
        materialize_transformation_artifact(
            baseline_path=baseline,
            output_path=artifact,
            edit_type="quant_rtn",
            parameters={"bits": 4, "group_size": 2},
            scope="ffn",
        )
    assert not artifact.exists()


@pytest.mark.parametrize(
    ("edit_type", "parameters", "scope"),
    [
        ("quant_rtn", {"bits": 1, "group_size": 32}, "ffn"),
        ("synthetic_lowrank_delta", {"rank": 0, "scale": 1.0}, "ffn"),
        ("synthetic_dense_update", {"step_size": 0.0, "iterations": 1}, "ffn"),
        ("quant_rtn", {"bits": 4, "group_size": 32}, "unknown"),
    ],
)
def test_streaming_materializer_rejects_invalid_parameters_and_scope(
    tmp_path: Path,
    edit_type: str,
    parameters: dict[str, object],
    scope: str,
) -> None:
    baseline = tmp_path / "baseline"
    _write_baseline(baseline)

    with pytest.raises(TransformationContractError):
        materialize_transformation_artifact(
            baseline_path=baseline,
            output_path=tmp_path / "artifact",
            edit_type=edit_type,
            parameters=parameters,
            scope=scope,
        )


def test_streaming_materializer_resumes_only_from_durable_chunk_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    baseline.mkdir()
    _write_json(
        baseline / "config.json", {"model_type": "qwen2", "num_hidden_layers": 3}
    )
    _write_json(baseline / "tokenizer_config.json", {"model_max_length": 128})
    save_file(
        {
            f"model.layers.{index}.mlp.up_proj.weight": torch.arange(
                160_000, dtype=torch.float32
            ).reshape(400, 400)
            + index
            for index in range(3)
        },
        baseline / "model.safetensors",
    )

    original_write = streaming_transform._write_json_atomic
    interrupted = False

    def interrupt_after_durable_progress(path: Path, payload: object) -> None:
        nonlocal interrupted
        assert isinstance(payload, dict)
        original_write(path, payload)
        if (
            path.name == TRANSFORMATION_PROGRESS_FILE
            and payload.get("completed_shards")
            and not interrupted
        ):
            interrupted = True
            raise KeyboardInterrupt("simulated interruption after durable progress")

    monkeypatch.setattr(
        streaming_transform, "_write_json_atomic", interrupt_after_durable_progress
    )
    with pytest.raises(KeyboardInterrupt, match="durable progress"):
        materialize_transformation_artifact(
            baseline_path=baseline,
            output_path=artifact,
            edit_type="synthetic_dense_update",
            parameters={"step_size": 0.001, "iterations": 2},
            scope="ffn",
            max_output_shard_bytes=1024 * 1024,
        )

    staging = artifact.parent / f".{artifact.name}.tmp"
    progress = json.loads(
        (staging / TRANSFORMATION_PROGRESS_FILE).read_text(encoding="utf-8")
    )
    assert progress["completed_shards"]
    assert not artifact.exists()
    monkeypatch.setattr(streaming_transform, "_write_json_atomic", original_write)
    with pytest.raises(RuntimeError, match="max_output_shard_bytes"):
        materialize_transformation_artifact(
            baseline_path=baseline,
            output_path=artifact,
            edit_type="synthetic_dense_update",
            parameters={"step_size": 0.001, "iterations": 2},
            scope="ffn",
            max_output_shard_bytes=2 * 1024 * 1024,
        )
    result = materialize_transformation_artifact(
        baseline_path=baseline,
        output_path=artifact,
        edit_type="synthetic_dense_update",
        parameters={"step_size": 0.001, "iterations": 2},
        scope="ffn",
        max_output_shard_bytes=1024 * 1024,
    )

    assert result["resumed"] is True
    assert result["output_shards"] == 3
    assert not staging.exists()


@pytest.mark.parametrize(
    ("edit_type", "parameters"),
    [
        ("quant_rtn", {"bits": 4, "group_size": 2}),
        ("synthetic_lowrank_delta", {"rank": 2, "scale": 2.0}),
        ("synthetic_dense_update", {"step_size": 0.001, "iterations": 2}),
    ],
)
def test_canonical_replay_is_deterministic_and_does_not_mutate_its_input(
    edit_type: str,
    parameters: dict[str, object],
) -> None:
    source = torch.tensor(
        [[0.37, -1.11, 2.43, 0.72], [1.79, -0.31, 0.16, -2.07]],
        dtype=torch.float32,
    )
    before = _tensor_bytes(source).clone()

    first = replay_transformation_tensor(
        source, edit_type=edit_type, parameters=parameters
    )
    second = replay_transformation_tensor(
        source, edit_type=edit_type, parameters=parameters
    )

    assert torch.equal(_tensor_bytes(source), before)
    assert torch.equal(_tensor_bytes(first), _tensor_bytes(second))
    assert not torch.equal(_tensor_bytes(first), before)


def test_dense_replay_executes_literal_storage_dtype_iterations() -> None:
    source = torch.full((4, 4), 0.37, dtype=torch.float16)
    once = replay_transformation_tensor(
        source,
        edit_type="synthetic_dense_update",
        parameters={"step_size": 0.001, "iterations": 1},
    )
    twice = replay_transformation_tensor(
        source,
        edit_type="synthetic_dense_update",
        parameters={"step_size": 0.001, "iterations": 2},
    )
    assert not torch.equal(_tensor_bytes(once), _tensor_bytes(twice))


def test_groupwise_rtn_replay_uses_per_row_group_scales_and_signed_bounds() -> None:
    source = torch.tensor([[1.0, 0.2, -1.0, -0.2]], dtype=torch.float32)
    output = replay_transformation_tensor(
        source,
        edit_type="quant_rtn",
        parameters={"bits": 2, "group_size": 2},
    )
    # For two-bit signed RTN, qmin=-2 and qmax=1.  Both groups have unit
    # maximum magnitude, so the canonical scale is one and +/-0.2 round to 0.
    assert torch.equal(
        output,
        torch.tensor([[1.0, 0.0, -1.0, 0.0]], dtype=torch.float32),
    )
