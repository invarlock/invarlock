from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from invarlock.pruning_contract import PruningContractError
from scripts.evidence_packs.python.create_edit_model import main as create_edit_main
from scripts.evidence_packs.python.editing import streaming_pruning
from scripts.evidence_packs.python.editing.streaming_pruning import (
    PRUNING_MATERIALIZATION_SCHEMA,
    PRUNING_PROGRESS_FILE,
    materialize_magnitude_pruned_artifact,
)
from scripts.evidence_packs.python.editing.validate_artifact import (
    validate_edit_artifact,
    validate_pruning_artifact,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_baseline(path: Path) -> None:
    path.mkdir(parents=True)
    _write_json(path / "config.json", {"model_type": "qwen2"})
    _write_json(path / "tokenizer_config.json", {"model_max_length": 128})
    _write_json(path / "tokenizer.json", {"version": "1.0"})
    (path / "vocab.json").write_bytes(b'{"a": 0}\n')
    save_file(
        {
            "model.layers.0.mlp.up_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.tensor(
                [[1.0, 1.0], [2.0, 2.0]]
            ),
            "model.layers.0.self_attn.q_proj.weight": torch.tensor(
                [[-0.0, 1.0], [2.0, 3.0]]
            ),
            "model.visual.blocks.0.mlp.up_proj.weight": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]]
            ),
        },
        path / "model.safetensors",
        metadata={"format": "pt"},
    )


def _tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(torch.uint8)


def test_streaming_pruning_rejects_duplicate_index_keys(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map":{"tensor":"one.safetensors","tensor":"two.safetensors"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(PruningContractError, match="invalid model.safetensors.index"):
        streaming_pruning._weight_map(checkpoint)


def test_streaming_materializer_preserves_support_files_and_passes_exact_replay(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_baseline(baseline)

    result = materialize_magnitude_pruned_artifact(
        baseline_path=baseline,
        output_path=artifact,
        sparsity=0.5,
        scope="ffn",
        device="cpu",
    )

    assert result["schema"] == PRUNING_MATERIALIZATION_SCHEMA
    assert result["ok"] is True
    assert result["selected_tensors"] == 2
    assert result["effective_changed_params"] == 4
    assert (artifact / "model.safetensors.index.json").is_file()
    for support_name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    ):
        assert (artifact / support_name).read_bytes() == (
            baseline / support_name
        ).read_bytes()

    replay = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )
    assert replay["ok"] is True
    assert replay["selected_tensors"] == 2
    assert replay["expected_pruned_params"] == 4

    from safetensors import safe_open

    with safe_open(
        str(baseline / "model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        baseline_attn = handle.get_tensor("model.layers.0.self_attn.q_proj.weight")
    artifact_map = json.loads(
        (artifact / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    with safe_open(
        str(artifact / artifact_map["model.layers.0.self_attn.q_proj.weight"]),
        framework="pt",
        device="cpu",
    ) as handle:
        artifact_attn = handle.get_tensor("model.layers.0.self_attn.q_proj.weight")
    assert torch.equal(_tensor_bytes(artifact_attn), _tensor_bytes(baseline_attn))


def test_create_edit_cli_routes_magnitude_prune_to_streaming_materializer(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_baseline(baseline)

    assert (
        create_edit_main(
            ["magnitude-prune", str(baseline), str(artifact), "0.5", "ffn"]
        )
        == 0
    )
    assert validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )["ok"]


def test_streaming_pruning_artifact_reloads_as_a_causal_model(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    baseline.mkdir()
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_ctx=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
        )
    )
    model.save_pretrained(baseline, safe_serialization=True)
    _write_json(baseline / "tokenizer_config.json", {"model_max_length": 16})

    materialize_magnitude_pruned_artifact(
        baseline_path=baseline,
        output_path=artifact,
        sparsity=0.5,
        scope="ffn",
        device="cpu",
    )

    reloaded = AutoModelForCausalLM.from_pretrained(artifact).eval()
    with torch.inference_mode():
        logits = reloaded(input_ids=torch.tensor([[1, 2, 3]])).logits
    assert logits.shape == (1, 3, 32)
    assert torch.isfinite(logits).all()


def test_streaming_materializer_rejects_noop_or_unmatched_pruning_scope(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    _write_baseline(baseline)

    with pytest.raises(ValueError, match="sparsity must be in"):
        materialize_magnitude_pruned_artifact(
            baseline_path=baseline,
            output_path=tmp_path / "noop",
            sparsity=0.0,
            scope="ffn",
            device="cpu",
        )
    with pytest.raises(ValueError, match="scope must be one of"):
        materialize_magnitude_pruned_artifact(
            baseline_path=baseline,
            output_path=tmp_path / "unmatched",
            sparsity=0.5,
            scope="unknown",
            device="cpu",
        )


def test_streaming_materializer_fails_closed_for_gpt_oss_mxfp4_storage(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    _write_baseline(baseline)
    _write_json(
        baseline / "config.json",
        {
            "model_type": "gpt_oss",
            "quantization_config": {"quant_method": "mxfp4"},
        },
    )

    with pytest.raises(ValueError, match="GPT-OSS/MXFP4"):
        materialize_magnitude_pruned_artifact(
            baseline_path=baseline,
            output_path=tmp_path / "artifact",
            sparsity=0.5,
            scope="all",
            device="cpu",
        )


def test_streaming_materializer_rejects_nonfinite_out_of_scope_tensor(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    _write_baseline(baseline)
    save_file(
        {
            "model.layers.0.mlp.up_proj.weight": torch.ones(2, 2),
            "model.layers.0.self_attn.q_proj.weight": torch.tensor(
                [[float("nan"), 1.0], [2.0, 3.0]]
            ),
        },
        baseline / "model.safetensors",
    )

    with pytest.raises(ValueError, match="non-finite"):
        materialize_magnitude_pruned_artifact(
            baseline_path=baseline,
            output_path=tmp_path / "artifact",
            sparsity=0.5,
            scope="ffn",
            device="cpu",
        )


def test_replay_rejects_forged_pruning_metadata_and_external_weight_path(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_baseline(baseline)
    materialize_magnitude_pruned_artifact(
        baseline_path=baseline,
        output_path=artifact,
        sparsity=0.5,
        scope="ffn",
        device="cpu",
    )

    metadata_path = artifact / "edit_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["coverage"]["edited_tensors"] = 999
    metadata["scope_policy"] = "forged-policy"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    replay = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )
    assert replay["ok"] is False
    assert any("coverage.edited_tensors" in issue for issue in replay["issues"])
    assert any("scope_policy" in issue for issue in replay["issues"])

    index_path = artifact / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    first_tensor = next(iter(index["weight_map"]))
    index["weight_map"][first_tensor] = "../outside.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    assert not validate_edit_artifact(artifact).ok


@pytest.mark.parametrize(
    ("field", "value", "expected_issue"),
    (
        ("total_params", 9, "coverage.total_params"),
        ("coverage_ratio", 0.25, "coverage.coverage_ratio"),
    ),
)
def test_replay_rejects_independently_forged_pruning_coverage(
    tmp_path: Path,
    field: str,
    value: int | float,
    expected_issue: str,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_baseline(baseline)
    materialize_magnitude_pruned_artifact(
        baseline_path=baseline,
        output_path=artifact,
        sparsity=0.5,
        scope="ffn",
        device="cpu",
    )

    metadata_path = artifact / "edit_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["coverage"][field] = value
    if field == "total_params":
        metadata["coverage"]["coverage_ratio"] = (
            metadata["coverage"]["edited_params"] / value
        )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    replay = validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )

    assert replay["ok"] is False
    assert any(expected_issue in issue for issue in replay["issues"])


def test_streaming_materialization_resumes_after_a_durable_progress_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    baseline.mkdir()
    _write_json(baseline / "config.json", {"model_type": "qwen2"})
    _write_json(baseline / "tokenizer_config.json", {"model_max_length": 128})
    tensors = {
        f"model.layers.{index}.mlp.up_proj.weight": torch.arange(
            160_000, dtype=torch.float32
        ).reshape(400, 400)
        + index
        for index in range(3)
    }
    save_file(tensors, baseline / "model.safetensors")

    original_write = streaming_pruning._write_json_atomic
    interrupted = False

    def interrupt_after_durable_progress(
        path: Path, payload: dict[str, object]
    ) -> None:
        nonlocal interrupted
        original_write(path, payload)
        if (
            path.name == PRUNING_PROGRESS_FILE
            and payload.get("completed_shards")
            and not interrupted
        ):
            interrupted = True
            raise KeyboardInterrupt("simulated interruption after durable progress")

    monkeypatch.setattr(
        streaming_pruning, "_write_json_atomic", interrupt_after_durable_progress
    )
    with pytest.raises(KeyboardInterrupt, match="durable progress"):
        materialize_magnitude_pruned_artifact(
            baseline_path=baseline,
            output_path=artifact,
            sparsity=0.5,
            scope="ffn",
            device="cpu",
            max_output_shard_bytes=1024 * 1024,
        )

    staging = artifact.parent / f".{artifact.name}.tmp"
    progress_path = staging / PRUNING_PROGRESS_FILE
    progress_bytes = progress_path.read_bytes()
    progress = json.loads(progress_bytes)
    assert progress["completed_shards"]
    monkeypatch.setattr(streaming_pruning, "_write_json_atomic", original_write)

    duplicate_progress = progress_bytes.replace(
        b'"schema":', b'"schema":"forged", "schema":', 1
    )
    progress_path.write_bytes(duplicate_progress)
    with pytest.raises(RuntimeError, match="not resumable"):
        materialize_magnitude_pruned_artifact(
            baseline_path=baseline,
            output_path=artifact,
            sparsity=0.5,
            scope="ffn",
            device="cpu",
            max_output_shard_bytes=1024 * 1024,
        )

    retired_progress = json.loads(progress_bytes)
    retired_progress["schema"] = "invarlock/pruning-materialization-progress-v2"
    progress_path.write_text(json.dumps(retired_progress), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract mismatch for schema"):
        materialize_magnitude_pruned_artifact(
            baseline_path=baseline,
            output_path=artifact,
            sparsity=0.5,
            scope="ffn",
            device="cpu",
            max_output_shard_bytes=1024 * 1024,
        )

    progress_path.write_bytes(progress_bytes)
    result = materialize_magnitude_pruned_artifact(
        baseline_path=baseline,
        output_path=artifact,
        sparsity=0.5,
        scope="ffn",
        device="cpu",
        max_output_shard_bytes=1024 * 1024,
    )

    assert result["resumed"] is True
    assert result["output_shards"] == 3
    assert validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )["ok"]


def test_streaming_materialization_rebuilds_tampered_completed_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    baseline.mkdir()
    _write_json(baseline / "config.json", {"model_type": "qwen2"})
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

    original_write = streaming_pruning._write_json_atomic
    interrupted = False

    def interrupt_after_first_receipt(path: Path, payload: dict[str, object]) -> None:
        nonlocal interrupted
        original_write(path, payload)
        if (
            path.name == PRUNING_PROGRESS_FILE
            and payload.get("completed_shards")
            and not interrupted
        ):
            interrupted = True
            raise KeyboardInterrupt("simulated interruption after shard receipt")

    monkeypatch.setattr(
        streaming_pruning, "_write_json_atomic", interrupt_after_first_receipt
    )
    with pytest.raises(KeyboardInterrupt, match="shard receipt"):
        materialize_magnitude_pruned_artifact(
            baseline_path=baseline,
            output_path=artifact,
            sparsity=0.5,
            scope="ffn",
            device="cpu",
            max_output_shard_bytes=1024 * 1024,
        )

    staging = artifact.parent / f".{artifact.name}.tmp"
    progress = json.loads((staging / PRUNING_PROGRESS_FILE).read_text(encoding="utf-8"))
    completed = progress["completed_shards"]
    assert isinstance(completed, list) and completed
    first_receipt = completed[0]
    assert isinstance(first_receipt, dict)
    shard_name = first_receipt["name"]
    assert isinstance(shard_name, str)
    staged_shard = staging / shard_name
    original_bytes = staged_shard.read_bytes()
    tampered_bytes = bytearray(original_bytes)
    tampered_bytes[-1] ^= 1
    staged_shard.write_bytes(tampered_bytes)
    assert staged_shard.read_bytes() != original_bytes

    monkeypatch.setattr(streaming_pruning, "_write_json_atomic", original_write)
    result = materialize_magnitude_pruned_artifact(
        baseline_path=baseline,
        output_path=artifact,
        sparsity=0.5,
        scope="ffn",
        device="cpu",
        max_output_shard_bytes=1024 * 1024,
    )

    assert result["resumed"] is True
    assert (artifact / shard_name).read_bytes() == original_bytes
    assert validate_pruning_artifact(
        artifact,
        baseline_dir=baseline,
        scope="ffn",
        target_sparsity=0.5,
    )["ok"]
