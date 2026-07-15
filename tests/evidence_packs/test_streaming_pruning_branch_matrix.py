from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from invarlock.pruning_contract import PruningContractError, checkpoint_pruning_contract
from scripts.evidence_packs.python.editing import streaming_pruning as pruning


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True)
    _write_json(path / "config.json", {"model_type": "qwen2"})
    save_file(tensors, path / "model.safetensors")


def _chunk(name: str = "model-00001-of-00001.safetensors") -> pruning._ShardChunk:
    return pruning._ShardChunk(
        name=name,
        source_path=Path("source.safetensors"),
        tensor_names=("weight",),
        byte_count=16,
    )


def _plan(*chunks: pruning._ShardChunk) -> pruning._MaterializationPlan:
    selected = chunks or (_chunk(),)
    return pruning._MaterializationPlan(
        weights={},
        index_path=None,
        chunks=selected,
        target_names=frozenset({"weight"}),
        target_manifest={"schema": "test"},
        target_manifest_sha256="sha256:" + "1" * 64,
        shard_plan_sha256="sha256:" + "2" * 64,
        total_params=4,
        total_weight_bytes=16,
        selected_tensors=1,
        selected_params=4,
        expected_pruned_params=2,
        original_zero_params=0,
    )


def _contract() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="qwen2",
        architecture="qwen2_dense",
        config_sha256="sha256:" + "3" * 64,
    )


def _progress(plan: pruning._MaterializationPlan) -> dict[str, object]:
    return pruning._progress_base(
        baseline_identity={
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "4" * 64,
        },
        contract=_contract(),
        scope="ffn",
        sparsity=0.5,
        plan=plan,
    )


def test_streaming_pruning_scalar_and_device_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert pruning._valid_sha256("sha256:" + "a" * 64)
    assert not pruning._valid_sha256(None)
    assert not pruning._valid_sha256("md5:" + "a" * 64)
    assert not pruning._valid_sha256("sha256:" + "g" * 64)
    assert pruning._is_nonnegative_int(0)
    assert not pruning._is_nonnegative_int(True)

    class MetadataHandle:
        def __init__(self, metadata: object) -> None:
            self._metadata = metadata

        def metadata(self) -> object:
            return self._metadata

    assert pruning._safe_metadata(MetadataHandle(None)) == {"format": "pt"}
    assert pruning._safe_metadata(MetadataHandle({1: 2})) == {
        "1": "2",
        "format": "pt",
    }

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA pruning"):
        pruning._resolve_device("cuda")
    assert pruning._resolve_device("auto") == torch.device("cpu")

    for invalid in (True, 1, 1024.5):
        with pytest.raises(ValueError, match="at least 1 MiB"):
            pruning._require_output_shard_bytes(invalid)  # type: ignore[arg-type]
    assert pruning._require_output_shard_bytes(1024 * 1024) == 1024 * 1024
    assert (
        pruning._chunk_source_tensors(
            Path("source"), [], {}, max_output_shard_bytes=1024
        )
        == []
    )


def test_weight_map_rejects_empty_missing_and_unsafe_checkpoint_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(PruningContractError, match="requires safetensors"):
        pruning._weight_map(missing)

    empty_index = tmp_path / "empty-index"
    empty_index.mkdir()
    _write_json(empty_index / "model.safetensors.index.json", {})
    with pytest.raises(PruningContractError, match="no weight_map"):
        pruning._weight_map(empty_index)

    nonstring = tmp_path / "nonstring"
    nonstring.mkdir()
    _write_json(
        nonstring / "model.safetensors.index.json",
        {"weight_map": {"weight": 3}},
    )
    with pytest.raises(PruningContractError, match="non-string"):
        pruning._weight_map(nonstring)

    missing_shard = tmp_path / "missing-shard"
    missing_shard.mkdir()
    _write_json(
        missing_shard / "model.safetensors.index.json",
        {"weight_map": {"weight": "missing.safetensors"}},
    )
    with pytest.raises(PruningContractError, match="missing"):
        pruning._weight_map(missing_shard)

    empty_single = tmp_path / "empty-single"
    empty_single.mkdir()
    (empty_single / "model.safetensors").write_bytes(b"placeholder")

    class EmptySafeOpen:
        def __enter__(self) -> EmptySafeOpen:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def keys(self) -> list[str]:
            return []

    monkeypatch.setattr(pruning, "safe_open", lambda *_args, **_kwargs: EmptySafeOpen())
    with pytest.raises(PruningContractError, match="contains no tensors"):
        pruning._weight_map(empty_single)


def test_plan_rejects_nonfloating_target_and_empty_effective_scope(
    tmp_path: Path,
) -> None:
    integer_checkpoint = tmp_path / "integer"
    _write_checkpoint(
        integer_checkpoint,
        {"model.layers.0.mlp.up_proj.weight": torch.ones(2, 2, dtype=torch.int64)},
    )
    weights, index = pruning._weight_map(integer_checkpoint)
    contract = checkpoint_pruning_contract(integer_checkpoint)
    with pytest.raises(PruningContractError, match="non-floating target"):
        pruning._build_plan(
            baseline_path=integer_checkpoint,
            weights=weights,
            index_path=index,
            scope="ffn",
            sparsity=0.5,
            contract=contract,
            max_output_shard_bytes=1024 * 1024,
        )

    unmatched = tmp_path / "unmatched"
    _write_checkpoint(
        unmatched,
        {"model.embed_tokens.weight": torch.ones(2, 2)},
    )
    weights, index = pruning._weight_map(unmatched)
    contract = checkpoint_pruning_contract(unmatched)
    with pytest.raises(PruningContractError, match="retain selected tensors"):
        pruning._build_plan(
            baseline_path=unmatched,
            weights=weights,
            index_path=index,
            scope="ffn",
            sparsity=0.5,
            contract=contract,
            max_output_shard_bytes=1024 * 1024,
        )


def test_support_copy_skips_weights_and_generated_receipts(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    staging = tmp_path / "staging"
    baseline.mkdir()
    staging.mkdir()
    nested = baseline / "nested"
    nested.mkdir()
    support = nested / "tokenizer.json"
    support.write_text("support", encoding="utf-8")
    weight = baseline / "model.safetensors"
    weight.write_bytes(b"weight")
    index = baseline / "model.safetensors.index.json"
    index.write_text("{}", encoding="utf-8")
    for generated in pruning._GENERATED_METADATA_FILES:
        (baseline / generated).write_text("stale", encoding="utf-8")

    pruning._copy_support_files(
        baseline,
        staging,
        weight_paths={weight},
        index_path=index,
    )

    assert (staging / "nested" / "tokenizer.json").read_text() == "support"
    assert not (staging / "model.safetensors").exists()
    assert not (staging / "model.safetensors.index.json").exists()
    assert all(
        not (staging / name).exists() for name in pruning._GENERATED_METADATA_FILES
    )


def test_storage_preflight_rejects_insufficient_bytes_and_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pruning.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1),
    )
    with pytest.raises(RuntimeError, match="insufficient free disk"):
        pruning._storage_preflight(
            tmp_path,
            output_weight_bytes=1024,
            output_shards=1,
        )

    monkeypatch.setattr(
        pruning.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=10**12),
    )
    monkeypatch.setattr(
        pruning.os, "statvfs", lambda _path: SimpleNamespace(f_favail=1)
    )
    with pytest.raises(RuntimeError, match="insufficient free inodes"):
        pruning._storage_preflight(
            tmp_path,
            output_weight_bytes=1024,
            output_shards=1,
        )


def test_completed_receipts_reject_each_malformed_authenticated_field() -> None:
    digest = "sha256:" + "a" * 64
    stats = pruning._CompletedShardStats(2, 1).as_dict()
    valid = {"name": "shard", "sha256": digest, "byte_size": 16, "stats": stats}
    assert pruning._completed_entries({"completed_shards": [valid]}) == {"shard": valid}
    malformed = (
        None,
        ["not-an-object"],
        [{**valid, "extra": True}],
        [{**valid, "name": ""}],
        [valid, valid],
        [{**valid, "sha256": "forged"}],
        [{**valid, "byte_size": True}],
        [{**valid, "stats": {}}],
        [{**valid, "stats": {**stats, "observed_zero_params": -1}}],
    )
    for completed_shards in malformed:
        with pytest.raises(RuntimeError, match="malformed"):
            pruning._completed_entries({"completed_shards": completed_shards})


def test_progress_loader_rejects_unsafe_or_malformed_resume_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "4" * 64,
    }
    common = {
        "baseline_path": baseline,
        "baseline_identity": identity,
        "contract": _contract(),
        "scope": "ffn",
        "sparsity": 0.5,
        "plan": plan,
        "restart": False,
    }

    staging_link = tmp_path / "staging-link"
    staging_target = tmp_path / "staging-target"
    staging_target.mkdir()
    staging_link.symlink_to(staging_target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        pruning._load_or_start_progress(staging_path=staging_link, **common)

    staging = tmp_path / "staging"
    staging.mkdir()
    expected = _progress(plan)
    progress_path = staging / pruning.PRUNING_PROGRESS_FILE

    monkeypatch.setattr(
        pruning,
        "read_json_object_snapshot",
        lambda *_args, **_kwargs: (b"[]", []),
    )
    with pytest.raises(RuntimeError, match="not an object"):
        pruning._load_or_start_progress(staging_path=staging, **common)

    monkeypatch.undo()
    _write_json(progress_path, {**expected, "extra": True})
    with pytest.raises(RuntimeError, match="fields are malformed"):
        pruning._load_or_start_progress(staging_path=staging, **common)

    _write_json(progress_path, {**expected, "resume_count": True})
    with pytest.raises(RuntimeError, match="resume_count"):
        pruning._load_or_start_progress(staging_path=staging, **common)


def test_completed_chunk_recovery_rejects_unsafe_files_and_repairs_stale_state(
    tmp_path: Path,
) -> None:
    chunk = _chunk()
    plan = _plan(chunk)
    with pytest.raises(RuntimeError, match="unknown output shards"):
        pruning._validate_completed_chunks(
            staging_path=tmp_path,
            plan=plan,
            completed={"unknown": {}},
        )

    partial = tmp_path / f".{chunk.name}.partial"
    partial.symlink_to(tmp_path / "missing-target")
    with pytest.raises(RuntimeError, match="unsafe partial"):
        pruning._validate_completed_chunks(
            staging_path=tmp_path,
            plan=plan,
            completed={},
        )
    partial.unlink()
    partial.write_bytes(b"partial")
    assert (
        pruning._validate_completed_chunks(
            staging_path=tmp_path, plan=plan, completed={}
        )
        == {}
    )
    assert not partial.exists()

    output = tmp_path / chunk.name
    output.write_bytes(b"unrecorded")
    assert (
        pruning._validate_completed_chunks(
            staging_path=tmp_path, plan=plan, completed={}
        )
        == {}
    )
    assert not output.exists()

    external_target = tmp_path / "external-target"
    external_target.write_bytes(b"external")
    output.symlink_to(external_target)
    with pytest.raises(RuntimeError, match="unsafe unrecorded"):
        pruning._validate_completed_chunks(
            staging_path=tmp_path,
            plan=plan,
            completed={},
        )
    output.unlink()

    stale = {
        chunk.name: {
            "name": chunk.name,
            "sha256": "sha256:" + "a" * 64,
            "byte_size": 16,
            "stats": pruning._CompletedShardStats(2, 1).as_dict(),
        }
    }
    assert (
        pruning._validate_completed_chunks(
            staging_path=tmp_path, plan=plan, completed=stale
        )
        == {}
    )

    output.symlink_to(external_target)
    with pytest.raises(RuntimeError, match="unsafe completed"):
        pruning._validate_completed_chunks(
            staging_path=tmp_path,
            plan=plan,
            completed=stale,
        )
    output.unlink()

    save_file({"weight": torch.ones(2, 2)}, output)
    wrong_size = {chunk.name: {**stale[chunk.name], "byte_size": 1}}
    assert (
        pruning._validate_completed_chunks(
            staging_path=tmp_path,
            plan=plan,
            completed=wrong_size,
        )
        == {}
    )
    assert not output.exists()


def test_materialize_chunk_removes_partial_file_after_serialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk()

    class Handle:
        def get_tensor(self, _name: str) -> torch.Tensor:
            return torch.ones(2, 2)

        def metadata(self) -> dict[str, str]:
            return {"format": "pt"}

    def fail_after_write(
        _tensors: dict[str, torch.Tensor],
        path: Path,
        **_kwargs: object,
    ) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("serialization interrupted")

    monkeypatch.setattr(pruning, "save_file", fail_after_write)
    with pytest.raises(RuntimeError, match="serialization interrupted"):
        pruning._materialize_chunk(
            handle=Handle(),
            chunk=chunk,
            staging_path=tmp_path,
            target_names=frozenset({"weight"}),
            sparsity=0.5,
            active_device=torch.device("cpu"),
        )
    assert not (tmp_path / f".{chunk.name}.partial").exists()


def test_finalize_rejects_noop_and_failed_generic_artifact_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    common = {
        "output_path": tmp_path / "output",
        "baseline_identity": {
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "4" * 64,
        },
        "contract": _contract(),
        "scope": "ffn",
        "sparsity": 0.5,
        "plan": plan,
        "active_device": torch.device("cpu"),
    }
    noop_staging = tmp_path / "noop-staging"
    noop_staging.mkdir()
    noop_progress = _progress(plan)
    noop_progress["completed_shards"] = [
        {
            "name": plan.chunks[0].name,
            "sha256": "sha256:" + "a" * 64,
            "byte_size": 16,
            "stats": pruning._CompletedShardStats(2, 0).as_dict(),
        }
    ]
    with pytest.raises(RuntimeError, match="no effective parameter changes"):
        pruning._finalize_artifact(
            staging_path=noop_staging,
            progress=noop_progress,
            **common,
        )

    failed_staging = tmp_path / "failed-staging"
    failed_staging.mkdir()
    failed_progress = _progress(plan)
    failed_progress["completed_shards"] = [
        {
            "name": plan.chunks[0].name,
            "sha256": "sha256:" + "a" * 64,
            "byte_size": 16,
            "stats": pruning._CompletedShardStats(2, 1).as_dict(),
        }
    ]
    monkeypatch.setattr(pruning, "write_edit_metadata", lambda *_args: None)
    monkeypatch.setattr(
        pruning,
        "validate_edit_artifact",
        lambda *_args, **_kwargs: SimpleNamespace(ok=False, issues=["forged output"]),
    )
    with pytest.raises(RuntimeError, match="forged output"):
        pruning._finalize_artifact(
            staging_path=failed_staging,
            progress=failed_progress,
            **common,
        )


def test_materializer_translates_unsafe_baseline_layout(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline-link"
    baseline.symlink_to(tmp_path / "missing")
    with pytest.raises(PruningContractError):
        pruning.materialize_magnitude_pruned_artifact(
            baseline_path=baseline,
            output_path=tmp_path / "output",
            sparsity=0.5,
            scope="ffn",
            device="cpu",
        )
