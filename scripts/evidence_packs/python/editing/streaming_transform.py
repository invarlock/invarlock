"""Canonical, resumable materialization for verifier-grade transformations.

The older generated-edit paths mutate an in-memory ``transformers`` model.
They are useful for small demonstrations, but they cannot prove that a large
checkpoint was edited only at its declared target tensors.  This module is the
storage-level counterpart: it consumes a regular safetensors checkpoint,
copies non-weight support files, writes bounded output shards through a
durable staging directory, and records enough immutable input information for
a later replay verifier to reproduce every selected tensor.

Only transformations with a public, canonical replay contract belong here.
The materializer owns its implementation; the verifier independently parses
the same repaired v1 ABI and derives expected values without importing this
module.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256

try:
    from .artifact_staging import replace_output, reset_staging, staging_path_for
    from .checkpoint_paths import (
        CheckpointLayoutError,
        require_checkpoint_child_file,
        require_regular_checkpoint_tree,
    )
    from .implementations import build_validation_edit_metadata, write_edit_metadata
    from .streaming_transform_core import (
        _build_plan,
        _canonical_sha256,
        _change_stats,
        _ChangeStats,
        _file_sha256,
        _MaterializationPlan,
        _reject_ambiguous_weight_files,
        _require_output_shard_bytes,
        _safe_metadata,
        _ShardChunk,
        _weight_map,
        replay_transformation_tensor,
    )
    from .transformation_contract import (
        SYNTHETIC_DENSE_UPDATE,
        SYNTHETIC_LOWRANK_DELTA,
        TransformationContractError,
        canonical_transformation_spec,
        checkpoint_transformation_contract,
        validate_transformation_scope,
        validate_transformation_scope_for_contract,
    )
except ImportError:  # pragma: no cover - direct script-path execution
    from artifact_staging import replace_output, reset_staging, staging_path_for
    from checkpoint_paths import (
        CheckpointLayoutError,
        require_checkpoint_child_file,
        require_regular_checkpoint_tree,
    )
    from implementations import build_validation_edit_metadata, write_edit_metadata
    from streaming_transform_core import (
        _build_plan,
        _canonical_sha256,
        _change_stats,
        _ChangeStats,
        _file_sha256,
        _MaterializationPlan,
        _reject_ambiguous_weight_files,
        _require_output_shard_bytes,
        _safe_metadata,
        _ShardChunk,
        _weight_map,
        replay_transformation_tensor,
    )
    from transformation_contract import (
        SYNTHETIC_DENSE_UPDATE,
        SYNTHETIC_LOWRANK_DELTA,
        TransformationContractError,
        canonical_transformation_spec,
        checkpoint_transformation_contract,
        validate_transformation_scope,
        validate_transformation_scope_for_contract,
    )


TRANSFORMATION_MATERIALIZATION_SCHEMA = "invarlock/streaming-transformation-v1"
TRANSFORMATION_PROGRESS_SCHEMA = "invarlock/transformation-materialization-progress-v1"
TRANSFORMATION_MATERIALIZATION_RECEIPT_SCHEMA = (
    "invarlock/transformation-materialization-v1"
)
TRANSFORMATION_MATERIALIZATION_RECEIPT = "transformation_materialization.json"
TRANSFORMATION_PROGRESS_FILE = ".transformation-materialization-progress.json"
DEFAULT_OUTPUT_SHARD_BYTES = 1024 * 1024 * 1024

# All transformation math is intentionally evaluated on CPU.  Cross-device
# reduction and transcendental kernels need not have identical rounding, while
# a verifier must be able to reproduce stored bytes without owning the same
# accelerator as the original run.
CANONICAL_EXECUTION_POLICY = "cpu-float32-or-float64-v1"
_GENERATED_METADATA_FILES = {
    "edit_metadata.json",
    TRANSFORMATION_MATERIALIZATION_RECEIPT,
    TRANSFORMATION_PROGRESS_FILE,
}
_SHA256_PREFIX = "sha256:"


def _copy_support_files(
    baseline_path: Path,
    staging_path: Path,
    *,
    weight_paths: set[Path],
    index_path: Path | None,
) -> None:
    for source in sorted(baseline_path.rglob("*"), key=lambda item: item.as_posix()):
        relative = source.relative_to(baseline_path)
        if source.is_dir():
            (staging_path / relative).mkdir(parents=True, exist_ok=True)
            continue
        if source in weight_paths or source == index_path:
            continue
        if relative.as_posix() in _GENERATED_METADATA_FILES:
            continue
        destination = staging_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _storage_preflight(
    staging_parent: Path,
    *,
    output_weight_bytes: int,
    output_shards: int,
) -> None:
    usage = shutil.disk_usage(staging_parent)
    required_bytes = int(output_weight_bytes * 1.10) + 16 * 1024 * 1024
    if usage.free < required_bytes:
        raise RuntimeError(
            "insufficient free disk for resumable transformation materialization: "
            f"need at least {required_bytes} bytes, have {usage.free}"
        )
    try:
        available_inodes = os.statvfs(staging_parent).f_favail
    except OSError:  # pragma: no cover - platform-specific filesystem support
        return
    required_inodes = output_shards + 32
    if available_inodes and available_inodes < required_inodes:
        raise RuntimeError(
            "insufficient free inodes for transformation materialization: "
            f"need at least {required_inodes}, have {available_inodes}"
        )


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _progress_base(
    *,
    baseline_identity: dict[str, str],
    spec: Mapping[str, object],
    scope: str,
    contract: Any,
    plan: _MaterializationPlan,
) -> dict[str, object]:
    return {
        "schema": TRANSFORMATION_PROGRESS_SCHEMA,
        "baseline_identity": baseline_identity,
        "transformation": dict(spec),
        "scope": scope,
        "model_type": contract.model_type,
        "architecture": contract.architecture,
        "config_sha256": contract.config_sha256,
        "layer_count": contract.layer_count,
        "target_manifest_sha256": plan.target_manifest_sha256,
        "max_output_shard_bytes": plan.max_output_shard_bytes,
        "source_shard_plan_sha256": plan.source_shard_plan_sha256,
        "output_shard_plan_sha256": plan.output_shard_plan_sha256,
        "total_output_shards": len(plan.chunks),
        "selected_tensors": plan.selected_tensors,
        "selected_params": plan.selected_params,
        "total_tensors": plan.total_tensors,
        "total_params": plan.total_params,
        "completed_shards": [],
        "resume_count": 0,
    }


def _valid_digest(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(_SHA256_PREFIX):
        return False
    return len(value) == len(_SHA256_PREFIX) + 64 and all(
        character in "0123456789abcdef" for character in value[len(_SHA256_PREFIX) :]
    )


def _change_stats_from_payload(payload: object) -> _ChangeStats:
    if not isinstance(payload, Mapping) or set(payload) != set(
        _ChangeStats().as_dict()
    ):
        raise RuntimeError("transformation staged change statistics are malformed")
    values: dict[str, int] = {}
    for key in _ChangeStats().as_dict():
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("transformation staged change statistics are malformed")
        values[key] = value
    return _ChangeStats(**values)


def _completed_entries(progress: Mapping[str, object]) -> dict[str, dict[str, object]]:
    entries = progress.get("completed_shards")
    if not isinstance(entries, list):
        raise RuntimeError("transformation staging completion receipt is malformed")
    parsed: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "sha256", "stats"}:
            raise RuntimeError("transformation staging completion receipt is malformed")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in parsed:
            raise RuntimeError("transformation staging completion receipt is malformed")
        if not _valid_digest(entry.get("sha256")):
            raise RuntimeError("transformation staging completion digest is malformed")
        _change_stats_from_payload(entry.get("stats"))
        parsed[name] = entry
    return parsed


def _load_or_start_progress(
    *,
    staging_path: Path,
    baseline_path: Path,
    baseline_identity: dict[str, str],
    spec: Mapping[str, object],
    scope: str,
    contract: Any,
    plan: _MaterializationPlan,
    restart: bool,
) -> tuple[dict[str, object], bool]:
    expected = _progress_base(
        baseline_identity=baseline_identity,
        spec=spec,
        scope=scope,
        contract=contract,
        plan=plan,
    )
    if staging_path.is_symlink():
        raise RuntimeError("transformation staging directory must not be a symlink")
    if restart and staging_path.exists():
        reset_staging(staging_path)
    if not staging_path.exists():
        staging_path.mkdir(parents=True, exist_ok=False)
        _copy_support_files(
            baseline_path,
            staging_path,
            weight_paths=set(plan.weights.values()),
            index_path=plan.index_path,
        )
        _write_json_atomic(staging_path / TRANSFORMATION_PROGRESS_FILE, expected)
        return expected, False

    progress_path = staging_path / TRANSFORMATION_PROGRESS_FILE
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"transformation staging directory is not resumable; use --restart: {exc}"
        ) from exc
    if not isinstance(progress, dict):
        raise RuntimeError(
            "transformation staging progress is not an object; use --restart"
        )
    expected_keys = set(expected)
    if set(progress) != expected_keys:
        raise RuntimeError(
            "transformation staging progress fields are malformed; use --restart"
        )
    for field, expected_value in expected.items():
        if field in {"completed_shards", "resume_count"}:
            continue
        if progress.get(field) != expected_value:
            raise RuntimeError(
                f"transformation staging contract mismatch for {field}; use --restart"
            )
    _completed_entries(progress)
    resume_count = progress.get("resume_count")
    if (
        isinstance(resume_count, bool)
        or not isinstance(resume_count, int)
        or resume_count < 0
    ):
        raise RuntimeError(
            "transformation staging resume_count is malformed; use --restart"
        )
    progress["resume_count"] = resume_count + 1
    _write_json_atomic(progress_path, progress)
    return progress, True


def _validate_completed_chunks(
    *,
    staging_path: Path,
    plan: _MaterializationPlan,
    completed: Mapping[str, dict[str, object]],
) -> None:
    expected_names = {chunk.name for chunk in plan.chunks}
    if not set(completed) <= expected_names:
        raise RuntimeError(
            "transformation staging names unknown output shards; use --restart"
        )
    for chunk in plan.chunks:
        output_path = staging_path / chunk.name
        partial_path = staging_path / f".{chunk.name}.partial"
        if partial_path.exists() or partial_path.is_symlink():
            if partial_path.is_symlink() or not partial_path.is_file():
                raise RuntimeError(
                    "transformation staging contains an unsafe partial shard; use --restart"
                )
            partial_path.unlink()
        if chunk.name not in completed:
            if output_path.exists() or output_path.is_symlink():
                if output_path.is_symlink() or not output_path.is_file():
                    raise RuntimeError(
                        "transformation staging contains an unsafe unrecorded shard; use --restart"
                    )
                # A process may have been interrupted after atomically writing a
                # shard but before persisting its receipt.  It is safe to discard
                # that unrecorded output and recompute it from the immutable input.
                output_path.unlink()
            continue
        try:
            require_checkpoint_child_file(
                staging_path,
                chunk.name,
                label="transformation staged output shard",
            )
            with safe_open(str(output_path), framework="pt", device="cpu") as handle:
                if tuple(sorted(handle.keys())) != tuple(sorted(chunk.tensor_names)):
                    raise RuntimeError(
                        "transformation staged output shard keys do not match plan"
                    )
        except (CheckpointLayoutError, RuntimeError) as exc:
            raise RuntimeError(
                f"transformation staged output shard is invalid ({chunk.name}); use --restart"
            ) from exc
        if _file_sha256(output_path) != completed[chunk.name]["sha256"]:
            raise RuntimeError(
                f"transformation staged output shard digest changed ({chunk.name}); use --restart"
            )


def _materialize_chunk(
    *,
    handle: Any,
    chunk: _ShardChunk,
    staging_path: Path,
    target_names: frozenset[str],
    edit_type: str,
    parameters: Mapping[str, object],
) -> _ChangeStats:
    output_tensors: dict[str, torch.Tensor] = {}
    changes = _ChangeStats()
    for name in chunk.tensor_names:
        source_tensor = handle.get_tensor(name).contiguous()
        if name in target_names:
            output_tensor = replay_transformation_tensor(
                source_tensor,
                edit_type=edit_type,
                parameters=parameters,
            )
            changes = changes.plus(_change_stats(source_tensor, output_tensor))
        else:
            output_tensor = source_tensor
        output_tensors[name] = output_tensor.contiguous()

    temporary = staging_path / f".{chunk.name}.partial"
    destination = staging_path / chunk.name
    if (
        destination.exists()
        or destination.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise RuntimeError(
            "transformation staging output path is unexpectedly occupied"
        )
    try:
        save_file(output_tensors, temporary, metadata=_safe_metadata(handle))
        temporary.replace(destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        output_tensors.clear()
    return changes


def _aggregate_completed(completed: Mapping[str, dict[str, object]]) -> _ChangeStats:
    total = _ChangeStats()
    for entry in completed.values():
        total = total.plus(_change_stats_from_payload(entry["stats"]))
    return total


def _synthetic_provenance(
    *,
    edit_type: str,
    spec: Mapping[str, object],
) -> dict[str, object] | None:
    if edit_type == SYNTHETIC_LOWRANK_DELTA:
        return {
            "edit_family": SYNTHETIC_LOWRANK_DELTA,
            "edit_method": spec["algorithm"],
            "edit_count": 1,
            "dynamic_runtime_required": False,
            "synthetic": True,
            "trained_adapter": False,
            "adapter_merge_performed": False,
        }
    if edit_type == SYNTHETIC_DENSE_UPDATE:
        parameters = spec["parameters"]
        assert isinstance(parameters, Mapping)
        return {
            "edit_family": SYNTHETIC_DENSE_UPDATE,
            "edit_method": spec["algorithm"],
            "edit_count": int(parameters["iterations"]),
            "dynamic_runtime_required": False,
            "synthetic": True,
            "optimization_performed": False,
            "training_data_used": False,
        }
    return None


def _output_weight_identity(
    *,
    staging_path: Path,
    chunks: tuple[_ShardChunk, ...],
) -> dict[str, object]:
    index_path = staging_path / "model.safetensors.index.json"
    payload = {
        "index_sha256": _file_sha256(index_path),
        "shards": [
            {"name": chunk.name, "sha256": _file_sha256(staging_path / chunk.name)}
            for chunk in chunks
        ],
    }
    return {"sha256": _canonical_sha256(payload), **payload}


def _finalize_artifact(
    *,
    staging_path: Path,
    output_path: Path,
    baseline_path: Path,
    baseline_identity: dict[str, str],
    spec: Mapping[str, object],
    scope: str,
    contract: Any,
    plan: _MaterializationPlan,
    progress: dict[str, object],
) -> dict[str, Any]:
    completed = _completed_entries(progress)
    actual = _aggregate_completed(completed)
    if actual.value_changed_params <= 0 or actual.byte_changed_params <= 0:
        raise RuntimeError("transformation selected no effective parameter changes")
    if checkpoint_tree_sha256(baseline_path) != baseline_identity["sha256"]:
        raise RuntimeError(
            "baseline checkpoint changed during transformation materialization"
        )

    output_weight_map = {
        tensor_name: chunk.name
        for chunk in plan.chunks
        for tensor_name in chunk.tensor_names
    }
    index_path = staging_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": plan.total_weight_bytes},
                "weight_map": output_weight_map,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    parameters = spec["parameters"]
    assert isinstance(parameters, Mapping)
    coverage = {
        "edited_tensors": plan.selected_tensors,
        "edited_params": plan.selected_params,
        "total_params": plan.total_params,
        "coverage_ratio": plan.selected_params / plan.total_params,
    }
    metadata = build_validation_edit_metadata(
        edit_type=str(spec["edit_type"]),
        scope=scope,
        parameters=dict(parameters),
        coverage=coverage,
        edit_provenance=_synthetic_provenance(
            edit_type=str(spec["edit_type"]), spec=spec
        ),
        extra={
            "transformation_contract": dict(spec),
            "scope_policy": plan.target_manifest["scope_policy"],
            "model_type": contract.model_type,
            "transformation_architecture": contract.architecture,
            "config_sha256": contract.config_sha256,
            "layer_count": contract.layer_count,
            "target_manifest": plan.target_manifest,
            "target_manifest_sha256": plan.target_manifest_sha256,
            "max_output_shard_bytes": plan.max_output_shard_bytes,
            "source_shard_plan": plan.source_shard_plan,
            "source_shard_plan_sha256": plan.source_shard_plan_sha256,
            "output_shard_plan": plan.output_shard_plan,
            "output_shard_plan_sha256": plan.output_shard_plan_sha256,
            "selected_tensors": plan.selected_tensors,
            "selected_params": plan.selected_params,
            "actual_changes": actual.as_dict(),
            "materialization": "resumable_bounded_safetensors_v1",
            "execution_policy": CANONICAL_EXECUTION_POLICY,
        },
    )
    write_edit_metadata(staging_path / "edit_metadata.json", metadata)
    output_weights = _output_weight_identity(
        staging_path=staging_path, chunks=plan.chunks
    )
    receipt = {
        "schema": TRANSFORMATION_MATERIALIZATION_RECEIPT_SCHEMA,
        "ok": True,
        "baseline_identity": baseline_identity,
        "transformation": dict(spec),
        "scope": scope,
        "scope_policy": plan.target_manifest["scope_policy"],
        "model_type": contract.model_type,
        "architecture": contract.architecture,
        "config_sha256": contract.config_sha256,
        "layer_count": contract.layer_count,
        "target_manifest": plan.target_manifest,
        "target_manifest_sha256": plan.target_manifest_sha256,
        "max_output_shard_bytes": plan.max_output_shard_bytes,
        "source_shard_plan": plan.source_shard_plan,
        "source_shard_plan_sha256": plan.source_shard_plan_sha256,
        "output_shard_plan": plan.output_shard_plan,
        "output_shard_plan_sha256": plan.output_shard_plan_sha256,
        "output_weights": output_weights,
        "execution_policy": CANONICAL_EXECUTION_POLICY,
        "output_shards": len(plan.chunks),
        "resume_count": progress["resume_count"],
        "selected_tensors": plan.selected_tensors,
        "selected_params": plan.selected_params,
        "out_of_scope_tensors": plan.total_tensors - plan.selected_tensors,
        "out_of_scope_params": plan.total_params - plan.selected_params,
        "total_tensors": plan.total_tensors,
        "total_params": plan.total_params,
        "actual_changes": actual.as_dict(),
    }
    _write_json_atomic(staging_path / TRANSFORMATION_MATERIALIZATION_RECEIPT, receipt)
    (staging_path / TRANSFORMATION_PROGRESS_FILE).unlink(missing_ok=True)
    artifact_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": checkpoint_tree_sha256(staging_path),
    }
    replace_output(staging_path, output_path)
    return {
        "schema": TRANSFORMATION_MATERIALIZATION_SCHEMA,
        "ok": True,
        "baseline_identity": baseline_identity,
        "artifact_identity": artifact_identity,
        "transformation": dict(spec),
        "scope": scope,
        "model_type": contract.model_type,
        "architecture": contract.architecture,
        "config_sha256": contract.config_sha256,
        "layer_count": contract.layer_count,
        "target_manifest_sha256": plan.target_manifest_sha256,
        "max_output_shard_bytes": plan.max_output_shard_bytes,
        "source_shard_plan": plan.source_shard_plan,
        "source_shard_plan_sha256": plan.source_shard_plan_sha256,
        "output_shard_plan": plan.output_shard_plan,
        "output_shard_plan_sha256": plan.output_shard_plan_sha256,
        "output_weights": output_weights,
        "execution_policy": CANONICAL_EXECUTION_POLICY,
        "selected_tensors": plan.selected_tensors,
        "selected_params": plan.selected_params,
        "total_tensors": plan.total_tensors,
        "total_params": plan.total_params,
        "actual_changes": actual.as_dict(),
        "output_shards": len(plan.chunks),
        "resume_count": progress["resume_count"],
    }


def materialize_transformation_artifact(
    *,
    baseline_path: Path,
    output_path: Path,
    edit_type: str,
    parameters: Mapping[str, object],
    scope: str,
    max_output_shard_bytes: int = DEFAULT_OUTPUT_SHARD_BYTES,
    restart: bool = False,
) -> dict[str, Any]:
    """Atomically materialize one canonical verifier-grade transformation.

    The destination is never made visible until every output shard and both
    receipts have been persisted.  A terminated run can be resumed only when
    its baseline identity, exact canonical parameters, target manifest, and
    source/output shard plans all still match.
    """

    spec = canonical_transformation_spec(edit_type, parameters)
    normalized_scope = validate_transformation_scope(scope)
    max_output_shard_bytes = _require_output_shard_bytes(max_output_shard_bytes)
    try:
        require_regular_checkpoint_tree(baseline_path, label="baseline checkpoint")
    except CheckpointLayoutError as exc:
        raise TransformationContractError(str(exc)) from exc
    if output_path.is_symlink():
        raise RuntimeError("transformation output path must not be a symlink")
    contract = checkpoint_transformation_contract(baseline_path)
    normalized_scope = validate_transformation_scope_for_contract(
        normalized_scope, contract=contract
    )
    baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": checkpoint_tree_sha256(baseline_path),
    }
    weights, index_path = _weight_map(baseline_path)
    _reject_ambiguous_weight_files(
        baseline_path,
        weight_paths=set(weights.values()),
        index_path=index_path,
    )
    canonical_parameters = spec["parameters"]
    assert isinstance(canonical_parameters, Mapping)
    plan = _build_plan(
        baseline_path=baseline_path,
        weights=weights,
        index_path=index_path,
        edit_type=str(spec["edit_type"]),
        parameters=canonical_parameters,
        scope=normalized_scope,
        contract=contract,
        max_output_shard_bytes=max_output_shard_bytes,
    )
    staging_path = staging_path_for(output_path)
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    _storage_preflight(
        staging_path.parent,
        output_weight_bytes=plan.total_weight_bytes,
        output_shards=len(plan.chunks),
    )
    progress, resumed = _load_or_start_progress(
        staging_path=staging_path,
        baseline_path=baseline_path,
        baseline_identity=baseline_identity,
        spec=spec,
        scope=normalized_scope,
        contract=contract,
        plan=plan,
        restart=restart,
    )
    completed = _completed_entries(progress)
    _validate_completed_chunks(
        staging_path=staging_path,
        plan=plan,
        completed=completed,
    )

    chunks_by_source: dict[Path, list[_ShardChunk]] = defaultdict(list)
    for chunk in plan.chunks:
        if chunk.name not in completed:
            chunks_by_source[chunk.source_path].append(chunk)
    for source_path in sorted(chunks_by_source, key=lambda path: path.as_posix()):
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            for chunk in chunks_by_source[source_path]:
                changes = _materialize_chunk(
                    handle=handle,
                    chunk=chunk,
                    staging_path=staging_path,
                    target_names=plan.target_names,
                    edit_type=str(spec["edit_type"]),
                    parameters=canonical_parameters,
                )
                destination = staging_path / chunk.name
                completed[chunk.name] = {
                    "name": chunk.name,
                    "sha256": _file_sha256(destination),
                    "stats": changes.as_dict(),
                }
                progress["completed_shards"] = [
                    completed[name] for name in sorted(completed)
                ]
                _write_json_atomic(
                    staging_path / TRANSFORMATION_PROGRESS_FILE, progress
                )

    if len(completed) != len(plan.chunks):
        raise RuntimeError(
            "transformation materialization did not complete every output shard"
        )
    result = _finalize_artifact(
        staging_path=staging_path,
        output_path=output_path,
        baseline_path=baseline_path,
        baseline_identity=baseline_identity,
        spec=spec,
        scope=normalized_scope,
        contract=contract,
        plan=plan,
        progress=progress,
    )
    result["resumed"] = resumed
    return result


__all__ = [
    "CANONICAL_EXECUTION_POLICY",
    "DEFAULT_OUTPUT_SHARD_BYTES",
    "TRANSFORMATION_MATERIALIZATION_RECEIPT",
    "TRANSFORMATION_MATERIALIZATION_RECEIPT_SCHEMA",
    "TRANSFORMATION_MATERIALIZATION_SCHEMA",
    "TRANSFORMATION_PROGRESS_FILE",
    "TRANSFORMATION_PROGRESS_SCHEMA",
    "materialize_transformation_artifact",
    "replay_transformation_tensor",
]
