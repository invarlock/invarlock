"""Resumable, architecture-aware materialization for dense pruning subjects."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.evidence_pack_json import StrictJsonError, read_json_object_snapshot
from invarlock.pruning_contract import (
    PRUNING_ALGORITHM,
    PRUNING_SCOPE_POLICY_VERSION,
    PRUNING_STORAGE_POLICY,
    PruningCheckpointContract,
    PruningContractError,
    checkpoint_pruning_contract,
    finite_pruning_sparsity,
    is_pruning_target,
    pruning_target_entry,
    pruning_target_manifest,
    pruning_target_manifest_sha256,
    validate_pruning_scope,
)

try:
    from .artifact_staging import replace_output, reset_staging, staging_path_for
    from .checkpoint_paths import (
        CheckpointLayoutError,
        require_checkpoint_child_file,
        require_regular_checkpoint_tree,
    )
    from .implementations import build_validation_edit_metadata, write_edit_metadata
    from .tensor_ops import magnitude_prune_tensor
    from .validate_artifact import validate_edit_artifact
except ImportError:  # pragma: no cover - direct script-path execution
    from artifact_staging import replace_output, reset_staging, staging_path_for
    from checkpoint_paths import (
        CheckpointLayoutError,
        require_checkpoint_child_file,
        require_regular_checkpoint_tree,
    )
    from implementations import build_validation_edit_metadata, write_edit_metadata
    from tensor_ops import magnitude_prune_tensor
    from validate_artifact import validate_edit_artifact


PRUNING_MATERIALIZATION_SCHEMA = "invarlock/streaming-magnitude-prune-v1"
PRUNING_PROGRESS_SCHEMA = "invarlock/pruning-materialization-progress-v1"
PRUNING_MATERIALIZATION_RECEIPT_SCHEMA = "invarlock/pruning-materialization-v1"
PRUNING_PROGRESS_FILE = ".pruning-materialization-progress.json"
PRUNING_MATERIALIZATION_RECEIPT = "pruning_materialization.json"
DEFAULT_OUTPUT_SHARD_BYTES = 1024 * 1024 * 1024
_GENERATED_METADATA_FILES = {
    "edit_metadata.json",
    PRUNING_MATERIALIZATION_RECEIPT,
    PRUNING_PROGRESS_FILE,
}


@dataclass(frozen=True)
class _ShardChunk:
    name: str
    source_path: Path
    tensor_names: tuple[str, ...]
    byte_count: int


@dataclass(frozen=True)
class _MaterializationPlan:
    weights: dict[str, Path]
    index_path: Path | None
    chunks: tuple[_ShardChunk, ...]
    target_names: frozenset[str]
    target_manifest: dict[str, object]
    target_manifest_sha256: str
    shard_plan_sha256: str
    total_params: int
    total_weight_bytes: int
    selected_tensors: int
    selected_params: int
    expected_pruned_params: int
    original_zero_params: int


@dataclass(frozen=True)
class _CompletedShardStats:
    observed_zero_params: int = 0
    effective_changed_params: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "observed_zero_params": self.observed_zero_params,
            "effective_changed_params": self.effective_changed_params,
        }

    def plus(self, other: _CompletedShardStats) -> _CompletedShardStats:
        return _CompletedShardStats(
            observed_zero_params=self.observed_zero_params + other.observed_zero_params,
            effective_changed_params=self.effective_changed_params
            + other.effective_changed_params,
        )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    return len(value) == len("sha256:") + 64 and all(
        character in "0123456789abcdef" for character in value[len("sha256:") :]
    )


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _safe_metadata(handle: Any) -> dict[str, str]:
    raw = handle.metadata()
    metadata = dict(raw) if isinstance(raw, dict) else {}
    metadata.setdefault("format", "pt")
    return {str(key): str(value) for key, value in metadata.items()}


def _tensor_byte_count(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _weight_map(checkpoint_dir: Path) -> tuple[dict[str, Path], Path | None]:
    single = checkpoint_dir / "model.safetensors"
    if single.is_file():
        require_checkpoint_child_file(
            checkpoint_dir, "model.safetensors", label="model.safetensors"
        )
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        if not keys:
            raise PruningContractError("model.safetensors contains no tensors")
        return dict.fromkeys(keys, single), None

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise PruningContractError(
            "streaming magnitude-prune requires safetensors checkpoint weights"
        )
    try:
        _, payload = read_json_object_snapshot(
            index_path, label="model.safetensors.index.json"
        )
    except StrictJsonError as exc:
        raise PruningContractError(
            f"invalid model.safetensors.index.json: {exc}"
        ) from exc
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise PruningContractError("model.safetensors.index.json has no weight_map")
    resolved: dict[str, Path] = {}
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
            raise PruningContractError(
                "model.safetensors.index.json has non-string entries"
            )
        try:
            resolved[tensor_name] = require_checkpoint_child_file(
                checkpoint_dir,
                shard_name,
                label="model.safetensors index shard",
            )
        except CheckpointLayoutError as exc:
            raise PruningContractError(str(exc)) from exc
    return resolved, index_path


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA pruning was requested but CUDA is unavailable")
    return device


def _require_output_shard_bytes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1024 * 1024:
        raise ValueError("max_output_shard_bytes must be an integer of at least 1 MiB")
    return value


def _chunk_source_tensors(
    source_path: Path,
    names: list[str],
    tensor_bytes: dict[str, int],
    *,
    max_output_shard_bytes: int,
) -> list[tuple[Path, tuple[str, ...], int]]:
    chunks: list[tuple[Path, tuple[str, ...], int]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = tensor_bytes[name]
        if current and current_bytes + size > max_output_shard_bytes:
            chunks.append((source_path, tuple(current), current_bytes))
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current:
        chunks.append((source_path, tuple(current), current_bytes))
    return chunks


def _build_plan(
    *,
    baseline_path: Path,
    weights: dict[str, Path],
    index_path: Path | None,
    scope: str,
    sparsity: float,
    contract: PruningCheckpointContract,
    max_output_shard_bytes: int,
) -> _MaterializationPlan:
    grouped: dict[Path, list[str]] = defaultdict(list)
    for tensor_name, shard_path in weights.items():
        grouped[shard_path].append(tensor_name)

    tensor_bytes: dict[str, int] = {}
    selected_entries: list[dict[str, object]] = []
    selected_tensors = 0
    selected_params = 0
    expected_pruned_params = 0
    original_zero_params = 0
    total_params = 0
    total_weight_bytes = 0

    for source_path in sorted(grouped, key=lambda path: path.as_posix()):
        names = sorted(grouped[source_path])
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                byte_count = _tensor_byte_count(tensor)
                tensor_bytes[name] = byte_count
                total_weight_bytes += byte_count
                total_params += int(tensor.numel())
                is_target = is_pruning_target(
                    name,
                    scope=scope,
                    contract=contract,
                    ndim=tensor.dim(),
                )
                if torch.is_floating_point(tensor) and not bool(
                    torch.isfinite(tensor).all().item()
                ):
                    raise PruningContractError(
                        f"magnitude-prune baseline tensor is non-finite: {name}"
                    )
                if not is_target:
                    continue
                if not torch.is_floating_point(tensor):
                    raise PruningContractError(
                        "magnitude-prune does not support non-floating target tensor: "
                        f"{name} ({tensor.dtype})"
                    )
                selected_entries.append(pruning_target_entry(name, tensor))
                selected_tensors += 1
                selected_params += int(tensor.numel())
                expected_pruned_params += int(tensor.numel() * sparsity)
                original_zero_params += int((tensor == 0).sum().item())

    target_manifest = pruning_target_manifest(
        scope=scope,
        contract=contract,
        targets=selected_entries,
    )
    target_manifest_sha256 = pruning_target_manifest_sha256(target_manifest)
    if selected_tensors <= 0 or selected_params <= 0 or expected_pruned_params <= 0:
        raise PruningContractError(
            f"magnitude-prune scope selected no effective floating matrices: {scope}"
        )

    chunk_specs: list[tuple[Path, tuple[str, ...], int]] = []
    for source_path in sorted(grouped, key=lambda path: path.as_posix()):
        chunk_specs.extend(
            _chunk_source_tensors(
                source_path,
                sorted(grouped[source_path]),
                tensor_bytes,
                max_output_shard_bytes=max_output_shard_bytes,
            )
        )
    chunks = tuple(
        _ShardChunk(
            name=f"model-{ordinal:05d}-of-{len(chunk_specs):05d}.safetensors",
            source_path=source_path,
            tensor_names=names,
            byte_count=byte_count,
        )
        for ordinal, (source_path, names, byte_count) in enumerate(chunk_specs, start=1)
    )
    plan_payload = {
        "scope": scope,
        "target_sparsity": sparsity,
        "scope_policy": PRUNING_SCOPE_POLICY_VERSION,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "target_manifest_sha256": target_manifest_sha256,
        "chunks": [
            {
                "name": chunk.name,
                "source": chunk.source_path.relative_to(baseline_path).as_posix(),
                "tensor_names": list(chunk.tensor_names),
                "byte_count": chunk.byte_count,
            }
            for chunk in chunks
        ],
    }
    return _MaterializationPlan(
        weights=weights,
        index_path=index_path,
        chunks=chunks,
        target_names=frozenset(entry["name"] for entry in selected_entries),
        target_manifest=target_manifest,
        target_manifest_sha256=target_manifest_sha256,
        shard_plan_sha256=_canonical_sha256(plan_payload),
        total_params=total_params,
        total_weight_bytes=total_weight_bytes,
        selected_tensors=selected_tensors,
        selected_params=selected_params,
        expected_pruned_params=expected_pruned_params,
        original_zero_params=original_zero_params,
    )


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
            "insufficient free disk for resumable pruning materialization: "
            f"need at least {required_bytes} bytes, have {usage.free}"
        )
    try:
        available_inodes = os.statvfs(staging_parent).f_favail
    except OSError:  # pragma: no cover - platform-specific filesystem support
        return
    required_inodes = output_shards + 32
    if available_inodes and available_inodes < required_inodes:
        raise RuntimeError(
            "insufficient free inodes for pruning materialization: "
            f"need at least {required_inodes}, have {available_inodes}"
        )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _progress_base(
    *,
    baseline_identity: dict[str, str],
    contract: PruningCheckpointContract,
    scope: str,
    sparsity: float,
    plan: _MaterializationPlan,
) -> dict[str, object]:
    return {
        "schema": PRUNING_PROGRESS_SCHEMA,
        "baseline_identity": baseline_identity,
        "model_type": contract.model_type,
        "architecture": contract.architecture,
        "config_sha256": contract.config_sha256,
        "scope": scope,
        "target_sparsity": sparsity,
        "scope_policy": PRUNING_SCOPE_POLICY_VERSION,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "target_manifest_sha256": plan.target_manifest_sha256,
        "shard_plan_sha256": plan.shard_plan_sha256,
        "total_output_shards": len(plan.chunks),
        "completed_shards": [],
        "resume_count": 0,
    }


def _completed_stats_from_payload(payload: object) -> _CompletedShardStats:
    expected_fields = set(_CompletedShardStats().as_dict())
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise RuntimeError("pruning staged shard statistics are malformed")
    values: dict[str, int] = {}
    for field in expected_fields:
        value = payload.get(field)
        if not _is_nonnegative_int(value):
            raise RuntimeError("pruning staged shard statistics are malformed")
        assert isinstance(value, int)
        values[field] = value
    return _CompletedShardStats(**values)


def _completed_entries(progress: dict[str, object]) -> dict[str, dict[str, object]]:
    raw_entries = progress.get("completed_shards")
    if not isinstance(raw_entries, list):
        raise RuntimeError("pruning staging completion receipts are malformed")
    parsed: dict[str, dict[str, object]] = {}
    expected_fields = {"name", "sha256", "byte_size", "stats"}
    for entry in raw_entries:
        if not isinstance(entry, dict) or set(entry) != expected_fields:
            raise RuntimeError("pruning staging completion receipts are malformed")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in parsed:
            raise RuntimeError("pruning staging completion receipts are malformed")
        if not _valid_sha256(entry.get("sha256")):
            raise RuntimeError("pruning staging completion digest is malformed")
        if not _is_nonnegative_int(entry.get("byte_size")):
            raise RuntimeError("pruning staging completion size is malformed")
        _completed_stats_from_payload(entry.get("stats"))
        parsed[name] = entry
    return parsed


def _completed_entry(
    *,
    name: str,
    output_path: Path,
    stats: _CompletedShardStats,
) -> dict[str, object]:
    return {
        "name": name,
        "sha256": _file_sha256(output_path),
        "byte_size": output_path.stat().st_size,
        "stats": stats.as_dict(),
    }


def _set_completed_entries(
    progress: dict[str, object], completed: dict[str, dict[str, object]]
) -> None:
    progress["completed_shards"] = [completed[name] for name in sorted(completed)]


def _aggregate_completed_stats(
    completed: dict[str, dict[str, object]],
) -> _CompletedShardStats:
    total = _CompletedShardStats()
    for entry in completed.values():
        total = total.plus(_completed_stats_from_payload(entry["stats"]))
    return total


def _load_or_start_progress(
    *,
    staging_path: Path,
    baseline_path: Path,
    baseline_identity: dict[str, str],
    contract: PruningCheckpointContract,
    scope: str,
    sparsity: float,
    plan: _MaterializationPlan,
    restart: bool,
) -> tuple[dict[str, object], bool]:
    expected = _progress_base(
        baseline_identity=baseline_identity,
        contract=contract,
        scope=scope,
        sparsity=sparsity,
        plan=plan,
    )
    if staging_path.is_symlink():
        raise RuntimeError("pruning staging directory must not be a symlink")
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
        _write_json_atomic(staging_path / PRUNING_PROGRESS_FILE, expected)
        return expected, False

    progress_path = staging_path / PRUNING_PROGRESS_FILE
    try:
        _, progress = read_json_object_snapshot(
            progress_path, label="pruning materialization progress"
        )
    except StrictJsonError as exc:
        raise RuntimeError(
            f"pruning staging directory is not resumable; use --restart: {exc}"
        ) from exc
    if not isinstance(progress, dict):
        raise RuntimeError("pruning staging progress is not an object; use --restart")
    if set(progress) != set(expected):
        raise RuntimeError(
            "pruning staging progress fields are malformed; use --restart"
        )
    for field, expected_value in expected.items():
        if field in {"completed_shards", "resume_count"}:
            continue
        if progress.get(field) != expected_value:
            raise RuntimeError(
                f"pruning staging contract mismatch for {field}; use --restart"
            )
    _completed_entries(progress)
    resume_count = progress.get("resume_count")
    if not _is_nonnegative_int(resume_count):
        raise RuntimeError("pruning staging resume_count is malformed; use --restart")
    assert isinstance(resume_count, int)
    progress["resume_count"] = resume_count + 1
    _write_json_atomic(progress_path, progress)
    return progress, True


def _validate_completed_chunks(
    *,
    staging_path: Path,
    plan: _MaterializationPlan,
    completed: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    valid_names = {chunk.name for chunk in plan.chunks}
    if not set(completed) <= valid_names:
        raise RuntimeError("pruning staging names unknown output shards; use --restart")
    recovered = dict(completed)
    for chunk in plan.chunks:
        output_path = staging_path / chunk.name
        partial_path = staging_path / f".{chunk.name}.partial"
        if partial_path.exists() or partial_path.is_symlink():
            if partial_path.is_symlink() or not partial_path.is_file():
                raise RuntimeError(
                    "pruning staging contains an unsafe partial shard; use --restart"
                )
            partial_path.unlink()
        if chunk.name not in recovered:
            if output_path.exists() or output_path.is_symlink():
                if output_path.is_symlink() or not output_path.is_file():
                    raise RuntimeError(
                        "pruning staging contains an unsafe unrecorded output shard; "
                        "use --restart"
                    )
                # A process can stop after replacing the shard but before the
                # receipt write.  The immutable baseline lets us discard and
                # reconstruct that unrecorded file safely.
                output_path.unlink()
            continue
        if not output_path.exists():
            # A missing file cannot satisfy its receipt.  Drop the stale
            # receipt so the chunk is materialized again from the baseline.
            recovered.pop(chunk.name)
            continue
        if output_path.is_symlink() or not output_path.is_file():
            raise RuntimeError(
                "pruning staging contains an unsafe completed output shard; "
                "use --restart"
            )
        entry = recovered[chunk.name]
        try:
            require_checkpoint_child_file(
                staging_path, chunk.name, label="pruning staged output shard"
            )
            if output_path.stat().st_size != entry["byte_size"]:
                raise RuntimeError("pruning staged output shard size changed")
            with safe_open(str(output_path), framework="pt", device="cpu") as handle:
                if tuple(sorted(handle.keys())) != tuple(sorted(chunk.tensor_names)):
                    raise RuntimeError(
                        "pruning staged output shard keys do not match plan"
                    )
            if _file_sha256(output_path) != entry["sha256"]:
                raise RuntimeError("pruning staged output shard digest changed")
        except (CheckpointLayoutError, OSError, RuntimeError, SafetensorError):
            # Do not trust a completed marker whose bytes no longer match its
            # durable receipt.  The output is disposable staging state, so
            # remove it and rebuild the exact chunk below.
            output_path.unlink()
            recovered.pop(chunk.name)
    return recovered


def _materialize_chunk(
    *,
    handle: Any,
    chunk: _ShardChunk,
    staging_path: Path,
    target_names: frozenset[str],
    sparsity: float,
    active_device: torch.device,
) -> _CompletedShardStats:
    output_tensors: dict[str, torch.Tensor] = {}
    observed_zeros = 0
    changed_params = 0
    for name in chunk.tensor_names:
        source_tensor = handle.get_tensor(name)
        if name in target_names:
            working = source_tensor.to(active_device)
            output_tensor = magnitude_prune_tensor(working, sparsity).to("cpu")
            observed_zeros += int((output_tensor == 0).sum().item())
            changed_params += int((output_tensor != source_tensor).sum().item())
            del working
        else:
            output_tensor = source_tensor.contiguous()
        output_tensors[name] = output_tensor.contiguous()

    temporary = staging_path / f".{chunk.name}.partial"
    destination = staging_path / chunk.name
    try:
        save_file(output_tensors, temporary, metadata=_safe_metadata(handle))
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
        output_tensors.clear()
    return _CompletedShardStats(
        observed_zero_params=observed_zeros,
        effective_changed_params=changed_params,
    )


def _finalize_artifact(
    *,
    staging_path: Path,
    output_path: Path,
    baseline_identity: dict[str, str],
    contract: PruningCheckpointContract,
    scope: str,
    sparsity: float,
    plan: _MaterializationPlan,
    progress: dict[str, object],
    active_device: torch.device,
) -> dict[str, Any]:
    completed = _completed_entries(progress)
    stats = _aggregate_completed_stats(completed)
    changed_params = stats.effective_changed_params
    observed_zero_params = stats.observed_zero_params
    if changed_params <= 0:
        raise RuntimeError("magnitude-prune selected no effective parameter changes")
    output_weight_map = {
        tensor_name: chunk.name
        for chunk in plan.chunks
        for tensor_name in chunk.tensor_names
    }
    (staging_path / "model.safetensors.index.json").write_text(
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
    coverage = {
        "edited_tensors": plan.selected_tensors,
        "edited_params": plan.selected_params,
        "total_params": plan.total_params,
        "coverage_ratio": plan.selected_params / plan.total_params,
    }
    metadata = build_validation_edit_metadata(
        edit_type="magnitude_prune",
        scope=scope,
        parameters={"target_sparsity": sparsity},
        coverage=coverage,
        extra={
            "target_sparsity": sparsity,
            "mask_sparsity": plan.expected_pruned_params / plan.selected_params,
            "actual_zero_fraction": observed_zero_params / plan.selected_params,
            "effective_changed_params": changed_params,
            "pruning_algorithm": PRUNING_ALGORITHM,
            "scope_policy": PRUNING_SCOPE_POLICY_VERSION,
            "storage_policy": PRUNING_STORAGE_POLICY,
            "model_type": contract.model_type,
            "pruning_architecture": contract.architecture,
            "config_sha256": contract.config_sha256,
            "target_manifest": plan.target_manifest,
            "target_manifest_sha256": plan.target_manifest_sha256,
            "shard_plan_sha256": plan.shard_plan_sha256,
            "materialization": "resumable_bounded_safetensors_v1",
        },
    )
    write_edit_metadata(staging_path / "edit_metadata.json", metadata)
    receipt = {
        "schema": PRUNING_MATERIALIZATION_RECEIPT_SCHEMA,
        "ok": True,
        "baseline_identity": baseline_identity,
        "scope": scope,
        "target_sparsity": sparsity,
        "scope_policy": PRUNING_SCOPE_POLICY_VERSION,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "model_type": contract.model_type,
        "architecture": contract.architecture,
        "config_sha256": contract.config_sha256,
        "target_manifest_sha256": plan.target_manifest_sha256,
        "shard_plan_sha256": plan.shard_plan_sha256,
        "output_shards": len(plan.chunks),
        "resume_count": progress["resume_count"],
        "selected_tensors": plan.selected_tensors,
        "selected_params": plan.selected_params,
        "expected_pruned_params": plan.expected_pruned_params,
        "original_zero_params": plan.original_zero_params,
        "observed_zero_params": observed_zero_params,
        "effective_changed_params": changed_params,
        "total_params": plan.total_params,
    }
    _write_json_atomic(staging_path / PRUNING_MATERIALIZATION_RECEIPT, receipt)
    progress_path = staging_path / PRUNING_PROGRESS_FILE
    progress_path.unlink(missing_ok=True)
    generic_validation = validate_edit_artifact(
        staging_path,
        require_metadata=True,
        expected_edit_type="magnitude_prune",
        expected_artifact_class="validation_subject_checkpoint",
    )
    if not generic_validation.ok:
        raise RuntimeError(
            "streaming pruning output failed generic validation: "
            + "; ".join(generic_validation.issues or [])
        )
    artifact_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": checkpoint_tree_sha256(staging_path),
    }
    replace_output(staging_path, output_path)
    return {
        "schema": PRUNING_MATERIALIZATION_SCHEMA,
        "ok": True,
        "baseline_identity": baseline_identity,
        "artifact_identity": artifact_identity,
        "scope": scope,
        "scope_policy": PRUNING_SCOPE_POLICY_VERSION,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "model_type": contract.model_type,
        "architecture": contract.architecture,
        "config_sha256": contract.config_sha256,
        "target_manifest_sha256": plan.target_manifest_sha256,
        "shard_plan_sha256": plan.shard_plan_sha256,
        "target_sparsity": sparsity,
        "device": str(active_device),
        "resume_count": progress["resume_count"],
        "selected_tensors": plan.selected_tensors,
        "selected_params": plan.selected_params,
        "expected_pruned_params": plan.expected_pruned_params,
        "original_zero_params": plan.original_zero_params,
        "observed_zero_params": observed_zero_params,
        "effective_changed_params": changed_params,
        "total_params": plan.total_params,
        "output_shards": len(plan.chunks),
    }


def materialize_magnitude_pruned_artifact(
    *,
    baseline_path: Path,
    output_path: Path,
    sparsity: float,
    scope: str,
    device: str = "auto",
    max_output_shard_bytes: int = DEFAULT_OUTPUT_SHARD_BYTES,
    restart: bool = False,
) -> dict[str, Any]:
    """Create a bounded, resumable, replayable magnitude-pruned checkpoint."""

    normalized_sparsity = finite_pruning_sparsity(sparsity)
    normalized_scope = validate_pruning_scope(scope)
    max_output_shard_bytes = _require_output_shard_bytes(max_output_shard_bytes)
    try:
        require_regular_checkpoint_tree(baseline_path, label="baseline checkpoint")
    except CheckpointLayoutError as exc:
        raise PruningContractError(str(exc)) from exc
    contract = checkpoint_pruning_contract(baseline_path)
    baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": checkpoint_tree_sha256(baseline_path),
    }
    weights, index_path = _weight_map(baseline_path)
    plan = _build_plan(
        baseline_path=baseline_path,
        weights=weights,
        index_path=index_path,
        scope=normalized_scope,
        sparsity=normalized_sparsity,
        contract=contract,
        max_output_shard_bytes=max_output_shard_bytes,
    )
    active_device = _resolve_device(device)
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
        contract=contract,
        scope=normalized_scope,
        sparsity=normalized_sparsity,
        plan=plan,
        restart=restart,
    )
    completed = _completed_entries(progress)
    recovered = _validate_completed_chunks(
        staging_path=staging_path,
        plan=plan,
        completed=completed,
    )
    if recovered != completed:
        completed = recovered
        _set_completed_entries(progress, completed)
        _write_json_atomic(staging_path / PRUNING_PROGRESS_FILE, progress)

    chunks_by_source: dict[Path, list[_ShardChunk]] = defaultdict(list)
    for chunk in plan.chunks:
        if chunk.name not in completed:
            chunks_by_source[chunk.source_path].append(chunk)
    for source_path in sorted(chunks_by_source, key=lambda path: path.as_posix()):
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            for chunk in chunks_by_source[source_path]:
                chunk_stats = _materialize_chunk(
                    handle=handle,
                    chunk=chunk,
                    staging_path=staging_path,
                    target_names=plan.target_names,
                    sparsity=normalized_sparsity,
                    active_device=active_device,
                )
                completed[chunk.name] = _completed_entry(
                    name=chunk.name,
                    output_path=staging_path / chunk.name,
                    stats=chunk_stats,
                )
                _set_completed_entries(progress, completed)
                _write_json_atomic(staging_path / PRUNING_PROGRESS_FILE, progress)

    if len(completed) != len(plan.chunks):
        raise RuntimeError(
            "pruning materialization did not complete every output shard"
        )
    result = _finalize_artifact(
        staging_path=staging_path,
        output_path=output_path,
        baseline_identity=baseline_identity,
        contract=contract,
        scope=normalized_scope,
        sparsity=normalized_sparsity,
        plan=plan,
        progress=progress,
        active_device=active_device,
    )
    result["resumed"] = resumed
    return result


__all__ = [
    "DEFAULT_OUTPUT_SHARD_BYTES",
    "PRUNING_MATERIALIZATION_RECEIPT",
    "PRUNING_MATERIALIZATION_RECEIPT_SCHEMA",
    "PRUNING_MATERIALIZATION_SCHEMA",
    "PRUNING_PROGRESS_FILE",
    "PRUNING_PROGRESS_SCHEMA",
    "materialize_magnitude_pruned_artifact",
]
