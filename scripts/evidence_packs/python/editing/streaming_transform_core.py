"""Canonical tensor replay and immutable materialization planning."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from invarlock.evidence_pack_json import StrictJsonError, load_json_object

try:
    from .checkpoint_paths import (
        CheckpointLayoutError,
        require_checkpoint_child_file,
    )
    from .transformation_contract import (
        QUANT_RTN,
        SYNTHETIC_DENSE_UPDATE,
        SYNTHETIC_LOWRANK_DELTA,
        TransformationContractError,
        canonical_transformation_spec,
        is_transformation_target,
        transformation_target_entry,
        transformation_target_manifest,
        transformation_target_manifest_sha256,
    )
except ImportError:  # pragma: no cover - direct script-path execution
    from checkpoint_paths import (
        CheckpointLayoutError,
        require_checkpoint_child_file,
    )
    from transformation_contract import (
        QUANT_RTN,
        SYNTHETIC_DENSE_UPDATE,
        SYNTHETIC_LOWRANK_DELTA,
        TransformationContractError,
        canonical_transformation_spec,
        is_transformation_target,
        transformation_target_entry,
        transformation_target_manifest,
        transformation_target_manifest_sha256,
    )

CANONICAL_EXECUTION_POLICY = "cpu-float32-or-float64-v1"
_ROW_CHUNK_SIZE = 256
_SHA256_PREFIX = "sha256:"
_REGULAR_FLOAT_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)


@dataclass(frozen=True)
class _SourceShard:
    path: Path
    relative_path: str
    sha256: str
    tensor_names: tuple[str, ...]
    byte_count: int


@dataclass(frozen=True)
class _ShardChunk:
    name: str
    source_path: Path
    source_relative_path: str
    source_sha256: str
    tensor_names: tuple[str, ...]
    byte_count: int


@dataclass(frozen=True)
class _MaterializationPlan:
    weights: dict[str, Path]
    index_path: Path | None
    source_shards: tuple[_SourceShard, ...]
    chunks: tuple[_ShardChunk, ...]
    target_names: frozenset[str]
    target_manifest: dict[str, object]
    target_manifest_sha256: str
    max_output_shard_bytes: int
    source_shard_plan: dict[str, object]
    source_shard_plan_sha256: str
    output_shard_plan: dict[str, object]
    output_shard_plan_sha256: str
    total_tensors: int
    total_params: int
    total_weight_bytes: int
    selected_tensors: int
    selected_params: int


@dataclass(frozen=True)
class _ChangeStats:
    value_changed_tensors: int = 0
    value_changed_params: int = 0
    byte_changed_tensors: int = 0
    byte_changed_params: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "value_changed_tensors": self.value_changed_tensors,
            "value_changed_params": self.value_changed_params,
            "byte_changed_tensors": self.byte_changed_tensors,
            "byte_changed_params": self.byte_changed_params,
        }

    def plus(self, other: _ChangeStats) -> _ChangeStats:
        return _ChangeStats(
            value_changed_tensors=(
                self.value_changed_tensors + other.value_changed_tensors
            ),
            value_changed_params=self.value_changed_params + other.value_changed_params,
            byte_changed_tensors=self.byte_changed_tensors + other.byte_changed_tensors,
            byte_changed_params=self.byte_changed_params + other.byte_changed_params,
        )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return _SHA256_PREFIX + digest.hexdigest()


def _safe_metadata(handle: Any) -> dict[str, str]:
    raw = handle.metadata()
    metadata = dict(raw) if isinstance(raw, dict) else {}
    metadata.setdefault("format", "pt")
    return {str(key): str(value) for key, value in metadata.items()}


def _tensor_byte_count(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _require_output_shard_bytes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1024 * 1024:
        raise ValueError("max_output_shard_bytes must be an integer of at least 1 MiB")
    return value


def _weight_map(checkpoint_dir: Path) -> tuple[dict[str, Path], Path | None]:
    single = checkpoint_dir / "model.safetensors"
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if single.is_file() and index_path.is_file():
        raise TransformationContractError(
            "checkpoint must not contain both model.safetensors and its shard index"
        )
    if single.is_file():
        try:
            require_checkpoint_child_file(
                checkpoint_dir, "model.safetensors", label="model.safetensors"
            )
        except CheckpointLayoutError as exc:
            raise TransformationContractError(str(exc)) from exc
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            keys = tuple(sorted(handle.keys()))
        if not keys:
            raise TransformationContractError("model.safetensors contains no tensors")
        return dict.fromkeys(keys, single), None

    if not index_path.is_file():
        raise TransformationContractError(
            "streaming transformation requires safetensors checkpoint weights"
        )
    try:
        require_checkpoint_child_file(
            checkpoint_dir,
            "model.safetensors.index.json",
            label="model.safetensors index",
        )
        payload = load_json_object(index_path, label="model.safetensors index")
    except (CheckpointLayoutError, OSError, StrictJsonError) as exc:
        raise TransformationContractError(
            f"invalid model.safetensors.index.json: {exc}"
        ) from exc
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise TransformationContractError(
            "model.safetensors.index.json has no weight_map"
        )
    resolved: dict[str, Path] = {}
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise TransformationContractError(
                "model.safetensors.index.json has an invalid tensor name"
            )
        if not isinstance(shard_name, str):
            raise TransformationContractError(
                "model.safetensors.index.json has a non-string shard path"
            )
        try:
            resolved[tensor_name] = require_checkpoint_child_file(
                checkpoint_dir,
                shard_name,
                label="model.safetensors index shard",
            )
        except CheckpointLayoutError as exc:
            raise TransformationContractError(str(exc)) from exc
    return resolved, index_path


def _reject_ambiguous_weight_files(
    checkpoint_dir: Path,
    *,
    weight_paths: set[Path],
    index_path: Path | None,
) -> None:
    """Reject sidecar weights that a loader could prefer over our output.

    A transformed checkpoint must have exactly one declared weight topology.
    Copying an unindexed safetensors file or a stale PyTorch checkpoint would
    leave an alternative set of parameters beside the replay-bound shards.
    """

    for candidate in checkpoint_dir.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(checkpoint_dir).as_posix()
        name_lower = candidate.name.lower()
        looks_like_weight = candidate.suffix.lower() == ".safetensors" or (
            candidate.suffix.lower() in {".bin", ".pt", ".pth"}
            and any(token in name_lower for token in ("model", "adapter", "checkpoint"))
        )
        if looks_like_weight and candidate not in weight_paths:
            raise TransformationContractError(
                "baseline has an unreferenced weight file: " + relative
            )
        if (
            candidate.name.endswith(".index.json")
            and any(token in candidate.name for token in ("model", "pytorch", "weight"))
            and candidate != index_path
        ):
            raise TransformationContractError(
                "baseline has an unreferenced weight index: " + relative
            )


def _chunk_source_tensors(
    source_path: Path,
    names: list[str],
    tensor_bytes: Mapping[str, int],
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


def _require_regular_floating_tensor(tensor: torch.Tensor) -> None:
    if tensor.dtype not in _REGULAR_FLOAT_DTYPES:
        raise TransformationContractError(
            "transformation targets must use float16, bfloat16, float32, or float64 storage"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise TransformationContractError("transformation input tensor is non-finite")


def _compute_dtype(tensor: torch.Tensor) -> torch.dtype:
    _require_regular_floating_tensor(tensor)
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _row_chunks(rows: int) -> range:
    return range(0, rows, _ROW_CHUNK_SIZE)


def _base_abs_mean(
    source_2d: torch.Tensor,
    *,
    rows: int,
) -> float:
    """Calculate a stable scale without allocating a full float copy."""

    total = 0.0
    for start in _row_chunks(rows):
        stop = min(start + _ROW_CHUNK_SIZE, rows)
        total += float(source_2d[start:stop].abs().sum(dtype=torch.float64).item())
    mean = total / source_2d.numel()
    return mean if math.isfinite(mean) and mean > 0.0 else 1.0


def _rtn_dequantized_tensor(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """Apply canonical groupwise RTN independently to each flattened row."""

    compute_dtype = _compute_dtype(tensor)
    original_shape = tensor.shape
    rows = int(original_shape[0])
    columns = int(math.prod(original_shape[1:]))
    if rows <= 0 or columns <= 0:
        raise TransformationContractError("transformation target shape is empty")
    qmin = -(2 ** (bits - 1))
    qmax = max((2 ** (bits - 1)) - 1, 1)
    effective_group = min(group_size, columns)
    groups = (columns + effective_group - 1) // effective_group
    padding = groups * effective_group - columns
    output = torch.empty_like(tensor, device="cpu")
    source_2d = tensor.reshape(rows, columns)
    output_2d = output.reshape(rows, columns)

    for start in _row_chunks(rows):
        stop = min(start + _ROW_CHUNK_SIZE, rows)
        flattened = source_2d[start:stop].to(dtype=compute_dtype)
        if padding:
            flattened = torch.nn.functional.pad(flattened, (0, padding))
        grouped = flattened.reshape(stop - start, groups, effective_group)
        max_abs = grouped.abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(max_abs / qmax, min=1e-10)
        quantized = torch.round(grouped / scale).clamp(qmin, qmax) * scale
        quantized = quantized.reshape(stop - start, groups * effective_group)
        if padding:
            quantized = quantized[:, :columns]
        output_2d[start:stop] = quantized.to(dtype=tensor.dtype)
    return output.contiguous()


def _lowrank_delta_tensor(
    tensor: torch.Tensor,
    *,
    rank: int,
    scale: float,
) -> torch.Tensor:
    """Apply the deterministic public low-rank delta in bounded row blocks."""

    compute_dtype = _compute_dtype(tensor)
    original_shape = tensor.shape
    rows = int(original_shape[0])
    columns = int(math.prod(original_shape[1:]))
    if rank > min(rows, columns):
        raise TransformationContractError(
            "synthetic low-rank rank exceeds a selected target's matrix rank"
        )
    source_2d = tensor.reshape(rows, columns)
    base_scale = _base_abs_mean(source_2d, rows=rows)
    basis = torch.arange(1, rank + 1, dtype=compute_dtype, device="cpu")
    cols = torch.arange(1, columns + 1, dtype=compute_dtype, device="cpu")
    right = torch.cos(basis[:, None] * cols[None, :] * 0.013)
    right = right / math.sqrt(columns)
    update_scale = (float(scale) / rank) * 0.001 * base_scale
    output = torch.empty_like(tensor, device="cpu")
    output_2d = output.reshape(rows, columns)

    for start in _row_chunks(rows):
        stop = min(start + _ROW_CHUNK_SIZE, rows)
        row_numbers = torch.arange(
            start + 1, stop + 1, dtype=compute_dtype, device="cpu"
        )
        left = torch.sin(row_numbers[:, None] * basis[None, :] * 0.017)
        left = left / math.sqrt(rows)
        # v1 fixes the accumulation order in the public ABI.  Do not replace
        # this with GEMM: mathematically equivalent matrix multiplication can
        # round differently for low-precision storage and would make exact
        # replay dependent on an opaque kernel choice.
        update = torch.zeros((stop - start, columns), dtype=compute_dtype)
        for component in range(rank):
            update = (
                update
                + left[:, component : component + 1] * right[component : component + 1]
            )
        update = update * update_scale
        result = source_2d[start:stop].to(dtype=compute_dtype) + update
        output_2d[start:stop] = result.to(dtype=tensor.dtype)
    if not bool(torch.isfinite(output).all().item()):
        raise TransformationContractError("synthetic low-rank output is non-finite")
    return output.contiguous()


def _dense_update_tensor(
    tensor: torch.Tensor,
    *,
    step_size: float,
    iterations: int,
) -> torch.Tensor:
    """Apply literal, deterministic, per-iteration dense synthetic updates.

    Every iteration writes back to the target storage dtype before the next
    iteration.  This is intentionally not a collapsed ``iterations * delta``
    shortcut: replaying N iterations has the same dtype-rounding behaviour as
    generation.  The public direction is dense (not an adapter or a rank-one
    substitute) and depends only on shape and the iteration number.
    """

    compute_dtype = _compute_dtype(tensor)
    original_shape = tensor.shape
    rows = int(original_shape[0])
    columns = int(math.prod(original_shape[1:]))
    source_2d = tensor.reshape(rows, columns)
    base_scale = _base_abs_mean(source_2d, rows=rows)
    columns_vector = torch.arange(1, columns + 1, dtype=compute_dtype, device="cpu")
    output = tensor.detach().clone().contiguous()
    output_2d = output.reshape(rows, columns)
    # The factor makes the documented small step sizes observable after a
    # float16/bfloat16 storage round-trip; it is a fixed part of this algorithm.
    update_magnitude = base_scale * float(step_size) * 100.0

    for iteration in range(1, iterations + 1):
        for start in _row_chunks(rows):
            stop = min(start + _ROW_CHUNK_SIZE, rows)
            row_numbers = torch.arange(
                start + 1, stop + 1, dtype=compute_dtype, device="cpu"
            )[:, None]
            # The multiplication term prevents this from degenerating to a
            # separable (low-rank) outer product.
            direction = torch.sin(
                row_numbers * columns_vector[None, :] * 0.00031
                + float(iteration) * 0.17
            ) * torch.cos(
                (row_numbers + columns_vector[None, :]) * 0.013
                - float(iteration) * 0.11
            )
            updated = output_2d[start:stop].to(dtype=compute_dtype) + (
                direction * update_magnitude
            )
            output_2d[start:stop] = updated.to(dtype=tensor.dtype)
    if not bool(torch.isfinite(output).all().item()):
        raise TransformationContractError("synthetic dense-update output is non-finite")
    return output.contiguous()


def replay_transformation_tensor(
    tensor: torch.Tensor,
    *,
    edit_type: str,
    parameters: Mapping[str, object],
) -> torch.Tensor:
    """Return the one canonical CPU result for a selected source tensor.

    This is a pure materializer function: it neither mutates ``tensor`` nor
    uses random state.  The installed verifier intentionally does *not* call
    it; it implements the v1 byte-level ABI independently.
    """

    spec = canonical_transformation_spec(edit_type, parameters)
    canonical_parameters = spec["parameters"]
    if not isinstance(canonical_parameters, Mapping):  # defensive narrowing
        raise AssertionError("canonical transformation parameters must be a mapping")
    source = tensor.detach().to(device="cpu").contiguous()
    if source.dim() < 2:
        raise TransformationContractError(
            "transformation targets must be matrices with at least two dimensions"
        )
    if spec["edit_type"] == QUANT_RTN:
        return _rtn_dequantized_tensor(
            source,
            bits=int(canonical_parameters["bits"]),
            group_size=int(canonical_parameters["group_size"]),
        )
    if spec["edit_type"] == SYNTHETIC_LOWRANK_DELTA:
        return _lowrank_delta_tensor(
            source,
            rank=int(canonical_parameters["rank"]),
            scale=float(canonical_parameters["scale"]),
        )
    if spec["edit_type"] == SYNTHETIC_DENSE_UPDATE:
        return _dense_update_tensor(
            source,
            step_size=float(canonical_parameters["step_size"]),
            iterations=int(canonical_parameters["iterations"]),
        )
    raise AssertionError(f"unsupported canonical transformation: {spec['edit_type']}")


def _change_stats(source: torch.Tensor, transformed: torch.Tensor) -> _ChangeStats:
    if source.shape != transformed.shape or source.dtype != transformed.dtype:
        raise RuntimeError("transformation changed selected tensor shape or dtype")
    source = source.detach().to(device="cpu").contiguous()
    transformed = transformed.detach().to(device="cpu").contiguous()
    value_changed_params = int(torch.ne(source, transformed).sum().item())
    source_bytes = source.view(torch.uint8).reshape(
        source.numel(), source.element_size()
    )
    output_bytes = transformed.view(torch.uint8).reshape(
        transformed.numel(), transformed.element_size()
    )
    byte_changed_params = int(
        torch.any(source_bytes != output_bytes, dim=1).sum().item()
    )
    return _ChangeStats(
        value_changed_tensors=int(value_changed_params > 0),
        value_changed_params=value_changed_params,
        byte_changed_tensors=int(byte_changed_params > 0),
        byte_changed_params=byte_changed_params,
    )


def _build_plan(
    *,
    baseline_path: Path,
    weights: dict[str, Path],
    index_path: Path | None,
    edit_type: str,
    parameters: Mapping[str, object],
    scope: str,
    contract: Any,
    max_output_shard_bytes: int,
) -> _MaterializationPlan:
    grouped: dict[Path, list[str]] = defaultdict(list)
    for tensor_name, shard_path in weights.items():
        grouped[shard_path].append(tensor_name)

    tensor_bytes: dict[str, int] = {}
    selected_entries: list[dict[str, object]] = []
    source_shards: list[_SourceShard] = []
    total_tensors = 0
    total_params = 0
    total_weight_bytes = 0
    selected_tensors = 0
    selected_params = 0

    for source_path in sorted(grouped, key=lambda path: path.as_posix()):
        names = sorted(grouped[source_path])
        relative_path = source_path.relative_to(baseline_path).as_posix()
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            actual_names = tuple(sorted(handle.keys()))
            if actual_names != tuple(names):
                raise TransformationContractError(
                    "safetensors shard keys do not exactly match the checkpoint index: "
                    f"{relative_path}"
                )
            for name in names:
                tensor = handle.get_tensor(name)
                tensor_bytes[name] = _tensor_byte_count(tensor)
                total_tensors += 1
                total_params += int(tensor.numel())
                total_weight_bytes += tensor_bytes[name]
                if torch.is_floating_point(tensor) and not bool(
                    torch.isfinite(tensor).all().item()
                ):
                    raise TransformationContractError(
                        f"transformation baseline tensor is non-finite: {name}"
                    )
                if not is_transformation_target(
                    name,
                    scope=scope,
                    contract=contract,
                    ndim=tensor.dim(),
                ):
                    continue
                _require_regular_floating_tensor(tensor)
                selected_entries.append(transformation_target_entry(name, tensor))
                selected_tensors += 1
                selected_params += int(tensor.numel())
        source_shards.append(
            _SourceShard(
                path=source_path,
                relative_path=relative_path,
                sha256=_file_sha256(source_path),
                tensor_names=tuple(names),
                byte_count=sum(tensor_bytes[name] for name in names),
            )
        )

    target_manifest = transformation_target_manifest(
        edit_type=edit_type,
        parameters=parameters,
        scope=scope,
        contract=contract,
        targets=selected_entries,
    )
    target_manifest_sha256 = transformation_target_manifest_sha256(target_manifest)
    if selected_tensors <= 0 or selected_params <= 0:
        raise TransformationContractError(
            f"transformation scope selected no effective floating matrices: {scope}"
        )

    source_shard_plan = {
        "source_shards": [
            {
                "path": shard.relative_path,
                "sha256": shard.sha256,
                "tensor_names": list(shard.tensor_names),
                "byte_count": shard.byte_count,
            }
            for shard in source_shards
        ]
    }
    source_shard_plan_sha256 = _canonical_sha256(source_shard_plan)
    source_by_path = {shard.path: shard for shard in source_shards}
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
            source_relative_path=source_by_path[source_path].relative_path,
            source_sha256=source_by_path[source_path].sha256,
            tensor_names=names,
            byte_count=byte_count,
        )
        for ordinal, (source_path, names, byte_count) in enumerate(chunk_specs, start=1)
    )
    output_shard_plan = {
        "source_shard_plan_sha256": source_shard_plan_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "chunks": [
            {
                "name": chunk.name,
                "source_path": chunk.source_relative_path,
                "source_sha256": chunk.source_sha256,
                "tensor_names": list(chunk.tensor_names),
                "byte_count": chunk.byte_count,
            }
            for chunk in chunks
        ],
    }
    output_shard_plan_sha256 = _canonical_sha256(output_shard_plan)
    return _MaterializationPlan(
        weights=weights,
        index_path=index_path,
        source_shards=tuple(source_shards),
        chunks=chunks,
        target_names=frozenset(str(entry["name"]) for entry in selected_entries),
        target_manifest=target_manifest,
        target_manifest_sha256=target_manifest_sha256,
        max_output_shard_bytes=max_output_shard_bytes,
        source_shard_plan=source_shard_plan,
        source_shard_plan_sha256=source_shard_plan_sha256,
        output_shard_plan=output_shard_plan,
        output_shard_plan_sha256=output_shard_plan_sha256,
        total_tensors=total_tensors,
        total_params=total_params,
        total_weight_bytes=total_weight_bytes,
        selected_tensors=selected_tensors,
        selected_params=selected_params,
    )
