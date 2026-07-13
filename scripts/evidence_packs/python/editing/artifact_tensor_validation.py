"""Strict tensor topology, loading, and replay primitives for edit validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from invarlock.evidence_pack_json import StrictJsonError, read_json_object_snapshot
from invarlock.pruning_contract import (
    PruningCheckpointContract,
    is_pruning_target,
    pruning_target_entry,
)

from .checkpoint_paths import CheckpointLayoutError, require_checkpoint_child_file
from .validate_deployable import _load_json_object

try:
    from safetensors import SafetensorError, safe_open
except ImportError:  # pragma: no cover - optional at import time
    safe_open = None
    SafetensorError = RuntimeError

PRUNING_MATERIALIZATION_RECEIPT = "pruning_materialization.json"
_MAX_PRUNING_REPLAY_WORKERS = 8
_MAX_PRUNING_REPLAY_THREADS = 8


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _has_tokenizer(edit_path: Path) -> bool:
    return any(
        (edit_path / name).is_file()
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "special_tokens_map.json",
        )
    )


def _validate_safetensors(path: Path) -> bool:
    if safe_open is None:
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return any(True for _ in handle.keys())
    except Exception:
        return False


def _validate_index_shards(edit_path: Path, index_path: Path) -> bool:
    try:
        require_checkpoint_child_file(
            edit_path,
            index_path.name,
            label="checkpoint weight index",
        )
    except CheckpointLayoutError:
        return False
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False

    if not all(isinstance(name, str) and name for name in weight_map.values()):
        return False
    shard_names = sorted(set(weight_map.values()))
    if not shard_names:
        return False

    for shard_name in shard_names:
        try:
            shard_path = require_checkpoint_child_file(
                edit_path,
                shard_name,
                label="weight shard",
            )
        except CheckpointLayoutError:
            return False
        if shard_path.suffix == ".safetensors" and not _validate_safetensors(
            shard_path
        ):
            return False
    return True


def _has_valid_weights(edit_path: Path) -> bool:
    single_safe = edit_path / "model.safetensors"
    safe_index = edit_path / "model.safetensors.index.json"
    single_bin = edit_path / "pytorch_model.bin"
    bin_index = edit_path / "pytorch_model.bin.index.json"

    if single_safe.is_file():
        try:
            require_checkpoint_child_file(
                edit_path,
                "model.safetensors",
                label="model.safetensors",
            )
        except CheckpointLayoutError:
            return False
        return _validate_safetensors(single_safe)
    if safe_index.is_file():
        return _validate_index_shards(edit_path, safe_index)
    if single_bin.is_file():
        try:
            require_checkpoint_child_file(
                edit_path,
                "pytorch_model.bin",
                label="pytorch_model.bin",
            )
        except CheckpointLayoutError:
            return False
        return True
    if bin_index.is_file():
        return _validate_index_shards(edit_path, bin_index)
    return False


def _safetensor_weight_map(checkpoint: Path) -> tuple[dict[str, Path], list[str]]:
    issues: list[str] = []
    if safe_open is None:
        return {}, ["safetensors is required for transformation replay validation"]
    single_safe = checkpoint / "model.safetensors"
    safe_index = checkpoint / "model.safetensors.index.json"
    if single_safe.is_file():
        try:
            require_checkpoint_child_file(
                checkpoint,
                "model.safetensors",
                label="model.safetensors",
            )
            with safe_open(str(single_safe), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        except Exception as exc:  # pragma: no cover - backend-specific details
            return {}, [f"model.safetensors unreadable: {exc}"]
        return dict.fromkeys(keys, single_safe), []
    if safe_index.is_file():
        payload = _load_json_object(safe_index)
        if payload is None:
            return {}, ["model.safetensors.index.json missing or invalid"]
        raw_weight_map = payload.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            return {}, ["model.safetensors.index.json has no weight_map"]
        result: dict[str, Path] = {}
        for key, shard_name in raw_weight_map.items():
            if not isinstance(key, str) or not isinstance(shard_name, str):
                issues.append(
                    "model.safetensors.index.json contains non-string entries"
                )
                continue
            try:
                shard_path = require_checkpoint_child_file(
                    checkpoint,
                    shard_name,
                    label="safetensors shard",
                )
            except CheckpointLayoutError as exc:
                issues.append(str(exc))
                continue
            result[key] = shard_path
        return result, issues
    return {}, ["safetensors model weights are required for transformation replay"]


def _load_safetensor_tensor(path: Path, key: str) -> Any:
    if safe_open is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("safetensors is unavailable")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _verifier_exact_magnitude_prune_reference(weight: Any, sparsity: float) -> Any:
    """Replay canonical magnitude pruning without delegating to materialization.

    The materializer and verifier deliberately own separate implementations.
    This reference uses a threshold plus a cumulative tie rank in flattened
    tensor order, whereas materialization owns its own write-side primitive.
    Keeping the reference here means a materializer regression cannot turn its
    own output into verifier evidence merely by sharing a helper.
    """

    import torch

    if not isinstance(weight, torch.Tensor):
        raise TypeError("magnitude-pruning reference requires a torch tensor")
    if not torch.is_floating_point(weight):
        raise ValueError("magnitude-pruning reference requires a floating-point tensor")
    if not math.isfinite(sparsity) or not 0.0 <= sparsity <= 1.0:
        raise ValueError("magnitude-pruning reference sparsity must be in [0, 1]")
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("magnitude-pruning reference rejects non-finite tensors")

    flattened = weight.detach().reshape(-1)
    element_count = int(flattened.numel())
    if element_count <= 0:
        raise ValueError("magnitude-pruning reference requires a non-empty tensor")
    prune_count = int(element_count * sparsity)
    if prune_count <= 0:
        return weight.clone()
    if prune_count >= element_count:
        return torch.zeros_like(weight)

    magnitudes = flattened.abs()
    cutoff = torch.kthvalue(magnitudes, prune_count).values
    strictly_lower = magnitudes < cutoff
    tied_at_cutoff = magnitudes == cutoff
    lower_count = int(torch.count_nonzero(strictly_lower).item())
    tie_count = int(torch.count_nonzero(tied_at_cutoff).item())
    remaining_ties = prune_count - lower_count
    if remaining_ties < 0 or remaining_ties > tie_count:
        raise RuntimeError("magnitude-pruning reference tie accounting is invalid")

    if remaining_ties == 0:
        prune_mask = strictly_lower
    elif remaining_ties == tie_count:
        prune_mask = strictly_lower | tied_at_cutoff
    else:
        # ``cumsum`` gives one-based stable ranks in the original flattened
        # tensor order, so ties always prune their earliest entries first.
        tie_rank = torch.cumsum(tied_at_cutoff.to(dtype=torch.int64), dim=0)
        prune_mask = strictly_lower | (tied_at_cutoff & (tie_rank <= remaining_ties))
    keep_mask = (~prune_mask).reshape(weight.shape).to(dtype=weight.dtype)
    # Multiplication intentionally preserves the sign bit of an original
    # signed zero, matching the canonical tensor semantics byte-for-byte.
    return weight * keep_mask


@dataclass(frozen=True)
class _PruningReplayResult:
    key: str
    issue: str | None
    checked_tensors: int
    total_params: int
    selected_tensors: int
    selected_params: int
    expected_pruned_params: int
    expected_changed_params: int
    observed_changed_params: int
    original_zero_params: int
    observed_zero_params: int
    out_of_scope_tensors: int
    out_of_scope_bytes_checked: int
    selected_entry: dict[str, object] | None


def _pruning_replay_one_tensor(
    *,
    key: str,
    baseline_path: Path,
    artifact_path: Path,
    scope: str,
    sparsity: float,
    contract: PruningCheckpointContract,
) -> _PruningReplayResult:
    """Independently replay and compare one tensor with bounded live memory."""

    import torch

    baseline = _load_safetensor_tensor(baseline_path, key)
    artifact = _load_safetensor_tensor(artifact_path, key)
    common = {
        "key": key,
        "checked_tensors": 1,
        "total_params": int(baseline.numel()),
    }

    def result(*, issue: str | None = None, **values: Any) -> _PruningReplayResult:
        defaults: dict[str, Any] = {
            "selected_tensors": 0,
            "selected_params": 0,
            "expected_pruned_params": 0,
            "expected_changed_params": 0,
            "observed_changed_params": 0,
            "original_zero_params": 0,
            "observed_zero_params": 0,
            "out_of_scope_tensors": 0,
            "out_of_scope_bytes_checked": 0,
            "selected_entry": None,
        }
        defaults.update(values)
        return _PruningReplayResult(**cast(Any, {"issue": issue, **common, **defaults}))

    if baseline.shape != artifact.shape:
        return result(issue=f"{key}: tensor shape mismatch")
    if baseline.dtype != artifact.dtype:
        return result(issue=f"{key}: tensor dtype mismatch")
    if torch.is_floating_point(baseline) and not bool(
        torch.isfinite(baseline).all().item()
    ):
        return result(issue=f"{key}: baseline tensor contains non-finite values")

    is_target = is_pruning_target(
        key,
        scope=scope,
        contract=contract,
        ndim=baseline.dim(),
    )
    if not is_target:
        byte_count = int(baseline.numel() * baseline.element_size())
        issue = (
            None
            if _tensor_bytes_equal(artifact, baseline)
            else f"{key}: out-of-scope tensor changed"
        )
        return result(
            issue=issue,
            out_of_scope_tensors=1,
            out_of_scope_bytes_checked=byte_count,
        )
    if not torch.is_floating_point(baseline):
        return result(
            issue=f"{key}: pruning replay requires floating-point target tensor"
        )

    parameter_count = int(baseline.numel())
    expected = _verifier_exact_magnitude_prune_reference(baseline, sparsity)
    issue = (
        None
        if _tensor_bytes_equal(artifact, expected)
        else f"{key}: artifact does not match exact prune replay"
    )
    return result(
        issue=issue,
        selected_tensors=1,
        selected_params=parameter_count,
        expected_pruned_params=int(parameter_count * sparsity),
        expected_changed_params=int((expected != baseline).sum().item()),
        observed_changed_params=int((artifact != baseline).sum().item()),
        original_zero_params=int((baseline == 0).sum().item()),
        observed_zero_params=int((artifact == 0).sum().item()),
        selected_entry=pruning_target_entry(key, baseline),
    )


def _bounded_pruning_replay_settings(
    *, workers: int, worker_threads: int
) -> tuple[int, int]:
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise ValueError("pruning replay workers must be an integer")
    if workers < 1 or workers > _MAX_PRUNING_REPLAY_WORKERS:
        raise ValueError(
            f"pruning replay workers must be between 1 and {_MAX_PRUNING_REPLAY_WORKERS}"
        )
    if isinstance(worker_threads, bool) or not isinstance(worker_threads, int):
        raise ValueError("pruning replay worker threads must be an integer")
    if worker_threads < 0 or worker_threads > _MAX_PRUNING_REPLAY_THREADS:
        raise ValueError(
            "pruning replay worker threads must be between 0 and "
            f"{_MAX_PRUNING_REPLAY_THREADS}"
        )
    return workers, worker_threads


def _tensor_bytes_equal(left: Any, right: Any) -> bool:
    import torch

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.detach().contiguous().view(torch.uint8)
    right_bytes = right.detach().contiguous().view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def _support_file_digests(
    checkpoint: Path,
    *,
    weight_paths: set[Path],
    generated_files: frozenset[str] = frozenset(
        {
            "edit_metadata.json",
            "model.safetensors.index.json",
            PRUNING_MATERIALIZATION_RECEIPT,
        }
    ),
) -> tuple[dict[str, str], list[str]]:
    digests: dict[str, str] = {}
    issues: list[str] = []
    for path in sorted(checkpoint.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(checkpoint).as_posix()
        if path.is_symlink():
            issues.append(f"support tree contains symlink: {relative}")
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            issues.append(f"support tree contains non-file entry: {relative}")
            continue
        if path in weight_paths or relative in generated_files:
            continue
        digests[relative] = _file_sha256(path)
    return digests, issues


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _strict_json_object(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Read a receipt/metadata object without JSON ambiguity.

    A duplicate field in a verifier receipt is an ambiguity, not a harmless
    parser detail. Transformation replay therefore uses this reader for every
    input it binds.
    """

    try:
        _, payload = read_json_object_snapshot(path, label=label)
    except StrictJsonError as exc:
        return None, [f"{label} is invalid: {exc}"]
    return payload, []


def _strict_safetensor_weight_map(
    checkpoint: Path,
    *,
    label: str,
) -> tuple[dict[str, Path], Path | None, list[str]]:
    """Return an unambiguous, complete safetensors topology.

    Replay verifies a checkpoint as a closed tree: every safetensors file must
    be declared by the sole model topology and every declared shard must expose
    exactly the indexed tensor keys.  This prevents an artifact from keeping a
    convenient but unbound alternate weight file beside its claimed output.
    """

    if safe_open is None:
        return {}, None, ["safetensors is required for transformation replay"]

    issues: list[str] = []
    single_path = checkpoint / "model.safetensors"
    index_path = checkpoint / "model.safetensors.index.json"
    has_single = single_path.exists()
    has_index = index_path.exists()
    if has_single and has_index:
        return {}, None, [f"{label} has both model.safetensors and its index"]
    if not has_single and not has_index:
        return {}, None, [f"{label} requires a safetensors weight topology"]

    result: dict[str, Path] = {}
    declared_paths: set[Path] = set()
    resolved_index: Path | None = None
    if has_single:
        try:
            resolved_single = require_checkpoint_child_file(
                checkpoint,
                "model.safetensors",
                label=f"{label} model.safetensors",
            )
            with safe_open(
                str(resolved_single), framework="pt", device="cpu"
            ) as handle:
                names = tuple(sorted(handle.keys()))
        except (CheckpointLayoutError, OSError, RuntimeError, SafetensorError) as exc:
            return {}, None, [f"{label} model.safetensors is unreadable: {exc}"]
        if not names:
            return {}, None, [f"{label} model.safetensors has no tensors"]
        result = dict.fromkeys(names, resolved_single)
        declared_paths.add(resolved_single)
    else:
        try:
            resolved_index = require_checkpoint_child_file(
                checkpoint,
                "model.safetensors.index.json",
                label=f"{label} model.safetensors index",
            )
        except CheckpointLayoutError as exc:
            return {}, None, [str(exc)]
        payload, index_issues = _strict_json_object(
            resolved_index,
            label=f"{label} model.safetensors index",
        )
        if index_issues:
            return {}, None, index_issues
        assert payload is not None
        raw_map = payload.get("weight_map")
        if not isinstance(raw_map, dict) or not raw_map:
            return {}, None, [f"{label} index has no weight_map"]
        for name, shard_name in raw_map.items():
            if not isinstance(name, str) or not name:
                issues.append(f"{label} index contains an invalid tensor name")
                continue
            if not isinstance(shard_name, str):
                issues.append(f"{label} index contains a non-string shard path")
                continue
            try:
                shard_path = require_checkpoint_child_file(
                    checkpoint,
                    shard_name,
                    label=f"{label} index shard",
                )
            except CheckpointLayoutError as exc:
                issues.append(str(exc))
                continue
            if shard_path.suffix != ".safetensors":
                issues.append(f"{label} index shard is not a safetensors file")
                continue
            result[name] = shard_path
            declared_paths.add(shard_path)
        if not result:
            issues.append(f"{label} index resolves no safetensors tensors")

    by_shard: dict[Path, list[str]] = defaultdict(list)
    for name, path in result.items():
        by_shard[path].append(name)
    for shard_path, expected_names in sorted(
        by_shard.items(), key=lambda item: item[0].as_posix()
    ):
        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                actual_names = tuple(sorted(handle.keys()))
        except (OSError, RuntimeError, SafetensorError) as exc:
            issues.append(f"{label} safetensors shard is unreadable: {exc}")
            continue
        if actual_names != tuple(sorted(expected_names)):
            issues.append(
                f"{label} shard keys do not exactly match its declared topology"
            )

    for candidate in sorted(checkpoint.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_dir():
            continue
        relative = candidate.relative_to(checkpoint).as_posix()
        if candidate.suffix == ".safetensors" and candidate not in declared_paths:
            issues.append(f"{label} has an unreferenced safetensors file: {relative}")
        elif candidate.suffix.lower() in {".bin", ".pt", ".pth"}:
            issues.append(f"{label} has a non-safetensors weight candidate: {relative}")
        elif candidate.name.endswith(".index.json") and candidate != resolved_index:
            issues.append(f"{label} has an unreferenced weight index: {relative}")
    return result, resolved_index, issues
