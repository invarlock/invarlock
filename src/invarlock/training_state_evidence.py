"""Bounded-memory state identity and LoRA delta evidence primitives."""

from __future__ import annotations

import json
import math
import os
import stat
import struct
from collections.abc import Iterator, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

HASH_DOMAIN = b"invarlock-training-runtime-v1\0"
CHUNK_ELEMENTS = 1024 * 1024


class TrainingStateEvidenceError(ValueError):
    """Raised when exact bounded state evidence cannot be produced."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _update_part(hasher: Any, value: bytes) -> None:
    hasher.update(struct.pack(">Q", len(value)))
    hasher.update(value)


def tensor_byte_chunks(tensor: Any, *, torch: Any) -> Iterator[bytes]:
    """Yield exact logical tensor bytes with a fixed maximum staging size."""

    value = tensor.detach()
    if not value.is_contiguous():
        raise TrainingStateEvidenceError(
            "bounded tensor hashing requires contiguous training state tensors"
        )
    flat = value.view(-1)
    for offset in range(0, int(flat.numel()), CHUNK_ELEMENTS):
        chunk = flat[offset : offset + CHUNK_ELEMENTS].to(device="cpu").contiguous()
        try:
            yield bytes(chunk.numpy().tobytes(order="C"))
        except TypeError:
            yield bytes(chunk.view(torch.uint8).numpy().tobytes(order="C"))


def _update_tensor_bytes(hasher: Any, tensor: Any, *, torch: Any) -> None:
    hasher.update(struct.pack(">Q", int(tensor.numel()) * int(tensor.element_size())))
    for chunk in tensor_byte_chunks(tensor, torch=torch):
        hasher.update(chunk)


def tensor_state_sha256(state: Mapping[str, Any], *, torch: Any) -> str:
    """Hash sorted tensor names, layouts, and bytes without full CPU copies."""

    hasher = sha256(HASH_DOMAIN + b"tensor-state\0")
    for name in sorted(state):
        tensor = state[name]
        _update_part(hasher, name.encode("utf-8"))
        _update_part(hasher, str(tensor.dtype).encode("ascii"))
        _update_part(hasher, _canonical_json_bytes(list(tensor.shape)))
        _update_tensor_bytes(hasher, tensor, torch=torch)
    return "sha256:" + hasher.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _stat_snapshot(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_matches(
    *, pre: os.stat_result, opened: os.stat_result, relative: str
) -> None:
    if _stat_snapshot(pre) != _stat_snapshot(opened):
        raise TrainingStateEvidenceError(
            f"artifact tree entry changed before descriptor binding: {relative}"
        )


def directory_sha256(path: Path, *, exclude: frozenset[str] = frozenset()) -> str:
    """Hash a tree through no-follow descriptors and detect concurrent mutation."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    try:
        root_pre = path.stat(follow_symlinks=False)
        root_fd = os.open(path, directory_flags)
    except OSError as exc:
        raise TrainingStateEvidenceError(
            "artifact tree could not be opened safely"
        ) from exc
    hasher = sha256(HASH_DOMAIN + b"directory-tree\0")

    def walk(directory_fd: int, prefix: str) -> None:
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise TrainingStateEvidenceError(
                f"artifact directory could not be enumerated: {prefix or '.'}"
            ) from exc
        for name in names:
            relative = f"{prefix}/{name}" if prefix else name
            try:
                pre = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise TrainingStateEvidenceError(
                    f"artifact tree entry disappeared before hashing: {relative}"
                ) from exc
            if stat.S_ISLNK(pre.st_mode):
                raise TrainingStateEvidenceError(
                    f"artifact tree contains a symlink: {relative}"
                )
            if stat.S_ISDIR(pre.st_mode):
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise TrainingStateEvidenceError(
                        f"artifact directory could not be opened safely: {relative}"
                    ) from exc
                try:
                    _open_matches(pre=pre, opened=os.fstat(child_fd), relative=relative)
                    walk(child_fd, relative)
                    post = os.fstat(child_fd)
                    path_post = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False
                    )
                    if _stat_snapshot(pre) != _stat_snapshot(post) or _stat_identity(
                        post
                    ) != _stat_identity(path_post):
                        raise TrainingStateEvidenceError(
                            f"artifact directory changed during hashing: {relative}"
                        )
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(pre.st_mode):
                raise TrainingStateEvidenceError(
                    f"artifact tree contains a non-regular entry: {relative}"
                )
            if relative in exclude:
                continue
            try:
                file_fd = os.open(name, flags, dir_fd=directory_fd)
            except OSError as exc:
                raise TrainingStateEvidenceError(
                    f"artifact file could not be opened safely: {relative}"
                ) from exc
            try:
                opened = os.fstat(file_fd)
                _open_matches(pre=pre, opened=opened, relative=relative)
                _update_part(hasher, relative.encode("utf-8"))
                hasher.update(struct.pack(">Q", opened.st_size))
                while chunk := os.read(file_fd, 1024 * 1024):
                    hasher.update(chunk)
                post = os.fstat(file_fd)
                path_post = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if _stat_snapshot(opened) != _stat_snapshot(post) or _stat_identity(
                    post
                ) != _stat_identity(path_post):
                    raise TrainingStateEvidenceError(
                        f"artifact file changed during hashing: {relative}"
                    )
            finally:
                os.close(file_fd)

    try:
        opened_root = os.fstat(root_fd)
        _open_matches(pre=root_pre, opened=opened_root, relative=".")
        walk(root_fd, "")
        root_post = os.fstat(root_fd)
        path_post = path.stat(follow_symlinks=False)
        if _stat_snapshot(opened_root) != _stat_snapshot(root_post) or _stat_identity(
            root_post
        ) != _stat_identity(path_post):
            raise TrainingStateEvidenceError("artifact tree changed during hashing")
    finally:
        os.close(root_fd)
    return "sha256:" + hasher.hexdigest()


def tensor_content_sha256(tensor: Any, *, torch: Any) -> str:
    """Hash tensor bytes without materializing the whole tensor on the CPU."""

    hasher = sha256()
    for chunk in tensor_byte_chunks(tensor, torch=torch):
        hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def state_manifest(
    state: Mapping[str, Any], *, torch: Any
) -> dict[str, dict[str, Any]]:
    """Return exact per-tensor identities without retaining tensor values."""

    return {
        name: {
            "sha256": tensor_content_sha256(state[name], torch=torch),
            "dtype": str(state[name].dtype),
            "shape": list(state[name].shape),
            "numel": int(state[name].numel()),
        }
        for name in sorted(state)
    }


def state_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return "sha256:" + sha256(_canonical_json_bytes(dict(manifest))).hexdigest()


def require_state_manifest(
    state: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    torch: Any,
    label: str,
) -> str:
    observed = state_manifest(state, torch=torch)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))[:3]
        extra = sorted(set(observed) - set(expected))[:3]
        changed = sorted(
            name
            for name in set(expected) & set(observed)
            if expected[name] != observed[name]
        )[:3]
        raise TrainingStateEvidenceError(
            f"{label} changed; missing={missing}, extra={extra}, changed={changed}"
        )
    return state_manifest_sha256(observed)


def streaming_lora_delta_evidence(
    *,
    baseline_manifest: Mapping[str, Mapping[str, Any]],
    baseline_targets: Mapping[str, Any],
    after: Mapping[str, Any],
    torch: Any,
) -> tuple[str, int, float, set[str]]:
    """Prove exact merge scope and delta bytes with bounded CPU staging."""

    if set(baseline_manifest) != set(after):
        raise TrainingStateEvidenceError(
            "trained model state keys differ from the baseline"
        )
    delta_hasher = sha256(HASH_DOMAIN + b"tensor-state\0")
    changed: set[str] = set()
    max_abs_delta = 0.0
    for name in sorted(after):
        right = after[name]
        record = baseline_manifest[name]
        if str(right.dtype) != record["dtype"] or list(right.shape) != record["shape"]:
            raise TrainingStateEvidenceError(f"trained tensor layout changed: {name}")
        observed_sha = tensor_content_sha256(right, torch=torch)
        is_target = name in baseline_targets
        if not is_target and observed_sha != record["sha256"]:
            raise TrainingStateEvidenceError(
                f"LoRA merge changed an out-of-scope tensor: {name}"
            )
        _update_part(delta_hasher, name.encode("utf-8"))
        _update_part(delta_hasher, str(torch.float64).encode("ascii"))
        _update_part(delta_hasher, _canonical_json_bytes(list(right.shape)))
        delta_hasher.update(struct.pack(">Q", int(right.numel()) * 8))
        right_flat = right.detach().view(-1)
        baseline_flat = baseline_targets[name].detach().view(-1) if is_target else None
        tensor_max = 0.0
        for offset in range(0, int(right_flat.numel()), CHUNK_ELEMENTS):
            end = offset + CHUNK_ELEMENTS
            if baseline_flat is None:
                count = min(end, int(right_flat.numel())) - offset
                raw = bytes(count * 8)
            else:
                difference = right_flat[offset:end].to(
                    device="cpu", dtype=torch.float64
                ) - baseline_flat[offset:end].to(dtype=torch.float64)
                raw = bytes(difference.contiguous().numpy().tobytes(order="C"))
                if difference.numel():
                    tensor_max = max(tensor_max, float(difference.abs().max().item()))
            delta_hasher.update(raw)
        if observed_sha != record["sha256"]:
            if not math.isfinite(tensor_max) or tensor_max <= 0.0:
                raise TrainingStateEvidenceError(f"LoRA merge delta is invalid: {name}")
            changed.add(name)
            max_abs_delta = max(max_abs_delta, tensor_max)
    return "sha256:" + delta_hasher.hexdigest(), len(changed), max_abs_delta, changed


def full_delta_evidence(
    before: Mapping[str, Any], after: Mapping[str, Any], *, torch: Any
) -> tuple[str, int, float, set[str]]:
    """Compute exact full-state delta evidence for fixture-sized training."""

    if set(before) != set(after):
        raise TrainingStateEvidenceError(
            "trained model state keys differ from the baseline"
        )
    delta_hasher = sha256(HASH_DOMAIN + b"tensor-state\0")
    changed: set[str] = set()
    max_abs_delta = 0.0
    for name in sorted(before):
        left, right = before[name], after[name]
        if left.shape != right.shape:
            raise TrainingStateEvidenceError(f"trained tensor shape changed: {name}")
        difference = right.detach().cpu().to(torch.float64) - left.to(torch.float64)
        _update_part(delta_hasher, name.encode("utf-8"))
        _update_part(delta_hasher, str(difference.dtype).encode("ascii"))
        _update_part(delta_hasher, _canonical_json_bytes(list(difference.shape)))
        raw = difference.contiguous().view(torch.uint8).numpy().tobytes(order="C")
        _update_part(delta_hasher, bytes(raw))
        if difference.numel():
            observed = float(difference.abs().max().item())
            if not math.isfinite(observed):
                raise TrainingStateEvidenceError(f"non-finite training delta: {name}")
            if observed > 0.0:
                changed.add(name)
                max_abs_delta = max(max_abs_delta, observed)
    return "sha256:" + delta_hasher.hexdigest(), len(changed), max_abs_delta, changed


__all__ = [
    "TrainingStateEvidenceError",
    "directory_sha256",
    "full_delta_evidence",
    "require_state_manifest",
    "state_manifest",
    "state_manifest_sha256",
    "streaming_lora_delta_evidence",
    "tensor_content_sha256",
    "tensor_state_sha256",
]
