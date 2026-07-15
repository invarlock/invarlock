"""Deterministic identities for model inputs used by assurance reports."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REMOTE_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
CHECKPOINT_TREE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
LEGACY_MODEL_IDENTITY_FIELDS = (
    "revision",
    "model_revision",
    "model_checkpoint_tree_sha256",
)

_HASH_DOMAIN = b"invarlock-model-checkpoint-tree-v1\0"
_WEIGHT_SUFFIXES = frozenset(
    {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack"}
)
_CHECKPOINT_FILENAMES = frozenset(
    {
        "adapter_config.json",
        "added_tokens.json",
        "chat_template.jinja",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "quantization_config.json",
        "quantize_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)
_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {".cache", ".git", "__pycache__", "log", "logs", "reports", "runs"}
)
_EXCLUDED_FILENAMES = frozenset(
    {
        ".ds_store",
        ".model_id",
        "adapter_runtime_summary.json",
        "checkpoint_refs.json",
        "evidence.meta.json",
        "external_edit_summary.json",
        "fixture_summary.json",
        "model_summary.json",
        "run_summary.txt",
        "training_binding.json",
        "training_receipt.json",
    }
)
_SECURE_FD_TRAVERSAL_AVAILABLE = (
    os.open in os.supports_dir_fd
    and os.scandir in os.supports_fd
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
)


class CheckpointIdentityError(ValueError):
    """Raised when a checkpoint cannot produce an unambiguous identity."""


@dataclass(frozen=True)
class _StatIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _TreeSnapshot:
    root: _StatIdentity
    directories: tuple[tuple[str, _StatIdentity], ...]
    files: tuple[tuple[str, _StatIdentity], ...]


@dataclass(frozen=True)
class CheckpointObservation:
    """Ephemeral digest and filesystem identity captured by one secure traversal."""

    digest: str
    root: _StatIdentity
    directories: tuple[tuple[str, _StatIdentity], ...]
    files: tuple[tuple[str, _StatIdentity], ...]


def canonical_remote_revision(value: object) -> str | None:
    """Return a canonical immutable remote revision, if supplied."""

    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if REMOTE_REVISION_RE.fullmatch(normalized) else None


def canonical_checkpoint_tree_digest(value: object) -> str | None:
    """Return a canonical checkpoint-tree digest, if supplied."""

    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if CHECKPOINT_TREE_RE.fullmatch(normalized) else None


def _checkpoint_file(path: Path) -> bool:
    name = path.name.lower()
    if name in _EXCLUDED_FILENAMES or name.startswith(".tmp-"):
        return False
    if name in _CHECKPOINT_FILENAMES or path.suffix.lower() in _WEIGHT_SUFFIXES:
        return True
    if name.endswith(".index.json") and any(
        token in name for token in ("model", "pytorch", "safetensors", "weight")
    ):
        return True
    # Backend-specific checkpoints use additional configuration and lookup-table
    # filenames. Hash every remaining regular file under the checkpoint root;
    # only the explicit operational exclusions above are omitted.
    return True


def _stat_identity(value: os.stat_result) -> _StatIdentity:
    return _StatIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        size=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
        ctime_ns=int(value.st_ctime_ns),
    )


def _directory_open_flags() -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _file_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _scan_checkpoint_tree(root_fd: int) -> _TreeSnapshot:
    directories: list[tuple[str, _StatIdentity]] = []
    files: list[tuple[str, _StatIdentity]] = []

    def scan(directory_fd: int, prefix: tuple[str, ...]) -> None:
        try:
            with os.scandir(directory_fd) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            location = "/".join(prefix) or "."
            raise CheckpointIdentityError(
                f"checkpoint tree changed while scanning: {location}"
            ) from exc

        for entry in entries:
            relative_parts = (*prefix, entry.name)
            relative = "/".join(relative_parts)
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise CheckpointIdentityError(
                    f"checkpoint tree changed while scanning: {relative}"
                ) from exc
            identity = _stat_identity(entry_stat)
            if stat.S_ISLNK(identity.mode):
                raise CheckpointIdentityError(
                    f"checkpoint tree contains a symlink: {relative}"
                )
            if stat.S_ISDIR(identity.mode):
                if entry.name in _EXCLUDED_DIRECTORY_NAMES:
                    continue
                try:
                    child_fd = os.open(
                        entry.name,
                        _directory_open_flags(),
                        dir_fd=directory_fd,
                    )
                except OSError as exc:
                    raise CheckpointIdentityError(
                        f"checkpoint directory changed or could not be securely opened: {relative}"
                    ) from exc
                try:
                    opened_identity = _stat_identity(os.fstat(child_fd))
                    if opened_identity != identity:
                        raise CheckpointIdentityError(
                            f"checkpoint directory changed while scanning: {relative}"
                        )
                    directories.append((relative, identity))
                    scan(child_fd, relative_parts)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(identity.mode):
                raise CheckpointIdentityError(
                    f"checkpoint tree contains a non-regular entry: {relative}"
                )
            if _checkpoint_file(Path(entry.name)):
                files.append((relative, identity))

    root_identity = _stat_identity(os.fstat(root_fd))
    if not stat.S_ISDIR(root_identity.mode):
        raise CheckpointIdentityError("checkpoint root is not a regular directory")
    scan(root_fd, ())
    return _TreeSnapshot(
        root=root_identity,
        directories=tuple(directories),
        files=tuple(files),
    )


def _open_checkpoint_file(root_fd: int, relative: str) -> int:
    parts = relative.split("/")
    directory_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child_fd = os.open(part, _directory_open_flags(), dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        return os.open(parts[-1], _file_open_flags(), dir_fd=directory_fd)
    except OSError as exc:
        raise CheckpointIdentityError(
            f"checkpoint file changed or could not be securely opened: {relative}"
        ) from exc
    finally:
        os.close(directory_fd)


def _hash_checkpoint_file(
    digest: Any,
    *,
    root_fd: int,
    relative: str,
    expected: _StatIdentity,
) -> None:
    file_fd = _open_checkpoint_file(root_fd, relative)
    try:
        before = _stat_identity(os.fstat(file_fd))
        if not stat.S_ISREG(before.mode) or before != expected:
            raise CheckpointIdentityError(
                f"checkpoint file changed before hashing: {relative}"
            )
        relative_bytes = relative.encode("utf-8")
        digest.update(struct.pack(">Q", len(relative_bytes)))
        digest.update(relative_bytes)
        digest.update(struct.pack(">Q", before.size))
        bytes_read = 0
        while True:
            try:
                chunk = os.read(file_fd, 1024 * 1024)
            except OSError as exc:
                raise CheckpointIdentityError(
                    f"checkpoint file changed while hashing: {relative}"
                ) from exc
            if not chunk:
                break
            bytes_read += len(chunk)
            digest.update(chunk)
        after = _stat_identity(os.fstat(file_fd))
        if before != after or bytes_read != before.size:
            raise CheckpointIdentityError(
                f"checkpoint file changed while hashing: {relative}"
            )
    finally:
        os.close(file_fd)


def _root_path_matches_fd(root: Path, root_fd: int) -> bool:
    try:
        path_identity = _stat_identity(root.stat(follow_symlinks=False))
        fd_identity = _stat_identity(os.fstat(root_fd))
    except OSError:
        return False
    return (
        stat.S_ISDIR(path_identity.mode)
        and path_identity.device == fd_identity.device
        and path_identity.inode == fd_identity.inode
    )


def checkpoint_tree_observation(path: str | Path) -> CheckpointObservation:
    """Observe the explicit model checkpoint tree, excluding logs and caches.

    The identity covers model weights, shard indexes, configuration, tokenizer,
    processor, and adapter files. Evidence receipts and operational sidecars are
    deliberately outside this model-content identity. Filesystem stat tokens are
    process-local TOCTOU evidence and must not be serialized into public reports.
    """

    root = Path(path).expanduser().absolute()
    if not _SECURE_FD_TRAVERSAL_AVAILABLE:
        raise CheckpointIdentityError(
            "secure file-descriptor checkpoint traversal is unavailable on this platform"
        )
    try:
        root_lstat = root.stat(follow_symlinks=False)
    except OSError as exc:
        raise CheckpointIdentityError(
            f"local checkpoint is not a regular directory: {root}"
        ) from exc
    if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
        raise CheckpointIdentityError(
            f"local checkpoint is not a regular directory: {root}"
        )
    try:
        root_fd = os.open(root, _directory_open_flags())
    except OSError as exc:
        raise CheckpointIdentityError(
            f"local checkpoint could not be securely opened: {root}"
        ) from exc
    try:
        if not _root_path_matches_fd(root, root_fd):
            raise CheckpointIdentityError(
                f"local checkpoint root changed while opening: {root}"
            )
        before = _scan_checkpoint_tree(root_fd)
        if not before.files:
            raise CheckpointIdentityError(
                f"local checkpoint contains no checkpoint files: {root}"
            )

        digest = hashlib.sha256(_HASH_DOMAIN)
        for relative, expected in before.files:
            _hash_checkpoint_file(
                digest,
                root_fd=root_fd,
                relative=relative,
                expected=expected,
            )

        after = _scan_checkpoint_tree(root_fd)
        if before != after or not _root_path_matches_fd(root, root_fd):
            raise CheckpointIdentityError(
                f"checkpoint tree changed while hashing: {root}"
            )
        return CheckpointObservation(
            digest="sha256:" + digest.hexdigest(),
            root=after.root,
            directories=after.directories,
            files=after.files,
        )
    finally:
        os.close(root_fd)


def checkpoint_tree_sha256(path: str | Path) -> str:
    """Return the portable content digest from a secure tree observation."""

    return checkpoint_tree_observation(path).digest


def resolve_model_identity(
    model_id: str,
    *,
    revision: str | None,
    strict: bool,
    side: str,
) -> dict[str, str] | None:
    """Resolve a producer-side remote revision or local checkpoint identity."""

    local_path = Path(model_id).expanduser()
    if local_path.exists() or local_path.is_symlink():
        if revision is not None and str(revision).strip():
            raise CheckpointIdentityError(
                f"{side} local checkpoint cannot also declare a remote revision"
            )
        return {
            "kind": "local_checkpoint_tree",
            "sha256": checkpoint_tree_sha256(local_path),
        }

    normalized_revision = canonical_remote_revision(revision)
    if revision is not None and str(revision).strip() and normalized_revision is None:
        raise CheckpointIdentityError(
            f"{side} remote model revision must be 40-64 lowercase hexadecimal characters"
        )
    if strict and normalized_revision is None:
        raise CheckpointIdentityError(
            f"{side} remote model revision must be 40-64 lowercase hexadecimal characters"
        )
    if normalized_revision is None:
        return None
    return {"kind": "remote_revision", "revision": normalized_revision}


def validated_model_identity(value: object) -> dict[str, str] | None:
    """Validate a portable report identity without consulting the filesystem."""

    if not isinstance(value, Mapping) or (
        set(value) != {"kind", "revision"} and set(value) != {"kind", "sha256"}
    ):
        return None
    kind = value.get("kind")
    if kind == "remote_revision":
        revision = canonical_remote_revision(value.get("revision"))
        return {"kind": kind, "revision": revision} if revision else None
    if kind == "local_checkpoint_tree":
        digest = canonical_checkpoint_tree_digest(value.get("sha256"))
        return {"kind": kind, "sha256": digest} if digest else None
    return None


__all__ = [
    "CheckpointObservation",
    "CheckpointIdentityError",
    "LEGACY_MODEL_IDENTITY_FIELDS",
    "canonical_checkpoint_tree_digest",
    "canonical_remote_revision",
    "checkpoint_tree_sha256",
    "checkpoint_tree_observation",
    "resolve_model_identity",
    "validated_model_identity",
]
