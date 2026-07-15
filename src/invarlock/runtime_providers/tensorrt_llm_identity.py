"""Bounded identity reader for a closed TensorRT-LLM engine layout.

The accepted layout is deliberately narrower than every file TensorRT-LLM can
emit.  A bundle is one directory containing exactly ``config.json`` and the
contiguous ``rank0.engine`` ... ``rankN.engine`` files declared by
``pretrained_config.mapping.world_size``.  LoRA adapters, managed weights,
quantization sidecars, and nested directories are rejected until their loading
and completeness rules are represented explicitly here.

The reader does not import TensorRT-LLM.  It authenticates the byte tree and
the small amount of portable metadata needed by the runtime-provider receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from invarlock.core.runtime_provider.types import TensorRTLLMArtifactIdentity

_MAX_FILE_COUNT = 256
_MAX_DIRECTORY_COUNT = 32
_MAX_FILE_BYTES = 256 * 1024**3
_MAX_TOTAL_BYTES = 2 * 1024**4
_MAX_JSON_BYTES = 16 * 1024**2
_MAX_JSON_DEPTH = 64
_MAX_JSON_ITEMS = 1_000_000
_MAX_TREE_DEPTH = 8
_MAX_LOGICAL_PATH_BYTES = 4096
_MAX_WORLD_SIZE = 256
_HASH_CHUNK_BYTES = 1024 * 1024

_COMPUTE_CAPABILITY = re.compile(r"^(0|[1-9][0-9]?)\.(0|[1-9][0-9]?)$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_RANK_ENGINE = re.compile(r"^rank(0|[1-9][0-9]*)\.engine$")


class TensorRTLLMIdentityError(ValueError):
    """Raised when an engine bundle cannot support a secure identity."""


@dataclass(frozen=True)
class _FileRecord:
    logical_name: str
    byte_length: int
    initial_stat: os.stat_result


@dataclass(frozen=True)
class _HashedFile:
    logical_name: str
    byte_length: int
    sha256: str


@dataclass
class _TreeBudget:
    files: int = 0
    directories: int = 0
    total_bytes: int = 0


@dataclass
class _JsonBudget:
    items: int = 0


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _logical_name(parts: tuple[str, ...]) -> str:
    value = PurePosixPath(*parts).as_posix()
    if value in {"", ".", ".."} or value.startswith("/"):
        raise TensorRTLLMIdentityError("engine bundle entry name is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TensorRTLLMIdentityError(
            "engine bundle entry name is not valid UTF-8"
        ) from exc
    if len(encoded) > _MAX_LOGICAL_PATH_BYTES:
        raise TensorRTLLMIdentityError(
            "engine bundle entry name exceeds the configured bound"
        )
    return value


def _open_root_without_symlinks(path: str | os.PathLike[str]) -> int:
    try:
        supplied = Path(os.fspath(path))
        if ".." in supplied.parts:
            raise TensorRTLLMIdentityError(
                "engine bundle path must not contain traversal components"
            )
        absolute = Path(os.path.abspath(supplied))
    except TensorRTLLMIdentityError:
        raise
    except (TypeError, ValueError, OSError) as exc:
        raise TensorRTLLMIdentityError("engine bundle path is invalid") from exc
    if absolute.name in {"", ".", ".."}:
        raise TensorRTLLMIdentityError("engine bundle path must identify a directory")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise TensorRTLLMIdentityError(
            "secure nofollow bundle opening is unavailable on this platform"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute.anchor, flags)
    except OSError as exc:
        raise TensorRTLLMIdentityError(
            "engine bundle root cannot be opened safely"
        ) from exc
    try:
        for component in absolute.parts[1:]:
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
                opened = os.fstat(next_descriptor)
            except OSError as exc:
                raise TensorRTLLMIdentityError(
                    "engine bundle path contains a symlink or inaccessible directory"
                ) from exc
            if not stat.S_ISDIR(before.st_mode) or _stat_identity(
                before
            ) != _stat_identity(opened):
                os.close(next_descriptor)
                raise TensorRTLLMIdentityError(
                    "engine bundle directory changed while being opened"
                )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _collect_files(root_descriptor: int) -> tuple[_FileRecord, ...]:
    budget = _TreeBudget()
    casefold_names: set[str] = set()
    inode_names: dict[tuple[int, int], str] = {}
    records: list[_FileRecord] = []

    def visit(directory_descriptor: int, parts: tuple[str, ...]) -> None:
        budget.directories += 1
        if budget.directories > _MAX_DIRECTORY_COUNT:
            raise TensorRTLLMIdentityError(
                "engine bundle directory count exceeds the configured bound"
            )
        if len(parts) > _MAX_TREE_DEPTH:
            raise TensorRTLLMIdentityError(
                "engine bundle tree depth exceeds the configured bound"
            )
        try:
            entries = sorted(os.listdir(directory_descriptor))
        except OSError as exc:
            raise TensorRTLLMIdentityError(
                "engine bundle directory cannot be listed safely"
            ) from exc
        logical_entries = [(entry, _logical_name((*parts, entry))) for entry in entries]
        for _entry, logical_name in logical_entries:
            folded = logical_name.casefold()
            if folded in casefold_names:
                raise TensorRTLLMIdentityError(
                    f"engine bundle contains a casefold collision at {logical_name!r}"
                )
            casefold_names.add(folded)
        for entry, logical_name in logical_entries:
            try:
                before = os.stat(
                    entry,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise TensorRTLLMIdentityError(
                    f"engine bundle entry {logical_name!r} is unavailable"
                ) from exc
            if stat.S_ISLNK(before.st_mode):
                raise TensorRTLLMIdentityError(
                    f"engine bundle entry {logical_name!r} must not be a symlink"
                )
            if stat.S_ISDIR(before.st_mode):
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                )
                try:
                    child_descriptor = os.open(
                        entry,
                        flags,
                        dir_fd=directory_descriptor,
                    )
                    opened = os.fstat(child_descriptor)
                except OSError as exc:
                    raise TensorRTLLMIdentityError(
                        f"engine bundle directory {logical_name!r} cannot be opened safely"
                    ) from exc
                try:
                    if not stat.S_ISDIR(opened.st_mode) or _stat_identity(
                        before
                    ) != _stat_identity(opened):
                        raise TensorRTLLMIdentityError(
                            f"engine bundle directory {logical_name!r} changed while opening"
                        )
                    visit(child_descriptor, (*parts, entry))
                finally:
                    os.close(child_descriptor)
                continue
            if not stat.S_ISREG(before.st_mode):
                raise TensorRTLLMIdentityError(
                    f"engine bundle entry {logical_name!r} must be a regular file"
                )
            if before.st_nlink != 1:
                raise TensorRTLLMIdentityError(
                    f"engine bundle entry {logical_name!r} must not be hard-linked"
                )
            inode = (before.st_dev, before.st_ino)
            previous = inode_names.setdefault(inode, logical_name)
            if previous != logical_name:
                raise TensorRTLLMIdentityError(
                    f"engine bundle entry {logical_name!r} aliases another file"
                )
            if before.st_size < 0 or before.st_size > _MAX_FILE_BYTES:
                raise TensorRTLLMIdentityError(
                    f"engine bundle entry {logical_name!r} exceeds the file-size bound"
                )
            budget.files += 1
            budget.total_bytes += before.st_size
            if budget.files > _MAX_FILE_COUNT:
                raise TensorRTLLMIdentityError(
                    "engine bundle file count exceeds the configured bound"
                )
            if budget.total_bytes > _MAX_TOTAL_BYTES:
                raise TensorRTLLMIdentityError(
                    "engine bundle total bytes exceed the configured bound"
                )
            records.append(
                _FileRecord(
                    logical_name=logical_name,
                    byte_length=before.st_size,
                    initial_stat=before,
                )
            )

    visit(root_descriptor, ())
    return tuple(sorted(records, key=lambda record: record.logical_name))


def _same_authenticated_records(
    initial: tuple[_FileRecord, ...], current: tuple[_FileRecord, ...]
) -> bool:
    """Compare mutation-relevant metadata while ignoring read-driven atime."""

    return len(initial) == len(current) and all(
        before.logical_name == after.logical_name
        and before.byte_length == after.byte_length
        and _stat_identity(before.initial_stat) == _stat_identity(after.initial_stat)
        for before, after in zip(initial, current, strict=True)
    )


def _open_file_by_components(
    root_descriptor: int, logical_name: str
) -> tuple[int, int]:
    components = PurePosixPath(logical_name).parts
    directory_descriptor = os.dup(root_descriptor)
    try:
        directory_flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        for component in components[:-1]:
            try:
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise TensorRTLLMIdentityError(
                    f"engine bundle directory for {logical_name!r} changed"
                ) from exc
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        try:
            file_descriptor = os.open(
                components[-1],
                file_flags,
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            raise TensorRTLLMIdentityError(
                f"engine bundle entry {logical_name!r} cannot be opened safely"
            ) from exc
        return directory_descriptor, file_descriptor
    except Exception:
        os.close(directory_descriptor)
        raise


def _hash_file(root_descriptor: int, record: _FileRecord) -> _HashedFile:
    parent_descriptor, file_descriptor = _open_file_by_components(
        root_descriptor, record.logical_name
    )
    try:
        opened = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise TensorRTLLMIdentityError(
                f"engine bundle entry {record.logical_name!r} is not a stable regular file"
            )
        if _stat_identity(opened) != _stat_identity(record.initial_stat):
            raise TensorRTLLMIdentityError(
                f"engine bundle entry {record.logical_name!r} changed before hashing"
            )
        remaining = record.byte_length
        digest = hashlib.sha256()
        while remaining:
            chunk = os.read(file_descriptor, min(remaining, _HASH_CHUNK_BYTES))
            if not chunk:
                raise TensorRTLLMIdentityError(
                    f"engine bundle entry {record.logical_name!r} was truncated while hashing"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(file_descriptor, 1):
            raise TensorRTLLMIdentityError(
                f"engine bundle entry {record.logical_name!r} grew while hashing"
            )
        after = os.fstat(file_descriptor)
        try:
            named_after = os.stat(
                PurePosixPath(record.logical_name).name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise TensorRTLLMIdentityError(
                f"engine bundle entry {record.logical_name!r} changed after hashing"
            ) from exc
        if _stat_identity(after) != _stat_identity(opened) or _stat_identity(
            named_after
        ) != _stat_identity(opened):
            raise TensorRTLLMIdentityError(
                f"engine bundle entry {record.logical_name!r} changed while hashing"
            )
        return _HashedFile(
            logical_name=record.logical_name,
            byte_length=record.byte_length,
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(file_descriptor)
        os.close(parent_descriptor)


def _read_bounded_file(
    root_descriptor: int, record: _FileRecord, maximum: int
) -> bytes:
    if record.byte_length > maximum:
        raise TensorRTLLMIdentityError(
            f"engine bundle entry {record.logical_name!r} exceeds the JSON-size bound"
        )
    parent_descriptor, file_descriptor = _open_file_by_components(
        root_descriptor, record.logical_name
    )
    try:
        opened = os.fstat(file_descriptor)
        if _stat_identity(opened) != _stat_identity(record.initial_stat):
            raise TensorRTLLMIdentityError(
                f"engine bundle entry {record.logical_name!r} changed before parsing"
            )
        chunks: list[bytes] = []
        remaining = record.byte_length
        while remaining:
            chunk = os.read(file_descriptor, min(remaining, _HASH_CHUNK_BYTES))
            if not chunk:
                raise TensorRTLLMIdentityError(
                    f"engine bundle entry {record.logical_name!r} was truncated while parsing"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(file_descriptor)
        if os.read(file_descriptor, 1) or _stat_identity(after) != _stat_identity(
            opened
        ):
            raise TensorRTLLMIdentityError(
                f"engine bundle entry {record.logical_name!r} changed while parsing"
            )
        return b"".join(chunks)
    finally:
        os.close(file_descriptor)
        os.close(parent_descriptor)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TensorRTLLMIdentityError(
                "TensorRT-LLM config contains a duplicate JSON key"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise TensorRTLLMIdentityError(
        f"TensorRT-LLM config contains non-finite JSON number {value!r}"
    )


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise TensorRTLLMIdentityError(
            "TensorRT-LLM config contains a non-finite JSON number"
        )
    return parsed


def _validate_json_budget(
    value: object,
    *,
    budget: _JsonBudget,
    depth: int = 0,
) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise TensorRTLLMIdentityError(
            "TensorRT-LLM config depth exceeds the configured bound"
        )
    if isinstance(value, dict):
        budget.items += len(value)
        for key, child in value.items():
            budget.items += 1
            if not isinstance(key, str):
                raise TensorRTLLMIdentityError(
                    "TensorRT-LLM config object keys must be strings"
                )
            try:
                key.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise TensorRTLLMIdentityError(
                    "TensorRT-LLM config contains an invalid Unicode key"
                ) from exc
            _validate_json_budget(child, budget=budget, depth=depth + 1)
    elif isinstance(value, list):
        budget.items += len(value)
        for child in value:
            _validate_json_budget(child, budget=budget, depth=depth + 1)
    elif isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise TensorRTLLMIdentityError(
                "TensorRT-LLM config contains an invalid Unicode string"
            ) from exc
    elif isinstance(value, float) and not math.isfinite(value):
        raise TensorRTLLMIdentityError(
            "TensorRT-LLM config contains a non-finite JSON number"
        )
    if budget.items > _MAX_JSON_ITEMS:
        raise TensorRTLLMIdentityError(
            "TensorRT-LLM config item count exceeds the configured bound"
        )


def _parse_config(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TensorRTLLMIdentityError(
            "TensorRT-LLM config must be valid UTF-8"
        ) from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_finite_float,
        )
    except TensorRTLLMIdentityError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TensorRTLLMIdentityError(
            "TensorRT-LLM config is not strict JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise TensorRTLLMIdentityError("TensorRT-LLM config must be a JSON object")
    _validate_json_budget(decoded, budget=_JsonBudget())
    return decoded


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, UnicodeError, ValueError) as exc:
        raise TensorRTLLMIdentityError(
            "TensorRT-LLM metadata is not finite canonical JSON"
        ) from exc


def _require_nonempty_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
    ):
        raise TensorRTLLMIdentityError(
            f"TensorRT-LLM config field {field!r} must be bounded non-empty text"
        )
    if any(ord(character) < 32 for character in value):
        raise TensorRTLLMIdentityError(
            f"TensorRT-LLM config field {field!r} must be printable"
        )
    return value


def _validated_engine_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, int]:
    if set(config) != {"version", "pretrained_config", "build_config"}:
        raise TensorRTLLMIdentityError(
            "TensorRT-LLM config must contain exactly version, pretrained_config, and build_config"
        )
    version = _require_nonempty_text(config["version"], field="version")
    pretrained = config["pretrained_config"]
    build = config["build_config"]
    if not isinstance(pretrained, dict) or not pretrained:
        raise TensorRTLLMIdentityError(
            "TensorRT-LLM pretrained_config must be a non-empty object"
        )
    if not isinstance(build, dict) or not build:
        raise TensorRTLLMIdentityError(
            "TensorRT-LLM build_config must be a non-empty object"
        )
    _require_nonempty_text(pretrained.get("architecture"), field="architecture")
    _require_nonempty_text(pretrained.get("dtype"), field="dtype")
    mapping = pretrained.get("mapping")
    if not isinstance(mapping, dict):
        raise TensorRTLLMIdentityError(
            "TensorRT-LLM pretrained_config.mapping must be an object"
        )
    world_size = mapping.get("world_size")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or not 1 <= world_size <= _MAX_WORLD_SIZE
    ):
        raise TensorRTLLMIdentityError(
            "TensorRT-LLM mapping.world_size is outside the supported bound"
        )
    return pretrained, build, version, world_size


def _inventory_payload(files: tuple[_HashedFile, ...]) -> list[dict[str, object]]:
    return [
        {
            "byte_length": entry.byte_length,
            "logical_name": entry.logical_name,
            "sha256": entry.sha256,
        }
        for entry in files
    ]


def _tree_digest(inventory_bytes: bytes) -> str:
    """Bind ordered names, lengths, and per-file byte digests once."""

    return hashlib.sha256(
        b"invarlock/tensorrt-llm-engine-tree-v1\0" + inventory_bytes
    ).hexdigest()


def read_tensorrt_llm_artifact_identity(
    bundle_path: str | os.PathLike[str],
    *,
    target_compute_capability: str,
    tokenizer_metadata_sha256: str,
) -> TensorRTLLMArtifactIdentity:
    """Authenticate one closed-layout TensorRT-LLM engine bundle.

    ``target_compute_capability`` is an authenticated build/deployment input.
    Current TensorRT-LLM engine ``config.json`` files do not carry it, so the
    caller must obtain it from the pinned build contract.  It is included both
    in the typed identity and the derived engine-metadata digest.

    TensorRT-LLM engines consume a tokenizer outside this closed bundle.
    ``tokenizer_metadata_sha256`` authenticates the external tokenizer contract
    used to encode prompts and decode generated token IDs.
    """

    if _COMPUTE_CAPABILITY.fullmatch(target_compute_capability) is None:
        raise TensorRTLLMIdentityError(
            "target_compute_capability must use major.minor notation"
        )
    if _SHA256.fullmatch(tokenizer_metadata_sha256) is None:
        raise TensorRTLLMIdentityError(
            "tokenizer_metadata_sha256 must be a lowercase sha256 digest"
        )
    root_descriptor = _open_root_without_symlinks(bundle_path)
    try:
        records = _collect_files(root_descriptor)
        if not records:
            raise TensorRTLLMIdentityError("TensorRT-LLM engine bundle is empty")
        by_name = {record.logical_name: record for record in records}
        config_record = by_name.get("config.json")
        if config_record is None:
            raise TensorRTLLMIdentityError(
                "TensorRT-LLM engine bundle is missing 'config.json'"
            )
        config = _parse_config(
            _read_bounded_file(root_descriptor, config_record, _MAX_JSON_BYTES)
        )
        pretrained, build, version, world_size = _validated_engine_config(config)

        expected_names = {"config.json"} | {
            f"rank{rank}.engine" for rank in range(world_size)
        }
        actual_names = set(by_name)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            if missing:
                raise TensorRTLLMIdentityError(
                    f"TensorRT-LLM engine bundle is missing {missing[0]!r}"
                )
            raise TensorRTLLMIdentityError(
                f"TensorRT-LLM engine bundle contains unsupported entry {unexpected[0]!r}"
            )
        for name in actual_names:
            rank_match = _RANK_ENGINE.fullmatch(name)
            if name != "config.json" and rank_match is None:
                raise TensorRTLLMIdentityError(
                    f"TensorRT-LLM engine bundle entry {name!r} is not canonical"
                )
            if name.endswith(".engine") and by_name[name].byte_length == 0:
                raise TensorRTLLMIdentityError(
                    f"TensorRT-LLM engine file {name!r} must not be empty"
                )

        hashed = tuple(_hash_file(root_descriptor, record) for record in records)
        if not _same_authenticated_records(records, _collect_files(root_descriptor)):
            raise TensorRTLLMIdentityError(
                "TensorRT-LLM engine bundle changed while being authenticated"
            )

        inventory = _inventory_payload(hashed)
        inventory_bytes = _canonical_json(inventory)
        inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
        tree_sha256 = _tree_digest(inventory_bytes)
        builder_sha256 = hashlib.sha256(_canonical_json(build)).hexdigest()
        hashed_by_name = {entry.logical_name: entry for entry in hashed}
        rank_inventory = [
            {
                "byte_length": hashed_by_name[f"rank{rank}.engine"].byte_length,
                "rank": rank,
                "sha256": hashed_by_name[f"rank{rank}.engine"].sha256,
            }
            for rank in range(world_size)
        ]
        engine_metadata = {
            "config_version": version,
            "pretrained_config": pretrained,
            "rank_engines": rank_inventory,
            "target_compute_capability": target_compute_capability,
            "tokenizer_metadata_sha256": tokenizer_metadata_sha256,
        }
        engine_metadata_sha256 = hashlib.sha256(
            _canonical_json(engine_metadata)
        ).hexdigest()
        return TensorRTLLMArtifactIdentity(
            bundle_name=f"tensorrt-llm-sha256-{tree_sha256}",
            engine_bundle_tree_sha256=tree_sha256,
            file_inventory_sha256=inventory_sha256,
            builder_config_sha256=builder_sha256,
            tokenizer_metadata_sha256=tokenizer_metadata_sha256,
            engine_metadata_sha256=engine_metadata_sha256,
            target_compute_capability=target_compute_capability,
        )
    finally:
        os.close(root_descriptor)


__all__ = [
    "TensorRTLLMIdentityError",
    "read_tensorrt_llm_artifact_identity",
]
