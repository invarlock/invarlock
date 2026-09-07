"""Closed contracts and bounded I/O for pipeline comparisons."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import math
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_EVIDENCE_BYTES = 192 * 1024 * 1024
MAX_RECORDS = 12000
_CANONICAL_BUFFER_BYTES = 65536


class PipelineError(ValueError):
    """Malformed, unsupported or contradictory pipeline evidence."""


def _fits_canonical_buffer(value: Any, remaining: int) -> bool:
    """Conservatively size a plain JSON subtree, stopping at the buffer bound."""
    ancestors: set[int] = set()
    stack: list[tuple[Iterator[Any], int | None]] = [(iter((value,)), None)]
    while stack:
        iterator, parent = stack[-1]
        try:
            item = next(iterator)
        except StopIteration:
            stack.pop()
            if parent is not None:
                ancestors.remove(parent)
            continue
        kind = type(item)
        if kind is str:
            # Control characters require six bytes; other Unicode uses at most
            # four UTF-8 bytes. Quotes add two bytes.
            remaining -= 6 * len(item) + 2
        elif item is None:
            remaining -= 4
        elif kind is bool:
            remaining -= 4 if item else 5
        elif kind is int:
            # 30103 / 100000 is an upper bound on log10(2).
            remaining -= max(1, (item.bit_length() * 30103) // 100000 + 1) + (item < 0)
        elif kind is float:
            if not math.isfinite(item):
                return False
            remaining -= 32
        elif kind in (dict, list):
            identity = id(item)
            if identity in ancestors:
                return False
            remaining -= 2 + len(item)
            if kind is dict:
                for key in item:
                    if type(key) is not str:
                        return False
                    remaining -= 6 * len(key) + 3
                    if remaining < 0:
                        return False
            if remaining < 0:
                return False
            ancestors.add(identity)
            stack.append((iter(item.values() if kind is dict else item), identity))
        else:
            return False
        if remaining < 0:
            return False
    return True


def _canonical_chunks(value: Any) -> Iterator[bytes]:
    """Encode bounded subtrees quickly and stream larger plain JSON containers.

    Composite buffers use a conservative 64 KiB bound. Large individual strings
    and unusual Python types retain the standard encoder's allocation behavior.
    """
    encoder = json.JSONEncoder(
        allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )

    def parts(item: Any) -> Iterator[tuple[bool, Any]]:
        if type(item) is dict:
            yield True, b"{"
            for index, key in enumerate(sorted(item)):
                if index:
                    yield True, b","
                yield True, encoder.encode(key).encode("utf-8")
                yield True, b":"
                yield False, item[key]
            yield True, b"}"
        else:
            yield True, b"["
            for index, child in enumerate(item):
                if index:
                    yield True, b","
                yield False, child
            yield True, b"]"

    active: set[int] = set()
    stack: list[tuple[Iterator[tuple[bool, Any]], int | None]] = [
        (iter(((False, value),)), None)
    ]
    try:
        while stack:
            iterator, parent = stack[-1]
            try:
                raw, item = next(iterator)
            except StopIteration:
                stack.pop()
                if parent is not None:
                    active.remove(parent)
                continue
            if raw:
                yield item
                continue
            if _fits_canonical_buffer(item, _CANONICAL_BUFFER_BYTES):
                yield encoder.encode(item).encode("utf-8")
                continue
            kind = type(item)
            if kind not in (dict, list) or (
                kind is dict and any(type(key) is not str for key in item)
            ):
                for chunk in encoder.iterencode(item):
                    yield chunk.encode("utf-8")
                continue
            identity = id(item)
            if identity in active:
                raise ValueError("Circular reference detected")
            active.add(identity)
            stack.append((parts(item), identity))
        yield b"\n"
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise PipelineError(f"value is not canonical JSON: {exc}") from exc


def digest(value: Any) -> str:
    hasher = hashlib.sha256()
    for chunk in _canonical_chunks(value):
        hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


@lru_cache(maxsize=6)
def _validator(name: str) -> Draft202012Validator:
    schema = parse_json_bytes(
        importlib.resources.files("invarlock")
        .joinpath("_data", "contracts", f"pipeline_{name}.schema.json")
        .read_bytes(),
        label="pipeline schema",
    )
    return Draft202012Validator(schema)


def validate(value: Any, name: str) -> None:
    try:
        limit = MAX_EVIDENCE_BYTES if name == "evidence" else MAX_INPUT_BYTES
        size = 0
        for chunk in _canonical_chunks(value):
            size += len(chunk)
            if size > limit:
                raise PipelineError(f"{name} exceeds the {limit} byte limit")
        error = next(_validator(name).iter_errors(value), None)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise PipelineError(f"invalid {name}: {exc}") from exc
    if error is not None:
        location = ".".join(str(p) for p in error.absolute_path)
        raise PipelineError(f"invalid {name} {location}: {error.message}")


def read_json(path: str | Path, *, max_bytes: int = MAX_INPUT_BYTES) -> Any:
    try:
        return parse_json_bytes(
            read_regular_file_bytes(
                Path(path), label="pipeline input", max_bytes=max_bytes
            ),
            label="pipeline input",
        )
    except (ValueError, OSError, RecursionError) as exc:
        raise PipelineError(str(exc)) from exc


def write_new(path: str | Path, payload: bytes) -> Path:
    """Publish an owner-readable new file without replacing user data."""
    import os
    import tempfile

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd, name = tempfile.mkstemp(dir=destination.parent, prefix=".pipeline-")
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    except OSError as exc:
        raise PipelineError(f"cannot create {destination}: {exc}") from exc
    return destination


def write_directory(path: Path, artifacts: dict[str, bytes]) -> None:
    """Publish a completed private tree using the core no-replace primitive."""
    import shutil
    import tempfile

    from invarlock.filesystem import publish_directory_no_replace

    path.parent.mkdir(parents=True, exist_ok=True)
    # Resolve the caller-selected parent, never an existing destination entry.
    destination = path.parent.resolve() / path.name
    staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".pipeline-"))
    try:
        for name, payload in artifacts.items():
            if Path(name).name != name or name in (".", ".."):
                raise PipelineError("artifact names must be plain file names")
            write_new(staging / name, payload)
        publish_directory_no_replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
