"""Fail-closed JSON loading for package-native evidence-pack verification.

Evidence packs are adversarial inputs at verification time.  Python's default
``json`` decoder silently accepts duplicate object members and non-standard
numeric constants, which makes a signed byte stream ambiguous to different
readers.  This module provides the small, dependency-free reader shared by
the package verifier and its immutable snapshot boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any


class StrictJsonError(ValueError):
    """Raised when JSON evidence is ambiguous, non-standard, or unsafe."""


def _no_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise StrictJsonError(f"JSON object has duplicate key {key!r}")
        payload[key] = value
    return payload


def _reject_nonstandard_json_constant(value: str) -> Any:
    raise StrictJsonError(f"JSON contains non-standard constant {value!r}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise StrictJsonError(f"JSON contains non-finite number {value!r}")
    return parsed


def _file_identity(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return the stable regular-file identity fields needed around a read."""

    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _regular_file_stat(path: Path, *, label: str) -> os.stat_result:
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise StrictJsonError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(file_stat.st_mode):
        raise StrictJsonError(f"{label} must not be a symlink")
    if not stat.S_ISREG(file_stat.st_mode):
        raise StrictJsonError(f"{label} must be a regular file")
    return file_stat


def read_regular_file_bytes(
    path: Path,
    *,
    label: str,
    max_bytes: int | None = None,
) -> bytes:
    """Read one regular file and reject final-component substitutions.

    The package snapshot independently rejects symlinked directories and files.
    This reader also protects direct package helper entry points, where only the
    final input path is available.
    """

    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0
    ):
        raise StrictJsonError("max_bytes must be a positive integer")
    before = _regular_file_stat(path, label=label)
    if max_bytes is not None and before.st_size > max_bytes:
        raise StrictJsonError(f"{label} exceeds the {max_bytes}-byte size limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StrictJsonError(f"{label} could not be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise StrictJsonError(f"{label} must be a regular file")
        if _file_identity(before) != _file_identity(opened):
            raise StrictJsonError(f"{label} changed while being opened")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
        if max_bytes is not None and len(payload) > max_bytes:
            raise StrictJsonError(f"{label} exceeds the {max_bytes}-byte size limit")
        after_read = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(after_read):
            raise StrictJsonError(f"{label} changed while being read")
    finally:
        os.close(descriptor)
    after = _regular_file_stat(path, label=label)
    if _file_identity(before) != _file_identity(after):
        raise StrictJsonError(f"{label} changed while being read")
    return payload


def copy_regular_file_snapshot(
    source: Path,
    destination: Path,
    *,
    label: str,
    mode: int | None = None,
    max_bytes: int | None = None,
) -> None:
    """Stream one immutable regular-file snapshot to a new destination.

    The source descriptor is validated before and after the copy, and the
    source pathname must still identify the same file when copying finishes.
    This gives large checkpoint shards the same safety as
    :func:`read_regular_file_bytes` without retaining a shard-sized byte string.
    """

    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0
    ):
        raise StrictJsonError("max_bytes must be a non-negative integer")
    before = _regular_file_stat(source, label=label)
    if max_bytes is not None and before.st_size > max_bytes:
        raise StrictJsonError(f"{label} exceeds the {max_bytes}-byte size limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise StrictJsonError(f"{label} could not be opened safely") from exc
    created = False
    try:
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or _file_identity(
                before
            ) != _file_identity(opened):
                raise StrictJsonError(f"{label} changed while being opened")
            try:
                with destination.open("xb") as output:
                    created = True
                    copied = 0
                    while chunk := os.read(
                        descriptor,
                        (
                            1024 * 1024
                            if max_bytes is None
                            else min(1024 * 1024, max_bytes - copied + 1)
                        ),
                    ):
                        copied += len(chunk)
                        if max_bytes is not None and copied > max_bytes:
                            raise StrictJsonError(
                                f"{label} exceeds the {max_bytes}-byte size limit"
                            )
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            except OSError as exc:
                raise StrictJsonError(f"{label} could not be copied safely") from exc
            after_read = os.fstat(descriptor)
            if _file_identity(opened) != _file_identity(after_read):
                raise StrictJsonError(f"{label} changed while being copied")
        finally:
            os.close(descriptor)
        after = _regular_file_stat(source, label=label)
        if _file_identity(before) != _file_identity(after):
            raise StrictJsonError(f"{label} changed while being copied")
        if mode is not None:
            try:
                destination.chmod(mode)
            except OSError as exc:
                raise StrictJsonError(
                    f"{label} destination mode could not be preserved"
                ) from exc
    except OSError as exc:
        if created:
            destination.unlink(missing_ok=True)
        raise StrictJsonError(f"{label} could not be copied safely") from exc
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise


def parse_json_bytes(payload: bytes, *, label: str) -> Any:
    """Parse one exact UTF-8 JSON byte stream without ambiguous extensions."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictJsonError(f"{label} is not UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except StrictJsonError:
        raise
    except (RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrictJsonError(f"{label} is not valid JSON") from exc


def load_json(path: Path, *, label: str) -> Any:
    """Load strict JSON from a regular local file."""

    return parse_json_bytes(read_regular_file_bytes(path, label=label), label=label)


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one strict JSON object from a regular local file."""

    _, payload = read_json_object_snapshot(path, label=label)
    return payload


def sha256_prefixed(payload: bytes) -> str:
    """Return the canonical ``sha256:`` digest for already-read bytes.

    Callers that both authenticate and parse an evidence file must derive both
    results from this same immutable byte string, rather than opening a mutable
    pathname once to hash it and again to parse it.
    """

    return "sha256:" + hashlib.sha256(payload).hexdigest()


def read_json_object_snapshot(
    path: Path, *, label: str
) -> tuple[bytes, dict[str, Any]]:
    """Read and parse one strict JSON object from exactly one file snapshot.

    The returned bytes are the authenticated regular-file read.  Callers can
    safely derive a digest with :func:`sha256_prefixed` and use the returned
    object without a second path-based read, closing the hash/parse TOCTOU gap
    for direct evidence staging helpers.
    """

    payload = read_regular_file_bytes(path, label=label)
    decoded = parse_json_bytes(payload, label=label)
    if not isinstance(decoded, dict):
        raise StrictJsonError(f"{label} must decode to a JSON object")
    return payload, decoded


def read_jsonl_snapshot(path: Path, *, label: str) -> tuple[bytes, list[Any]]:
    """Read one regular JSONL file once and reject blank or ambiguous rows."""

    payload = read_regular_file_bytes(path, label=label)
    lines = payload.splitlines()
    if not lines:
        raise StrictJsonError(f"{label} contains no JSON records")
    records: list[Any] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise StrictJsonError(f"{label} contains a blank row at line {line_number}")
        records.append(parse_json_bytes(line, label=f"{label} line {line_number}"))
    return payload, records


__all__ = [
    "StrictJsonError",
    "copy_regular_file_snapshot",
    "load_json",
    "load_json_object",
    "parse_json_bytes",
    "read_regular_file_bytes",
    "read_json_object_snapshot",
    "read_jsonl_snapshot",
    "sha256_prefixed",
]
