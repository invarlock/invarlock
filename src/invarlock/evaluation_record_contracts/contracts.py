"""Closed contracts and bounded I/O for captured evaluation records."""

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

from invarlock.evaluation_record_contracts.validation_limits import (
    check_record_counts,
    format_validation_error,
)
from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes

MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_RECORDS = 50000
_CANONICAL_BUFFER_BYTES = 65536


class EvaluationRecordsError(ValueError):
    """Expected record validation, capacity, adapter, or safe I/O failure."""


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
        raise EvaluationRecordsError(f"value is not canonical JSON: {exc}") from exc


def digest(value: Any) -> str:
    hasher = hashlib.sha256()
    for chunk in _canonical_chunks(value):
        hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


@lru_cache(maxsize=6)
def _validator(name: str) -> Draft202012Validator:
    schema = parse_json_bytes(
        importlib.resources.files("invarlock")
        .joinpath(
            "_data",
            "contracts",
            {
                "run": "evaluation_run.schema.json",
                "case_set": "evaluation_case_set.schema.json",
                "policy": "comparison_policy.schema.json",
                "comparison": "multi_metric_comparison.schema.json",
            }[name],
        )
        .read_bytes(),
        label="evaluation record schema",
    )
    return Draft202012Validator(schema)


def validate(value: Any, name: str) -> None:
    try:
        check_record_counts(value, name, max_records=MAX_RECORDS)
        limit = MAX_INPUT_BYTES
        size = 0
        for chunk in _canonical_chunks(value):
            size += len(chunk)
            if size > limit:
                raise EvaluationRecordsError(f"{name} exceeds the {limit} byte limit")
        error = next(_validator(name).iter_errors(value), None)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise EvaluationRecordsError(f"invalid {name}: {exc}") from exc
    if error is not None:
        raise EvaluationRecordsError(format_validation_error(error, name))


def read_json(path: str | Path, *, max_bytes: int = MAX_INPUT_BYTES) -> Any:
    try:
        return parse_json_bytes(
            read_regular_file_bytes(
                Path(path), label="evaluation input", max_bytes=max_bytes
            ),
            label="evaluation input",
        )
    except (ValueError, OSError, RecursionError) as exc:
        raise EvaluationRecordsError(str(exc)) from exc


__all__ = [
    "EvaluationRecordsError",
    "MAX_INPUT_BYTES",
    "MAX_RECORDS",
    "digest",
    "read_json",
    "validate",
]
