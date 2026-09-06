"""Closed contracts and bounded I/O for pipeline comparisons."""

from __future__ import annotations

import hashlib
import importlib.resources
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_EVIDENCE_BYTES = 192 * 1024 * 1024
MAX_RECORDS = 10000


class PipelineError(ValueError):
    """Malformed, unsupported or contradictory pipeline evidence."""


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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
        if len(canonical_json_bytes(value)) > limit:
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
