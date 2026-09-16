"""Public I/O helpers for normalized records and independently pinned artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    digest,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import _file_identity


def physical_file_digest(path: str | Path) -> str:
    """Hash exact file bytes in bounded chunks, rejecting changes and symlinks."""
    from invarlock.captured_contracts import secure_directory

    path = Path(path)
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        with secure_directory(path.parent) as parent:
            initial = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            descriptor = os.open(path.name, flags, dir_fd=parent)
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise EvaluationRecordsError("artifact must be a regular file")
                if _file_identity(initial) != _file_identity(before):
                    raise EvaluationRecordsError("artifact changed during hashing")
                hasher = hashlib.sha256()
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(chunk)
                after = os.fstat(descriptor)
                current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                if _file_identity(before) != _file_identity(after) or _file_identity(
                    before
                ) != _file_identity(current):
                    raise EvaluationRecordsError("artifact changed during hashing")
            finally:
                os.close(descriptor)
        return "sha256:" + hasher.hexdigest()
    except (OSError, ValueError) as exc:
        raise EvaluationRecordsError(str(exc)) from exc


def run_digest(run: Mapping[str, Any]) -> str:
    """Hash a validated complete run without changing its record order."""
    from invarlock.evaluation_comparison.comparison import _check_run

    value = dict(run)
    _check_run(value)
    return digest(value)


def write_run(path: str | Path, run: Mapping[str, Any]) -> Path:
    """Publish a canonical run without replacing existing user files."""
    from invarlock.captured_contracts import atomic_write, secure_directory
    from invarlock.evaluation_comparison.comparison import _check_run

    value = dict(run)
    _check_run(value)
    path = Path(path)
    try:
        with secure_directory(path.parent):
            atomic_write(path, canonical_json_bytes(value))
    except (OSError, ValueError) as exc:
        raise EvaluationRecordsError(f"cannot create run: {exc}") from exc
    return path
