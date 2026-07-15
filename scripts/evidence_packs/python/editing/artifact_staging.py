"""Neutral atomic staging helpers for checkpoint-producing workflows.

These filesystem operations deliberately have no dependency on a materializer
or verifier.  Keeping them here prevents an output-producing path from
importing validation code merely to obtain atomic publication primitives.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def staging_path_for(output_path: Path) -> Path:
    """Return the sibling staging directory for one published artifact."""

    return output_path.parent / f".{output_path.name}.tmp"


def backup_path_for(output_path: Path) -> Path:
    """Return the sibling rollback location for one published artifact."""

    return output_path.parent / f".{output_path.name}.bak"


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def replace_output(staging_path: Path, output_path: Path) -> None:
    """Atomically replace an artifact while retaining rollback on failure."""

    backup_path = backup_path_for(output_path)
    if backup_path.exists():
        _remove_path(backup_path)

    moved_existing = False
    try:
        if output_path.exists():
            output_path.rename(backup_path)
            moved_existing = True
        staging_path.rename(output_path)
    except Exception:
        if output_path.exists():
            _remove_path(output_path)
        if moved_existing and backup_path.exists():
            backup_path.rename(output_path)
        raise

    if backup_path.exists():
        _remove_path(backup_path)


def remove_staging(staging_path: Path, output_path: Path) -> None:
    """Discard staging data and restore a prior output when necessary."""

    if staging_path.exists():
        shutil.rmtree(staging_path)
    backup_path = backup_path_for(output_path)
    if backup_path.exists() and not output_path.exists():
        backup_path.rename(output_path)


def reset_staging(staging_path: Path) -> None:
    """Discard a prior staging directory before an explicit restart."""

    if staging_path.exists():
        shutil.rmtree(staging_path)
