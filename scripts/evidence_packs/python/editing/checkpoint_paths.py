"""Path containment helpers for local checkpoint trees."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


class CheckpointLayoutError(ValueError):
    """Raised when an identity-sensitive checkpoint tree is unsafe."""


def require_regular_checkpoint_tree(checkpoint: Path, *, label: str) -> None:
    """Require a regular directory with no symlinks or special entries."""

    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise CheckpointLayoutError(
            f"{label} must be a regular directory: {checkpoint}"
        )
    for candidate in sorted(checkpoint.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(checkpoint).as_posix()
        if candidate.is_symlink():
            raise CheckpointLayoutError(
                f"{label} must not contain symlinks: {relative}"
            )
        try:
            mode = candidate.lstat().st_mode
        except OSError as exc:
            raise CheckpointLayoutError(
                f"{label} entry cannot be inspected: {relative}: {exc}"
            ) from exc
        if not (os.path.isdir(candidate) or os.path.isfile(candidate)):
            raise CheckpointLayoutError(
                f"{label} contains unsupported entry: {relative} (mode={mode:o})"
            )


def resolve_checkpoint_child_path(
    checkpoint: Path,
    raw_name: str,
    *,
    label: str = "checkpoint entry",
) -> tuple[Path | None, str | None]:
    """Resolve a checkpoint-relative path without allowing tree escape.

    Hugging Face weight indexes are supposed to point to files inside the
    checkpoint directory.  Treat absolute paths, ``..`` components, and symlink
    traversal as invalid because otherwise a checkpoint digest can exclude bytes
    that a validator later reads as weights.
    """

    if checkpoint.is_symlink() or not checkpoint.is_dir():
        return None, f"{label} root must be a regular directory"
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, f"{label} path is missing"
    if "\\" in raw_name:
        return None, f"{label} path must use a safe relative path: {raw_name}"
    raw_path = PurePosixPath(raw_name)
    if raw_path.is_absolute() or any(
        part in {"", ".", ".."} for part in raw_path.parts
    ):
        return None, f"{label} path must be checkpoint-relative: {raw_name}"

    try:
        root = checkpoint.resolve(strict=True)
    except OSError as exc:
        return None, f"{label} root cannot be resolved: {exc}"
    candidate = checkpoint.joinpath(*raw_path.parts)
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except OSError:
        resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(root)
    except ValueError:
        return None, f"{label} path escapes checkpoint tree: {raw_name}"

    current = checkpoint
    for part in raw_path.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            return None, f"{label} path traverses symlink: {raw_name}"

    return candidate, None


def require_checkpoint_child_file(
    checkpoint: Path,
    raw_name: str,
    *,
    label: str = "checkpoint entry",
) -> Path:
    path, error = resolve_checkpoint_child_path(checkpoint, raw_name, label=label)
    if error is not None or path is None:
        raise CheckpointLayoutError(error or f"{label} path is invalid")
    if not path.is_file():
        raise CheckpointLayoutError(f"{label} file is missing: {raw_name}")
    return path


__all__ = [
    "CheckpointLayoutError",
    "require_checkpoint_child_file",
    "require_regular_checkpoint_tree",
    "resolve_checkpoint_child_path",
]
