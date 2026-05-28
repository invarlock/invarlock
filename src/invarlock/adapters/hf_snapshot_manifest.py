from __future__ import annotations

from pathlib import Path

__all__ = ["record_snapshot_member_filename", "resolve_snapshot_member_path"]


def resolve_snapshot_member_path(
    snapshot_dir: Path,
    filename: str,
    *,
    entry_kind: str,
    entry_name: str,
) -> Path:
    if not isinstance(filename, str) or not filename:
        raise TypeError(
            f"Invalid snapshot manifest filename for {entry_kind}: {entry_name}"
        )
    if (
        Path(filename).is_absolute()
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
    ):
        raise ValueError(
            f"Invalid snapshot manifest filename for {entry_kind}: {entry_name}"
        )

    file_path = snapshot_dir / filename
    try:
        snapshot_root = snapshot_dir.resolve(strict=True)
        resolved_file = file_path.resolve(strict=True)
    except FileNotFoundError:
        return file_path
    if not resolved_file.is_relative_to(snapshot_root):
        raise ValueError(
            f"Snapshot manifest filename escapes snapshot directory for "
            f"{entry_kind}: {entry_name}"
        )
    return file_path


def record_snapshot_member_filename(
    seen_filenames: dict[str, str],
    filename: str,
    *,
    entry_kind: str,
    entry_name: str,
) -> None:
    owner = f"{entry_kind}:{entry_name}"
    previous_owner = seen_filenames.get(filename)
    if previous_owner is not None:
        raise ValueError(
            f"Duplicate snapshot manifest filename {filename!r} for {owner}; "
            f"already used by {previous_owner}"
        )
    seen_filenames[filename] = owner
