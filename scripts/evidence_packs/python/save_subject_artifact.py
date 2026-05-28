from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

try:
    from edit_metadata import write_edit_metadata
    from validate_edit_artifact import validate_edit_artifact
except ImportError:  # pragma: no cover - direct module load under pytest
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from edit_metadata import write_edit_metadata
    from validate_edit_artifact import validate_edit_artifact


def staging_path_for(output_path: Path) -> Path:
    return output_path.parent / f".{output_path.name}.tmp"


def backup_path_for(output_path: Path) -> Path:
    return output_path.parent / f".{output_path.name}.bak"


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _replace_output(staging_path: Path, output_path: Path) -> None:
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


def _remove_staging(staging_path: Path, output_path: Path) -> None:
    if staging_path.exists():
        shutil.rmtree(staging_path)
    backup_path = backup_path_for(output_path)
    if backup_path.exists() and not output_path.exists():
        backup_path.rename(output_path)


def _reset_staging(staging_path: Path) -> None:
    if staging_path.exists():
        shutil.rmtree(staging_path)


def save_edited_subject_artifact(
    *,
    model: Any,
    tokenizer: Any,
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    staging_path = staging_path_for(output_path)
    _reset_staging(staging_path)
    staging_path.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer.save_pretrained(staging_path)
        model.save_pretrained(staging_path, safe_serialization=True)
        write_edit_metadata(staging_path / "edit_metadata.json", metadata)
        result = validate_edit_artifact(
            staging_path,
            require_metadata=True,
            expected_edit_type=str(metadata.get("edit_type") or ""),
            expected_artifact_class=str(metadata.get("artifact_class") or ""),
        )
        if not result.ok:
            raise RuntimeError(
                "saved edit artifact failed validation: " + "; ".join(result.issues)
            )
        _replace_output(staging_path, output_path)
    except Exception:
        _remove_staging(staging_path, output_path)
        raise


__all__ = ["backup_path_for", "save_edited_subject_artifact", "staging_path_for"]
