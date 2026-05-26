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


def _replace_output(staging_path: Path, output_path: Path) -> None:
    if output_path.exists():
        shutil.rmtree(output_path)
    staging_path.rename(output_path)


def save_edited_subject_artifact(
    *,
    model: Any,
    tokenizer: Any,
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    staging_path = staging_path_for(output_path)
    if staging_path.exists():
        shutil.rmtree(staging_path)
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
        if staging_path.exists():
            shutil.rmtree(staging_path)
        raise


__all__ = ["save_edited_subject_artifact", "staging_path_for"]
