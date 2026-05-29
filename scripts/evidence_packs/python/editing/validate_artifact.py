from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from .metadata import read_edit_metadata, validate_edit_metadata
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from editing.metadata import read_edit_metadata, validate_edit_metadata

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - optional at import time
    safe_open = None  # type: ignore[assignment]


@dataclass
class EditArtifactValidationResult:
    ok: bool
    has_config: bool = False
    has_weights: bool = False
    has_tokenizer: bool = False
    has_metadata: bool = False
    artifact_class: str | None = None
    edit_type: str | None = None
    issues: list[str] | None = None

    def __bool__(self) -> bool:
        return self.ok

    def to_json_payload(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "has_config": self.has_config,
            "has_weights": self.has_weights,
            "has_tokenizer": self.has_tokenizer,
            "has_metadata": self.has_metadata,
            "artifact_class": self.artifact_class,
            "edit_type": self.edit_type,
            "issues": list(self.issues or []),
        }


def _has_tokenizer(edit_path: Path) -> bool:
    return any(
        (edit_path / name).is_file()
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "special_tokens_map.json",
        )
    )


def _validate_safetensors(path: Path) -> bool:
    if safe_open is None:
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return any(True for _ in handle.keys())
    except Exception:
        return False


def _validate_index_shards(edit_path: Path, index_path: Path) -> bool:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False

    shard_names = sorted({str(name) for name in weight_map.values() if str(name)})
    if not shard_names:
        return False

    for shard_name in shard_names:
        shard_path = edit_path / shard_name
        if not shard_path.is_file():
            return False
        if shard_path.suffix == ".safetensors" and not _validate_safetensors(
            shard_path
        ):
            return False
    return True


def _has_valid_weights(edit_path: Path) -> bool:
    single_safe = edit_path / "model.safetensors"
    safe_index = edit_path / "model.safetensors.index.json"
    single_bin = edit_path / "pytorch_model.bin"
    bin_index = edit_path / "pytorch_model.bin.index.json"

    if single_safe.is_file():
        return _validate_safetensors(single_safe)
    if safe_index.is_file():
        return _validate_index_shards(edit_path, safe_index)
    if single_bin.is_file():
        return True
    if bin_index.is_file():
        return _validate_index_shards(edit_path, bin_index)
    return False


def validate_edit_artifact(
    edit_path: Path,
    *,
    require_metadata: bool = False,
    expected_edit_type: str | None = None,
    expected_artifact_class: str | None = None,
) -> EditArtifactValidationResult:
    issues: list[str] = []
    if not edit_path.is_dir():
        return EditArtifactValidationResult(
            ok=False,
            issues=[f"edit artifact directory not found: {edit_path}"],
        )

    has_config = (edit_path / "config.json").is_file()
    has_tokenizer = _has_tokenizer(edit_path)
    has_weights = _has_valid_weights(edit_path)

    if not has_config:
        issues.append("config.json missing")
    if not has_tokenizer:
        issues.append("tokenizer files missing")
    if not has_weights:
        issues.append("model weights missing or invalid")

    metadata_path = edit_path / "edit_metadata.json"
    has_metadata = metadata_path.is_file()
    artifact_class: str | None = None
    edit_type: str | None = None
    if require_metadata and not has_metadata:
        issues.append("edit_metadata.json missing")
    if has_metadata:
        try:
            metadata = read_edit_metadata(metadata_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            issues.append(f"edit_metadata.json invalid: {exc}")
        else:
            artifact_class_raw = metadata.get("artifact_class")
            edit_type_raw = metadata.get("edit_type")
            artifact_class = (
                artifact_class_raw if isinstance(artifact_class_raw, str) else None
            )
            edit_type = edit_type_raw if isinstance(edit_type_raw, str) else None
            issues.extend(
                validate_edit_metadata(
                    metadata,
                    expected_edit_type=expected_edit_type,
                    expected_artifact_class=expected_artifact_class,
                )
            )

    return EditArtifactValidationResult(
        ok=not issues,
        has_config=has_config,
        has_weights=has_weights,
        has_tokenizer=has_tokenizer,
        has_metadata=has_metadata,
        artifact_class=artifact_class,
        edit_type=edit_type,
        issues=issues,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an evidence-pack edit artifact."
    )
    parser.add_argument("edit_path")
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument("--expected-edit-type")
    parser.add_argument("--expected-artifact-class")
    parser.add_argument("--json", action="store_true", dest="json_out")
    args = parser.parse_args(argv[1:])

    result = validate_edit_artifact(
        Path(args.edit_path),
        require_metadata=bool(args.require_metadata),
        expected_edit_type=args.expected_edit_type,
        expected_artifact_class=args.expected_artifact_class,
    )
    if args.json_out:
        print(json.dumps(result.to_json_payload(), sort_keys=True))
    elif not result.ok:
        for issue in result.issues or []:
            print(issue, file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
