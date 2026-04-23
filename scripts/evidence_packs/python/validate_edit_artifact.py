from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - optional at import time
    safe_open = None  # type: ignore[assignment]


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


def validate_edit_artifact(edit_path: Path) -> bool:
    if not edit_path.is_dir():
        return False
    if not (edit_path / "config.json").is_file():
        return False
    if not _has_tokenizer(edit_path):
        return False

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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: validate_edit_artifact.py <edit_path>", file=sys.stderr)
        return 2
    return 0 if validate_edit_artifact(Path(argv[1])) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
