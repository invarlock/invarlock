"""Fail-closed safetensors inventory for the built-in HF provider."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from invarlock.evidence_pack_json import StrictJsonError, read_json_object_snapshot


class HFSafetensorsIdentityError(RuntimeError):
    """Raised when a checkpoint does not have one canonical safe layout."""


def _read_weight_map(index_path: Path) -> dict[str, str]:
    try:
        _, index = read_json_object_snapshot(
            index_path,
            label="HF safetensors index",
        )
    except StrictJsonError as exc:
        raise HFSafetensorsIdentityError(
            "HF safetensors index must be an immutable regular JSON file"
        ) from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise HFSafetensorsIdentityError("HF safetensors index has no weight_map")
    normalized: dict[str, str] = {}
    for key, shard in weight_map.items():
        if (
            not isinstance(key, str)
            or not key
            or key != key.strip()
            or not isinstance(shard, str)
            or not shard
            or shard != Path(shard).name
            or not shard.endswith(".safetensors")
        ):
            raise HFSafetensorsIdentityError(
                "HF safetensors index has an invalid weight_map entry"
            )
        normalized[key] = shard
    return normalized


def safetensors_storage_keys(checkpoint: Path) -> set[str]:
    """Return the exact keys in one canonical single- or indexed-shard layout."""

    try:
        from safetensors import SafetensorError, safe_open
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise HFSafetensorsIdentityError(
            "strict HF identity requires the safetensors runtime"
        ) from exc

    root = Path(checkpoint)
    try:
        if root.is_symlink() or not root.is_dir():
            raise HFSafetensorsIdentityError("HF checkpoint must be a real directory")
        shards = sorted(
            (path for path in root.iterdir() if path.name.endswith(".safetensors")),
            key=lambda path: path.name,
        )
    except OSError as exc:
        raise HFSafetensorsIdentityError(
            "HF checkpoint safetensors inventory is unavailable"
        ) from exc
    if not shards:
        raise HFSafetensorsIdentityError("HF checkpoint has no safetensors shards")

    index_path = root / "model.safetensors.index.json"
    indexed = index_path.exists() or index_path.is_symlink()
    indexed_key_to_shard = _read_weight_map(index_path) if indexed else None
    shard_names = {path.name for path in shards}
    if indexed_key_to_shard is None:
        if [path.name for path in shards] != ["model.safetensors"]:
            raise HFSafetensorsIdentityError(
                "unindexed HF safetensors layout must contain only model.safetensors"
            )
    elif shard_names != set(indexed_key_to_shard.values()):
        raise HFSafetensorsIdentityError(
            "HF safetensors shards do not exactly match the index"
        )

    keys: set[str] = set()
    key_to_shard: dict[str, str] = {}
    for shard in shards:
        if shard.is_symlink() or not shard.is_file():
            raise HFSafetensorsIdentityError(
                "HF safetensors shard must be a regular non-symlink file"
            )
        try:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                shard_keys = list(handle.keys())
        except (OSError, RuntimeError, SafetensorError, ValueError) as exc:
            raise HFSafetensorsIdentityError(
                "HF safetensors shard cannot be audited"
            ) from exc
        if not shard_keys or any(
            not isinstance(key, str) or not key or key != key.strip()
            for key in shard_keys
        ):
            raise HFSafetensorsIdentityError(
                "HF safetensors shard has invalid storage keys"
            )
        if keys.intersection(shard_keys):
            raise HFSafetensorsIdentityError(
                "HF safetensors shards contain duplicate storage keys"
            )
        keys.update(shard_keys)
        key_to_shard.update(dict.fromkeys(shard_keys, shard.name))
    if indexed_key_to_shard is not None and key_to_shard != indexed_key_to_shard:
        raise HFSafetensorsIdentityError(
            "HF safetensors keys do not exactly match the index"
        )
    return keys


__all__ = ["HFSafetensorsIdentityError", "safetensors_storage_keys"]
