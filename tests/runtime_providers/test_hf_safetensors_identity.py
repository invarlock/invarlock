from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.runtime_providers._hf_safetensors_identity import (
    HFSafetensorsIdentityError,
    safetensors_storage_keys,
)


def _save(path: Path, *keys: str) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    safetensors_torch.save_file(
        {
            key: torch.tensor([index], dtype=torch.float32)
            for index, key in enumerate(keys)
        },
        path,
    )


def _index(path: Path, mapping: dict[str, str]) -> None:
    path.write_text(
        json.dumps({"metadata": {}, "weight_map": mapping}),
        encoding="utf-8",
    )


def test_single_safetensors_layout_has_exact_inventory(tmp_path: Path) -> None:
    _save(tmp_path / "model.safetensors", "layer.bias", "layer.weight")

    assert safetensors_storage_keys(tmp_path) == {"layer.bias", "layer.weight"}


def test_indexed_shards_require_exact_shard_and_key_map(tmp_path: Path) -> None:
    _save(tmp_path / "model-00001-of-00002.safetensors", "layer.0")
    _save(tmp_path / "model-00002-of-00002.safetensors", "layer.1")
    _index(
        tmp_path / "model.safetensors.index.json",
        {
            "layer.0": "model-00001-of-00002.safetensors",
            "layer.1": "model-00002-of-00002.safetensors",
        },
    )

    assert safetensors_storage_keys(tmp_path) == {"layer.0", "layer.1"}

    _save(tmp_path / "extra.safetensors", "extra")
    with pytest.raises(HFSafetensorsIdentityError, match="exactly match the index"):
        safetensors_storage_keys(tmp_path)


def test_unindexed_multiple_shards_are_rejected(tmp_path: Path) -> None:
    _save(tmp_path / "one.safetensors", "one")
    _save(tmp_path / "two.safetensors", "two")

    with pytest.raises(HFSafetensorsIdentityError, match="must contain only"):
        safetensors_storage_keys(tmp_path)


def test_index_and_shards_must_not_be_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    _save(outside / "model.safetensors", "weight")
    (tmp_path / "model.safetensors").symlink_to(outside / "model.safetensors")
    with pytest.raises(HFSafetensorsIdentityError, match="non-symlink"):
        safetensors_storage_keys(tmp_path)

    (tmp_path / "model.safetensors").unlink()
    _save(tmp_path / "model-00001-of-00001.safetensors", "weight")
    _index(
        outside / "index.json",
        {"weight": "model-00001-of-00001.safetensors"},
    )
    (tmp_path / "model.safetensors.index.json").symlink_to(outside / "index.json")
    with pytest.raises(HFSafetensorsIdentityError, match="regular JSON file"):
        safetensors_storage_keys(tmp_path)


def test_index_must_bind_physical_key_to_exact_shard(tmp_path: Path) -> None:
    _save(tmp_path / "model-00001-of-00002.safetensors", "shared")
    _save(tmp_path / "model-00002-of-00002.safetensors", "shared")
    _index(
        tmp_path / "model.safetensors.index.json",
        {"shared": "model-00001-of-00002.safetensors"},
    )
    with pytest.raises(HFSafetensorsIdentityError, match="exactly match the index"):
        safetensors_storage_keys(tmp_path)

    _index(
        tmp_path / "model.safetensors.index.json",
        {
            "shared": "model-00001-of-00002.safetensors",
            "other": "model-00002-of-00002.safetensors",
        },
    )
    with pytest.raises(HFSafetensorsIdentityError, match="duplicate storage keys"):
        safetensors_storage_keys(tmp_path)


def test_index_rejects_duplicate_json_members_and_traversal(tmp_path: Path) -> None:
    _save(tmp_path / "model-00001-of-00001.safetensors", "weight")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        '{"weight_map":{"weight":"model-00001-of-00001.safetensors",'
        '"weight":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    with pytest.raises(HFSafetensorsIdentityError, match="regular JSON file"):
        safetensors_storage_keys(tmp_path)

    _index(index, {"weight": "../model-00001-of-00001.safetensors"})
    with pytest.raises(HFSafetensorsIdentityError, match="invalid weight_map"):
        safetensors_storage_keys(tmp_path)
