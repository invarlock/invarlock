from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.runtime_providers._hf_safetensors_identity import (
    HFSafetensorsIdentityError,
    safetensors_storage_keys,
)


def _save(path: Path, keys: tuple[str, ...]) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    safetensors_torch.save_file(
        {key: torch.tensor([index]) for index, key in enumerate(keys)},
        path,
    )


def test_checkpoint_must_be_a_directory_with_safetensors(tmp_path: Path) -> None:
    checkpoint_file = tmp_path / "checkpoint"
    checkpoint_file.write_bytes(b"not a directory")
    with pytest.raises(HFSafetensorsIdentityError, match="real directory"):
        safetensors_storage_keys(checkpoint_file)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(HFSafetensorsIdentityError, match="no safetensors shards"):
        safetensors_storage_keys(empty)


def test_index_requires_nonempty_weight_map(tmp_path: Path) -> None:
    _save(tmp_path / "model-00001-of-00001.safetensors", ("weight",))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}}),
        encoding="utf-8",
    )

    with pytest.raises(HFSafetensorsIdentityError, match="has no weight_map"):
        safetensors_storage_keys(tmp_path)


@pytest.mark.parametrize("keys", [(), ("",), (" weight",)])
def test_shard_requires_nonempty_trimmed_storage_keys(
    tmp_path: Path,
    keys: tuple[str, ...],
) -> None:
    _save(tmp_path / "model.safetensors", keys)

    with pytest.raises(HFSafetensorsIdentityError, match="invalid storage keys"):
        safetensors_storage_keys(tmp_path)


def test_corrupt_safetensors_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"not-safetensors")

    with pytest.raises(HFSafetensorsIdentityError, match="cannot be audited"):
        safetensors_storage_keys(tmp_path)


def test_index_must_bind_every_physical_key_to_its_exact_shard(tmp_path: Path) -> None:
    shard = "model-00001-of-00001.safetensors"
    _save(tmp_path / shard, ("actual",))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"declared": shard}}),
        encoding="utf-8",
    )

    with pytest.raises(HFSafetensorsIdentityError, match="keys do not exactly match"):
        safetensors_storage_keys(tmp_path)
