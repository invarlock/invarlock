from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from invarlock.runtime_providers.hf_transformers import (
    load_hf_model_with_strict_loading_info,
)
from tests.runtime_providers._hf_transformers_helpers import _save_safetensors


def _loader() -> Mock:
    return Mock(
        return_value=(
            object(),
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            },
        )
    )


@pytest.mark.parametrize(
    "payload",
    [
        "parent",
        "absolute",
        "nested",
        "nested-parent",
        "symlink-shard",
        "fifo-shard",
        "directory-shard",
        "symlink-index",
        "fifo-index",
    ],
)
def test_unsafe_checkpoint_never_reaches_native_loader(
    tmp_path: Path, payload: str
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    shard = checkpoint / "shard.safetensors"
    _save_safetensors(shard, "weight")
    outside = tmp_path / "outside.safetensors"
    _save_safetensors(outside, "weight")
    reference = {
        "parent": "../outside.safetensors",
        "absolute": str(outside),
        "nested": "subdir/shard.safetensors",
        "nested-parent": "subdir/../../outside.safetensors",
    }.get(payload, shard.name)
    index = checkpoint / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": reference}}))
    if payload in {"symlink-shard", "fifo-shard", "directory-shard"}:
        shard.unlink()
        if payload == "symlink-shard":
            shard.symlink_to(outside)
        elif payload == "fifo-shard":
            os.mkfifo(shard)
        else:
            shard.mkdir()
    elif payload == "symlink-index":
        outside_index = tmp_path / "index.json"
        index.rename(outside_index)
        index.symlink_to(outside_index)
    elif payload == "fifo-index":
        index.unlink()
        os.mkfifo(index)
    loader = _loader()

    with pytest.raises(RuntimeError, match="canonical safetensors layout"):
        load_hf_model_with_strict_loading_info(loader, checkpoint)

    loader.assert_not_called()


@pytest.mark.parametrize("sharded", [False, True])
def test_valid_checkpoint_reaches_native_loader(tmp_path: Path, sharded: bool) -> None:
    if sharded:
        _save_safetensors(tmp_path / "one.safetensors", "one")
        _save_safetensors(tmp_path / "two.safetensors", "two")
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {"weight_map": {"one": "one.safetensors", "two": "two.safetensors"}}
            )
        )
    else:
        _save_safetensors(tmp_path / "model.safetensors", "weight")
    loader = _loader()

    assert (
        load_hf_model_with_strict_loading_info(loader, tmp_path)
        is loader.return_value[0]
    )

    loader.assert_called_once_with(
        str(tmp_path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
        dtype="auto",
    )
