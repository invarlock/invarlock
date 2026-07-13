from __future__ import annotations

from pathlib import Path

import pytest
import torch

from invarlock.adapters.hf_mixin_loading import _is_local_loader_cache_miss
from invarlock.adapters.hf_mixin_snapshot_manifest import (
    _preflight_chunked_manifest,
    _validate_tensor_manifest_meta,
)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (FileNotFoundError("missing"), True),
        (ValueError("not an operating-system error"), False),
        (OSError("Could not locate the cached model"), True),
        (OSError("permission denied"), False),
    ],
)
def test_local_loader_cache_miss_classifies_retryable_failures(
    error: Exception, expected: bool
) -> None:
    assert _is_local_loader_cache_miss(error) is expected


def test_tensor_manifest_metadata_rejects_shape_and_dtype_drift() -> None:
    tensor = torch.zeros(2, 3, dtype=torch.float32)

    _validate_tensor_manifest_meta(tensor, None, kind="param", name="weight")
    _validate_tensor_manifest_meta(tensor, {}, kind="param", name="weight")
    _validate_tensor_manifest_meta(
        tensor,
        {"shape": [2, 3], "dtype": "torch.float32"},
        kind="param",
        name="weight",
    )
    with pytest.raises(ValueError, match="shape mismatch for param: weight"):
        _validate_tensor_manifest_meta(
            tensor,
            {"shape": [3, 2]},
            kind="param",
            name="weight",
        )
    with pytest.raises(ValueError, match="dtype mismatch for buffer: running"):
        _validate_tensor_manifest_meta(
            tensor,
            {"dtype": "torch.float16"},
            kind="buffer",
            name="running",
        )


def test_chunked_manifest_preflight_returns_only_validated_paths(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "weight.safetensors"
    running_path = tmp_path / "running.safetensors"
    weight_path.write_bytes(b"weight")
    running_path.write_bytes(b"running")
    loaded: list[str] = []

    def load_tensor(path: Path) -> torch.Tensor:
        loaded.append(path.name)
        return (
            torch.zeros(2, 3) if path.name == "weight.safetensors" else torch.zeros(3)
        )

    param_paths, buffer_paths = _preflight_chunked_manifest(
        snapshot_dir=tmp_path,
        params_manifest={"weight": "weight.safetensors"},
        buffers_manifest={"running": "running.safetensors"},
        params_meta={"weight": {"shape": [2, 3], "dtype": "torch.float32"}},
        buffers_meta={"running": {"shape": [3], "dtype": "torch.float32"}},
        param_map={"weight": torch.nn.Parameter(torch.zeros(2, 3))},
        buffer_map={"running": torch.zeros(3)},
        load_tensor=load_tensor,
    )

    assert param_paths == {"weight": weight_path.resolve()}
    assert buffer_paths == {"running": running_path.resolve()}
    assert loaded == ["weight.safetensors", "running.safetensors"]


@pytest.mark.parametrize(
    ("params_manifest", "buffers_manifest", "match"),
    [
        ({"missing": "missing.safetensors"}, {}, "parameter missing"),
        ({}, {"missing": "missing.safetensors"}, "buffer missing"),
    ],
)
def test_chunked_manifest_preflight_rejects_unknown_model_members(
    tmp_path: Path,
    params_manifest: dict[str, str],
    buffers_manifest: dict[str, str],
    match: str,
) -> None:
    with pytest.raises(KeyError, match=match):
        _preflight_chunked_manifest(
            snapshot_dir=tmp_path,
            params_manifest=params_manifest,
            buffers_manifest=buffers_manifest,
            params_meta={},
            buffers_meta={},
            param_map={},
            buffer_map={},
            load_tensor=lambda _path: torch.zeros(1),
        )


def test_chunked_manifest_preflight_rejects_missing_tensor_before_load(
    tmp_path: Path,
) -> None:
    load_calls: list[Path] = []
    with pytest.raises(FileNotFoundError, match="Missing snapshot tensor for param"):
        _preflight_chunked_manifest(
            snapshot_dir=tmp_path,
            params_manifest={"weight": "missing.safetensors"},
            buffers_manifest={},
            params_meta={},
            buffers_meta={},
            param_map={"weight": torch.nn.Parameter(torch.zeros(1))},
            buffer_map={},
            load_tensor=lambda path: load_calls.append(path) or torch.zeros(1),
        )
    assert load_calls == []
