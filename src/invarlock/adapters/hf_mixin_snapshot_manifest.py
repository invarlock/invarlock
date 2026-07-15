"""Preflight validation for chunked Hugging Face snapshot manifests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from invarlock.adapters.base import (
    _record_snapshot_member_filename,
    _resolve_snapshot_member_path,
)


def _validate_tensor_manifest_meta(
    tensor: torch.Tensor,
    meta: Any,
    *,
    kind: str,
    name: str,
) -> None:
    if not isinstance(meta, dict):
        return
    expected_shape = meta.get("shape")
    expected_dtype = meta.get("dtype")
    if isinstance(expected_shape, list) and list(tensor.shape) != list(expected_shape):
        raise ValueError(f"Snapshot tensor shape mismatch for {kind}: {name}")
    if isinstance(expected_dtype, str) and expected_dtype:
        if str(tensor.dtype) != expected_dtype:
            raise ValueError(f"Snapshot tensor dtype mismatch for {kind}: {name}")


def _preflight_chunked_manifest(
    *,
    snapshot_dir: Path,
    params_manifest: dict[Any, Any],
    buffers_manifest: dict[Any, Any],
    params_meta: Any,
    buffers_meta: Any,
    param_map: dict[str, torch.nn.Parameter],
    buffer_map: dict[str, torch.Tensor],
    load_tensor: Callable[[Path], torch.Tensor],
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Validate every chunk and return safe paths before any model mutation."""

    seen_filenames: dict[str, str] = {}
    paths_by_kind: dict[str, dict[str, Path]] = {"param": {}, "buffer": {}}
    manifests = (
        ("param", params_manifest, params_meta, param_map),
        ("buffer", buffers_manifest, buffers_meta, buffer_map),
    )
    for kind, manifest, metadata, target_map in manifests:
        for raw_name, filename in manifest.items():
            name = str(raw_name)
            if raw_name not in target_map:
                noun = "parameter" if kind == "param" else "buffer"
                raise KeyError(f"Snapshot {noun} missing in target model: {name}")
            file_path = _resolve_snapshot_member_path(
                snapshot_dir,
                filename,
                entry_kind=kind,
                entry_name=name,
            )
            _record_snapshot_member_filename(
                seen_filenames,
                filename,
                entry_kind=kind,
                entry_name=name,
            )
            paths_by_kind[kind][name] = file_path
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing snapshot tensor for {kind}: {file_path}"
                )
            tensor = load_tensor(file_path)
            meta = metadata.get(raw_name) if isinstance(metadata, dict) else None
            _validate_tensor_manifest_meta(tensor, meta, kind=kind, name=name)

    return paths_by_kind["param"], paths_by_kind["buffer"]
