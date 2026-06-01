"""
Snapshot helpers for HuggingFace adapter mixins.
"""

from __future__ import annotations

import base64
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from invarlock.adapters.base import (
    _record_snapshot_member_filename,
    _resolve_snapshot_member_path,
)
from invarlock.security import is_secure_path


def _sanitize_param_name(name: str) -> str:
    """Return a filesystem-safe parameter name."""
    return name.replace(".", "__").replace("/", "_")


def _ensure_secure_dir(path: Path) -> None:
    """Ensure snapshot directory uses 0o700 permissions."""
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    if not is_secure_path(path):
        raise RuntimeError(
            f"Snapshot directory {path} must have permissions 0o700 for security."
        )


def _resolve_named_parameter(
    module: torch.nn.Module, path: str
) -> torch.nn.Parameter | None:
    """Resolve a parameter by dotted path, returning None if missing."""
    current: Any = module
    parts = path.split(".")
    for name in parts[:-1]:
        current = getattr(current, name, None)
        if current is None:
            return None
    leaf = getattr(current, parts[-1], None)
    if isinstance(leaf, torch.nn.Parameter):
        return leaf
    return None


def _require_safetensors_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        from safetensors.torch import load as load_tensors
        from safetensors.torch import load_file as load_tensor_file
        from safetensors.torch import save as save_tensors
        from safetensors.torch import save_file as save_tensor_file
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "safetensors is required for secure HF snapshot restore. "
            "Install the adapters extra or add safetensors to the runtime image."
        ) from exc
    return save_tensors, load_tensors, save_tensor_file, load_tensor_file


def _serialize_snapshot_blob(
    *,
    tensors: Mapping[str, torch.Tensor],
    metadata: dict[str, Any],
) -> bytes:
    save_tensors, _, _, _ = _require_safetensors_runtime()
    tensor_blob = save_tensors(
        {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    )
    envelope = {
        "format": "invarlock-safetensors-v1",
        "metadata": metadata,
        "tensors_base64": base64.b64encode(tensor_blob).decode("ascii"),
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _deserialize_snapshot_blob(
    blob: bytes,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    _, load_tensors, _, _ = _require_safetensors_runtime()
    envelope = json.loads(blob.decode("utf-8"))
    if not isinstance(envelope, dict):
        raise TypeError("Invalid snapshot payload")
    if envelope.get("format") != "invarlock-safetensors-v1":
        raise ValueError("Unsupported snapshot payload format")
    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("Invalid snapshot payload metadata")
    encoded = envelope.get("tensors_base64")
    if not isinstance(encoded, str) or not encoded:
        raise TypeError("Invalid snapshot payload tensors")
    tensor_blob = base64.b64decode(encoded.encode("ascii"))
    tensors = load_tensors(tensor_blob)
    if not isinstance(tensors, Mapping):
        raise TypeError("Invalid snapshot tensor mapping")
    return metadata, tensors


def _load_chunked_tensor(path: Path) -> torch.Tensor:
    _, _, _, load_tensor_file = _require_safetensors_runtime()
    payload = load_tensor_file(str(path))
    tensor = payload.get("tensor") if isinstance(payload, Mapping) else None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Invalid snapshot tensor payload for {path.name}")
    return tensor


def snapshot_model(adapter: Any, model: torch.nn.Module) -> bytes:
    """Serialize model state with device awareness and weight-tying metadata."""

    state_dict: dict[str, torch.Tensor] = {}
    device_map: dict[str, str] = {}

    for name, param in model.named_parameters():
        state_key = f"params.{name}"
        state_dict[state_key] = param.detach().cpu().clone()
        device_map[state_key] = str(param.device)

    for name, buffer in model.named_buffers():
        state_key = f"buffers.{name}"
        state_dict[state_key] = buffer.detach().cpu().clone()
        device_map[state_key] = str(buffer.device)

    metadata: dict[str, Any] = {
        "config": adapter._serialize_config(model.config)
        if hasattr(model, "config")
        else {},
        "device_map": device_map,
        "model_class": model.__class__.__name__,
        "weight_tying": adapter._extract_weight_tying_info(model),
    }
    return _serialize_snapshot_blob(tensors=state_dict, metadata=metadata)


def restore_model(adapter: Any, model: torch.nn.Module, blob: bytes) -> None:
    """Restore model state produced by `snapshot_model`."""

    metadata, state_dict = _deserialize_snapshot_blob(blob)
    device_map = metadata.get("device_map", {})
    if not isinstance(device_map, dict):
        device_map = {}

    for name, param in model.named_parameters():
        state_key = f"params.{name}"
        if state_key not in state_dict:
            continue
        target_device = torch.device(device_map.get(state_key, "cpu"))
        with torch.no_grad():
            param.copy_(state_dict[state_key].to(target_device))

    for name, buffer_param in model.named_buffers():
        state_key = f"buffers.{name}"
        if state_key not in state_dict:
            continue
        target_device = torch.device(device_map.get(state_key, "cpu"))
        buffer_param.copy_(state_dict[state_key].to(target_device))

    original_tying = metadata.get("weight_tying", {})
    if isinstance(original_tying, dict) and original_tying:
        current_tying = adapter._extract_weight_tying_info(model)
        for tied_param, source_param in original_tying.items():
            if current_tying.get(tied_param) != source_param:
                adapter._restore_weight_tying(model, tied_param, source_param)


def snapshot_model_chunked(
    adapter: Any, model: torch.nn.Module, *, prefix: str = "invarlock-snap-"
) -> str:
    """Create a chunked snapshot on disk to minimise in-memory footprint."""

    snapshot_dir = Path(tempfile.mkdtemp(prefix=prefix))
    _ensure_secure_dir(snapshot_dir)
    _, _, save_tensor_file, _ = _require_safetensors_runtime()

    manifest: dict[str, Any] = {
        "model_class": model.__class__.__name__,
        "config": adapter._serialize_config(model.config)
        if hasattr(model, "config")
        else {},
        "tensor_format": "safetensors",
        "params": {},
        "params_meta": {},
        "buffers": {},
        "buffers_meta": {},
        "device_map": {},
        "weight_tying": adapter._extract_weight_tying_info(model),
    }

    for name, param in model.named_parameters():
        filename = f"param__{_sanitize_param_name(name)}.safetensors"
        file_path = snapshot_dir / filename
        save_tensor_file(
            {"tensor": param.detach().cpu().contiguous()},
            str(file_path),
        )
        manifest["params"][name] = filename
        manifest["params_meta"][name] = {
            "shape": [int(x) for x in param.shape],
            "dtype": str(param.dtype),
        }
        manifest["device_map"][name] = str(param.device)

    for name, buffer in model.named_buffers():
        filename = f"buffer__{_sanitize_param_name(name)}.safetensors"
        file_path = snapshot_dir / filename
        save_tensor_file(
            {"tensor": buffer.detach().cpu().contiguous()},
            str(file_path),
        )
        manifest["buffers"][name] = filename
        manifest["buffers_meta"][name] = {
            "shape": [int(x) for x in buffer.shape],
            "dtype": str(buffer.dtype),
        }
        manifest["device_map"][f"buffer::{name}"] = str(buffer.device)

    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(snapshot_dir)


def _preflight_chunked_manifest(
    *,
    snapshot_dir: Path,
    params_manifest: dict[Any, Any],
    buffers_manifest: dict[Any, Any],
    params_meta: Any,
    buffers_meta: Any,
    param_map: dict[str, torch.nn.Parameter],
    buffer_map: dict[str, torch.Tensor],
) -> tuple[dict[str, Path], dict[str, Path]]:
    seen_filenames: dict[str, str] = {}
    param_paths: dict[str, Path] = {}
    buffer_paths: dict[str, Path] = {}

    for name, filename in params_manifest.items():
        if name not in param_map:
            raise KeyError(f"Snapshot parameter missing in target model: {name}")
        file_path = _resolve_snapshot_member_path(
            snapshot_dir, filename, entry_kind="param", entry_name=str(name)
        )
        _record_snapshot_member_filename(
            seen_filenames,
            filename,
            entry_kind="param",
            entry_name=str(name),
        )
        param_paths[str(name)] = file_path
        if not file_path.exists():
            raise FileNotFoundError(f"Missing snapshot tensor for param: {file_path}")
        tensor = _load_chunked_tensor(file_path)
        meta = params_meta.get(name) if isinstance(params_meta, dict) else None
        _validate_tensor_manifest_meta(tensor, meta, kind="param", name=str(name))

    for name, filename in buffers_manifest.items():
        if name not in buffer_map:
            raise KeyError(f"Snapshot buffer missing in target model: {name}")
        file_path = _resolve_snapshot_member_path(
            snapshot_dir, filename, entry_kind="buffer", entry_name=str(name)
        )
        _record_snapshot_member_filename(
            seen_filenames,
            filename,
            entry_kind="buffer",
            entry_name=str(name),
        )
        buffer_paths[str(name)] = file_path
        if not file_path.exists():
            raise FileNotFoundError(f"Missing snapshot tensor for buffer: {file_path}")
        tensor = _load_chunked_tensor(file_path)
        meta = buffers_meta.get(name) if isinstance(buffers_meta, dict) else None
        _validate_tensor_manifest_meta(tensor, meta, kind="buffer", name=str(name))

    return param_paths, buffer_paths


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


def restore_model_chunked(
    adapter: Any, model: torch.nn.Module, snapshot_path: str
) -> None:
    """Restore a chunked snapshot produced by `snapshot_model_chunked`."""

    snapshot_dir = Path(snapshot_path)
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for snapshot at {snapshot_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Invalid snapshot manifest: payload must be a mapping")
    param_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())

    device_map = manifest.get("device_map", {})
    if not isinstance(device_map, dict):
        device_map = {}

    params_manifest = manifest.get("params", {})
    if not isinstance(params_manifest, dict):
        raise TypeError("Invalid snapshot manifest: params must be a mapping")
    buffers_manifest = manifest.get("buffers", {})
    if not isinstance(buffers_manifest, dict):
        raise TypeError("Invalid snapshot manifest: buffers must be a mapping")

    param_paths, buffer_paths = _preflight_chunked_manifest(
        snapshot_dir=snapshot_dir,
        params_manifest=params_manifest,
        buffers_manifest=buffers_manifest,
        params_meta=manifest.get("params_meta", {}),
        buffers_meta=manifest.get("buffers_meta", {}),
        param_map=param_map,
        buffer_map=buffer_map,
    )

    for name in params_manifest:
        target = param_map[name]
        target_device = torch.device(device_map.get(name, str(target.device)))
        tensor = _load_chunked_tensor(param_paths[str(name)])
        with torch.no_grad():
            target.copy_(tensor.to(target_device))

    for name in buffers_manifest:
        target = buffer_map[name]
        key = f"buffer::{name}"
        target_device = torch.device(device_map.get(key, str(target.device)))
        tensor = _load_chunked_tensor(buffer_paths[str(name)])
        target.copy_(tensor.to(target_device))

    original_tying = manifest.get("weight_tying", {})
    if isinstance(original_tying, dict) and original_tying:
        current_tying = adapter._extract_weight_tying_info(model)
        for tied_param, source_param in original_tying.items():
            if current_tying.get(tied_param) != source_param:
                adapter._restore_weight_tying(model, tied_param, source_param)
