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

from invarlock.adapters.hf_mixin_snapshot_manifest import (
    _preflight_chunked_manifest,
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


def _set_named_parameter_alias(
    module: torch.nn.Module,
    tied_path: str,
    source: torch.nn.Parameter,
) -> None:
    """Bind ``tied_path`` to the exact source ``Parameter`` object."""

    parts = tied_path.split(".")
    current: Any = module
    for name in parts[:-1]:
        current = getattr(current, name, None)
        if current is None:
            raise KeyError(f"Unable to resolve tied parameter parent: {tied_path}")
    leaf = parts[-1]
    if not isinstance(current, torch.nn.Module):
        raise TypeError(f"Tied parameter parent is not a module: {tied_path}")
    setattr(current, leaf, source)
    rebound = getattr(current, leaf, None)
    if rebound is not source:
        raise RuntimeError(f"Failed to restore parameter alias: {tied_path}")


def _require_exact_snapshot_members(
    *,
    snapshot_names: set[str],
    target_names: set[str],
    kind: str,
    allowed_target_extras: set[str] | None = None,
) -> None:
    allowed_extras = allowed_target_extras or set()
    missing = sorted((target_names - snapshot_names) - allowed_extras)
    unexpected = sorted(snapshot_names - target_names)
    if missing or unexpected:
        raise KeyError(
            f"Snapshot {kind} set mismatch: missing={missing} unexpected={unexpected}"
        )


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
    return json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


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
    original_tying = metadata.get("weight_tying", {})
    if not isinstance(original_tying, dict):
        original_tying = {}

    param_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())
    snapshot_param_names = {
        key.removeprefix("params.")
        for key in state_dict
        if str(key).startswith("params.")
    }
    snapshot_buffer_names = {
        key.removeprefix("buffers.")
        for key in state_dict
        if str(key).startswith("buffers.")
    }
    _require_exact_snapshot_members(
        snapshot_names=snapshot_param_names,
        target_names=set(param_map),
        kind="parameter",
        allowed_target_extras={str(name) for name in original_tying},
    )
    _require_exact_snapshot_members(
        snapshot_names=snapshot_buffer_names,
        target_names=set(buffer_map),
        kind="buffer",
    )

    # Validate the entire payload before mutating the target so a corrupt or
    # incompatible late tensor cannot leave a partially restored model.
    for name, param in param_map.items():
        state_key = f"params.{name}"
        if state_key not in state_dict and name in original_tying:
            continue
        tensor = state_dict[state_key]
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(f"Snapshot tensor shape mismatch for param: {name}")
        if tensor.dtype != param.dtype:
            raise ValueError(f"Snapshot tensor dtype mismatch for param: {name}")
    for name, buffer_param in buffer_map.items():
        state_key = f"buffers.{name}"
        tensor = state_dict[state_key]
        if tuple(tensor.shape) != tuple(buffer_param.shape):
            raise ValueError(f"Snapshot tensor shape mismatch for buffer: {name}")
        if tensor.dtype != buffer_param.dtype:
            raise ValueError(f"Snapshot tensor dtype mismatch for buffer: {name}")

    for name, param in param_map.items():
        state_key = f"params.{name}"
        if state_key not in state_dict and name in original_tying:
            # The tied alias was intentionally deduplicated in the snapshot.  It
            # is rebound to its source after source tensors are restored.
            continue
        target_device = torch.device(device_map.get(state_key, "cpu"))
        with torch.no_grad():
            param.copy_(state_dict[state_key].to(target_device))

    for name, buffer_param in buffer_map.items():
        state_key = f"buffers.{name}"
        target_device = torch.device(device_map.get(state_key, "cpu"))
        buffer_param.copy_(state_dict[state_key].to(target_device))

    if original_tying:
        current_tying = adapter._extract_weight_tying_info(model)
        for tied_param, source_param in original_tying.items():
            if current_tying.get(tied_param) != source_param:
                adapter._restore_weight_tying(model, tied_param, source_param)
        adapter.validate_weight_tying(model, expected_tying=original_tying)


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
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    return str(snapshot_dir)


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
    original_tying = manifest.get("weight_tying", {})
    if not isinstance(original_tying, dict):
        original_tying = {}
    _require_exact_snapshot_members(
        snapshot_names={str(name) for name in params_manifest},
        target_names=set(param_map),
        kind="parameter",
        allowed_target_extras={str(name) for name in original_tying},
    )
    _require_exact_snapshot_members(
        snapshot_names={str(name) for name in buffers_manifest},
        target_names=set(buffer_map),
        kind="buffer",
    )

    param_paths, buffer_paths = _preflight_chunked_manifest(
        snapshot_dir=snapshot_dir,
        params_manifest=params_manifest,
        buffers_manifest=buffers_manifest,
        params_meta=manifest.get("params_meta", {}),
        buffers_meta=manifest.get("buffers_meta", {}),
        param_map=param_map,
        buffer_map=buffer_map,
        load_tensor=_load_chunked_tensor,
    )

    for name in params_manifest:
        param_target = param_map[name]
        target_device = torch.device(device_map.get(name, str(param_target.device)))
        tensor = _load_chunked_tensor(param_paths[str(name)])
        with torch.no_grad():
            param_target.copy_(tensor.to(target_device))

    for name in buffers_manifest:
        buffer_target = buffer_map[name]
        key = f"buffer::{name}"
        target_device = torch.device(device_map.get(key, str(buffer_target.device)))
        tensor = _load_chunked_tensor(buffer_paths[str(name)])
        buffer_target.copy_(tensor.to(target_device))

    if original_tying:
        current_tying = adapter._extract_weight_tying_info(model)
        for tied_param, source_param in original_tying.items():
            if current_tying.get(tied_param) != source_param:
                adapter._restore_weight_tying(model, tied_param, source_param)
        adapter.validate_weight_tying(model, expected_tying=original_tying)
