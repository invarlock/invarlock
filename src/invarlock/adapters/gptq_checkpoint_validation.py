"""Fail-closed validation for tensors ignored by GPTQModel during loading."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any


class GPTQCheckpointValidationError(ValueError):
    """Raised when a GPTQ checkpoint contains unbound or unsafe tensors."""


def _checkpoint_layout(
    root: Path,
) -> tuple[tuple[Path, ...], dict[str, str] | None]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GPTQCheckpointValidationError(
                "GPTQ safetensors index could not be decoded"
            ) from exc
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise GPTQCheckpointValidationError(
                "GPTQ safetensors index must contain a non-empty weight_map"
            )
        shard_names: set[str] = set()
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise GPTQCheckpointValidationError(
                    "GPTQ safetensors index contains an invalid tensor name"
                )
            if (
                not isinstance(shard_name, str)
                or not shard_name
                or Path(shard_name).name != shard_name
            ):
                raise GPTQCheckpointValidationError(
                    "GPTQ safetensors index contains an unsafe shard path"
                )
            shard_names.add(shard_name)
        shards = tuple(root / name for name in sorted(shard_names))
    else:
        weight_map = None
        shards = tuple(sorted(root.glob("*.safetensors")))
        if len(shards) != 1:
            raise GPTQCheckpointValidationError(
                "GPTQ checkpoint must contain one safetensors file or an index"
            )
    if any(not path.is_file() or path.is_symlink() for path in shards):
        raise GPTQCheckpointValidationError(
            "GPTQ checkpoint contains a missing or symbolic-link shard"
        )
    return shards, weight_map


def _checkpoint_tensor_files(root: Path) -> dict[str, Path]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - GPTQModel requires safetensors
        raise GPTQCheckpointValidationError(
            "safetensors is required to validate a GPTQ checkpoint"
        ) from exc

    tensor_files: dict[str, Path] = {}
    shards, weight_map = _checkpoint_layout(root)
    for shard in shards:
        try:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                keys = tuple(handle.keys())
        except (OSError, RuntimeError, ValueError) as exc:
            raise GPTQCheckpointValidationError(
                f"GPTQ safetensors shard could not be inspected: {shard.name}"
            ) from exc
        for key in keys:
            if key in tensor_files:
                raise GPTQCheckpointValidationError(
                    f"GPTQ checkpoint tensor is declared more than once: {key}"
                )
            tensor_files[key] = shard
    if not tensor_files:
        raise GPTQCheckpointValidationError("GPTQ checkpoint contains no tensors")
    if weight_map is not None:
        declared_keys = set(weight_map)
        actual_keys = set(tensor_files)
        if declared_keys != actual_keys:
            missing = sorted(declared_keys - actual_keys)
            undeclared = sorted(actual_keys - declared_keys)
            raise GPTQCheckpointValidationError(
                "GPTQ safetensors index does not exactly describe shard tensors: "
                f"missing={missing[:5]}, undeclared={undeclared[:5]}"
            )
        misplaced = sorted(
            key for key, shard in tensor_files.items() if shard.name != weight_map[key]
        )
        if misplaced:
            raise GPTQCheckpointValidationError(
                "GPTQ safetensors index assigns tensors to the wrong shard: "
                f"{misplaced[:5]}"
            )
    return tensor_files


def _unbound_bias_keys(runtime_model: Any) -> set[str]:
    allowed: set[str] = set()
    named_modules = getattr(runtime_model, "named_modules", None)
    if not callable(named_modules):
        raise GPTQCheckpointValidationError(
            "loaded GPTQ model does not expose named modules"
        )
    for name, module in named_modules():
        module_path = type(module).__module__
        if not module_path.startswith("gptqmodel.nn_modules.qlinear."):
            continue
        if getattr(module, "bias", None) is not None:
            continue
        if isinstance(name, str) and name:
            allowed.add(f"{name}.bias")
    return allowed


def validate_gptq_checkpoint_bindings(model: Any) -> tuple[str, ...]:
    """Reject checkpoint-only tensors unless they are finite all-zero QLinear biases.

    GPTQModel may omit serialized bias tensors when it replaces dense projections
    with packed QLinear modules. Those tensors are semantically inert only when the
    loaded module has no bias and the serialized tensor is exactly finite zero.
    """

    local_path = getattr(model, "model_local_path", None)
    if not isinstance(local_path, str) or not local_path:
        raise GPTQCheckpointValidationError(
            "loaded GPTQ model does not expose its resolved checkpoint path"
        )
    root = Path(local_path)
    if not root.is_dir() or root.is_symlink():
        raise GPTQCheckpointValidationError(
            "loaded GPTQ model checkpoint path is not a regular directory"
        )
    runtime_model = getattr(model, "model", None)
    state_dict = getattr(runtime_model, "state_dict", None)
    if runtime_model is None or not callable(state_dict):
        raise GPTQCheckpointValidationError(
            "loaded GPTQ model does not expose its runtime state"
        )

    tensor_files = _checkpoint_tensor_files(root)
    loaded_keys = set(state_dict().keys())
    ignored_keys = set(tensor_files) - loaded_keys
    if not ignored_keys:
        return ()

    allowed_keys = _unbound_bias_keys(runtime_model)
    unexpected = sorted(ignored_keys - allowed_keys)
    if unexpected:
        raise GPTQCheckpointValidationError(
            "GPTQ checkpoint contains loader-ignored tensors outside the exact "
            f"unbound QLinear bias set: {unexpected[:5]}"
        )

    from safetensors import safe_open

    handles: dict[Path, Any] = {}
    with ExitStack() as stack:
        for key in sorted(ignored_keys):
            shard = tensor_files[key]
            handle = handles.get(shard)
            if handle is None:
                handle = stack.enter_context(
                    safe_open(str(shard), framework="pt", device="cpu")
                )
                handles[shard] = handle
            tensor = handle.get_tensor(key)
            try:
                import torch

                safe = bool(
                    tensor.numel() > 0
                    and torch.is_floating_point(tensor)
                    and torch.isfinite(tensor).all()
                    and torch.count_nonzero(tensor) == 0
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise GPTQCheckpointValidationError(
                    f"GPTQ loader-ignored tensor could not be validated: {key}"
                ) from exc
            if not safe:
                raise GPTQCheckpointValidationError(
                    "GPTQ loader-ignored bias must be a non-empty finite all-zero "
                    f"floating tensor: {key}"
                )
    return tuple(sorted(ignored_keys))


__all__ = [
    "GPTQCheckpointValidationError",
    "validate_gptq_checkpoint_bindings",
]
