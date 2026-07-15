"""Narrow runtime boundary for dense PEFT LoRA construction.

PEFT probes optional quantization dispatchers before its ordinary dense
dispatcher.  Consequently, an installed but API-incompatible GPTQModel can
break LoRA construction for a model that contains no quantized modules.  This
module isolates those irrelevant probes only after proving that the complete
input model is dense.  Quantized models remain on their native backend path.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

_PEFT_RUNTIME_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class PeftRuntimeError(RuntimeError):
    """Raised when the named PEFT construction boundary cannot be trusted."""


def normalize_base_key(name: str) -> str:
    """Normalize PEFT wrapper paths to serialized base-model state names."""

    while name.startswith("base_model."):
        name = name.removeprefix("base_model.")
    if name.startswith("model."):
        name = name.removeprefix("model.")
    return name.replace(".base_layer.", ".")


def peft_base_state(model: Any) -> dict[str, Any]:
    """Return live base tensor references, excluding adapter-only state."""

    state: dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        if "lora_" in name or "modules_to_save" in name:
            continue
        normalized = normalize_base_key(name)
        if normalized in state:
            raise PeftRuntimeError(f"ambiguous PEFT base tensor name: {normalized}")
        state[normalized] = tensor
    return state


def peft_merge_target_names(
    model: Any, baseline_state: Mapping[str, Any]
) -> frozenset[str]:
    """Derive exact serialized weights selected by the installed LoRA modules."""

    targets: set[str] = set()
    for name in model.state_dict():
        if ".lora_" not in name:
            continue
        module = normalize_base_key(name.split(".lora_", maxsplit=1)[0])
        candidate = f"{module}.weight"
        if candidate in baseline_state:
            targets.add(candidate)
    if not targets:
        raise PeftRuntimeError("PEFT adapter selected no merge target tensors")
    return frozenset(targets)


def peft_base_snapshot(model: Any, baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot fixture-sized base state for compatibility tests."""

    state: dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        if "lora_" in name or "modules_to_save" in name:
            continue
        normalized = normalize_base_key(name)
        if normalized in state:
            raise PeftRuntimeError(f"ambiguous PEFT base tensor name: {normalized}")
        state[normalized] = tensor.detach().cpu().clone()
    if set(state) != set(baseline):
        missing = sorted(set(baseline) - set(state))[:3]
        extra = sorted(set(state) - set(baseline))[:3]
        raise PeftRuntimeError(
            f"PEFT base state does not match baseline keys; missing={missing}, extra={extra}"
        )
    return state


def adapter_module_count(state: Mapping[str, Any]) -> int:
    modules = {
        name.split("lora_", maxsplit=1)[0].rstrip(".")
        for name in state
        if "lora_" in name
    }
    return len(modules)


def dense_model_quantization_markers(model: Any) -> tuple[str, ...]:
    """Return stable reasons why ``model`` cannot use the dense-only path."""

    markers: set[str] = set()
    config = getattr(model, "config", None)
    if getattr(config, "quantization_config", None) is not None:
        markers.add("config.quantization_config")
    if getattr(model, "is_quantized", False) is True:
        markers.add("model.is_quantized")
    if getattr(model, "hf_quantizer", None) is not None:
        markers.add("model.hf_quantizer")
    if getattr(model, "quantization_method", None) is not None:
        markers.add("model.quantization_method")

    modules = getattr(model, "modules", None)
    if not callable(modules):
        markers.add("model.modules unavailable")
        return tuple(sorted(markers))
    try:
        quantized_weights = importlib.import_module(
            "invarlock.guards.quantized_weights"
        )
        is_packed_quantized_module = quantized_weights.is_packed_quantized_module
        if not callable(is_packed_quantized_module):
            markers.add("packed-module inspection unavailable")
            return tuple(sorted(markers))
        model_modules = modules()
        for module in model_modules:
            module_name = type(module).__module__.lower()
            if is_packed_quantized_module(module):
                markers.add(f"packed-module:{module_name}")
    except _PEFT_RUNTIME_ERRORS as exc:
        markers.add(f"packed-module inspection failed:{type(exc).__name__}")
    return tuple(sorted(markers))


def _configure_dense_dispatch(config: Any) -> None:
    """Put PEFT's own dense dispatcher first on this configuration only."""

    try:
        torch = importlib.import_module("torch")
        transformers_layers = importlib.import_module("transformers.pytorch_utils")
        lora_layer = importlib.import_module("peft.tuners.lora.layer")
    except _PEFT_RUNTIME_ERRORS as exc:
        raise PeftRuntimeError(
            "PEFT dense dispatcher boundary is unavailable: " + type(exc).__name__
        ) from exc
    register = getattr(config, "_register_custom_module", None)
    dispatch_default = getattr(lora_layer, "dispatch_default", None)
    conv1d = getattr(transformers_layers, "Conv1D", None)
    nn = getattr(torch, "nn", None)
    supported_names = (
        "Embedding",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "MultiheadAttention",
        "Linear",
    )
    if not callable(register) or not callable(dispatch_default) or nn is None:
        raise PeftRuntimeError("PEFT dense dispatcher API is incompatible")
    supported_types = tuple(getattr(nn, name, None) for name in supported_names)
    if conv1d is None or any(not isinstance(item, type) for item in supported_types):
        raise PeftRuntimeError("PEFT dense layer type inventory is incompatible")

    def dense_dispatch(
        target: Any,
        adapter_name: str,
        config: Any,
        **kwargs: Any,
    ) -> Any:
        result = dispatch_default(
            target,
            adapter_name,
            config=config,
            **kwargs,
        )
        if result is None:
            raise PeftRuntimeError(
                "PEFT dense dispatcher rejected a declared dense target"
            )
        return result

    mapping = dict.fromkeys((*supported_types, conv1d), dense_dispatch)
    try:
        register(mapping)
    except _PEFT_RUNTIME_ERRORS as exc:
        raise PeftRuntimeError(
            "PEFT dense dispatcher registration failed: " + type(exc).__name__
        ) from exc


def get_dense_peft_model(model: Any, config: Any, *, get_peft_model: Any) -> Any:
    """Construct dense LoRA without consulting irrelevant quantized backends.

    This function deliberately rejects every observed quantization marker.  It
    is not a compatibility shortcut for GPTQ/AWQ LoRA: those models must use
    PEFT's native quantized dispatchers and prove that integration separately.
    """

    markers = dense_model_quantization_markers(model)
    if markers:
        raise PeftRuntimeError(
            "dense PEFT LoRA boundary rejected a quantized or uninspectable model: "
            + ", ".join(markers)
        )
    if not callable(get_peft_model):
        raise PeftRuntimeError("PEFT get_peft_model entry point is not callable")
    _configure_dense_dispatch(config)
    try:
        return get_peft_model(model, config)
    except PeftRuntimeError:
        raise
    except _PEFT_RUNTIME_ERRORS as exc:
        raise PeftRuntimeError(
            "PEFT dense LoRA construction failed: " + type(exc).__name__
        ) from exc


def load_dense_peft_model(
    model: Any,
    config: Any,
    adapter_path: Any,
    *,
    from_pretrained: Any,
    **options: Any,
) -> Any:
    """Reload serialized dense LoRA without probing quantized dispatchers.

    The caller supplies the configuration reloaded from the serialized adapter,
    so this boundary preserves the save/reload proof while applying the same
    dense-model validation used during initial adapter construction.
    """

    markers = dense_model_quantization_markers(model)
    if markers:
        raise PeftRuntimeError(
            "dense PEFT LoRA reload rejected a quantized or uninspectable model: "
            + ", ".join(markers)
        )
    if not callable(from_pretrained):
        raise PeftRuntimeError("PEFT from_pretrained entry point is not callable")
    _configure_dense_dispatch(config)
    try:
        return from_pretrained(
            model,
            adapter_path,
            config=config,
            **options,
        )
    except PeftRuntimeError:
        raise
    except _PEFT_RUNTIME_ERRORS as exc:
        raise PeftRuntimeError(
            "PEFT dense LoRA reload failed: " + type(exc).__name__
        ) from exc


__all__ = [
    "PeftRuntimeError",
    "adapter_module_count",
    "dense_model_quantization_markers",
    "get_dense_peft_model",
    "load_dense_peft_model",
    "normalize_base_key",
    "peft_base_snapshot",
    "peft_base_state",
    "peft_merge_target_names",
]
