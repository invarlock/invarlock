from __future__ import annotations

from typing import Any

import torch

_QUANTIZED_CLASS_MARKERS = (
    "affinequantized",
    "awq",
    "bitsandbytes",
    "bnb",
    "compressed_tensors",
    "compressedtensors",
    "eetq",
    "exllama",
    "fbgemm",
    "fp8",
    "gptq",
    "hqq",
    "llmcompressor",
    "marlin",
    "optimum.quanto",
    "quanto",
    "quantizedtensor",
    "quantized_tensor",
    "aqlm",
)

_PACKED_QUANTIZED_ATTRS = (
    "qweight",
    "qzeros",
    "scales",
    "SCB",
    "CB",
    "packed_weight",
    "packed_weights",
)


def _known_quantized_dtypes() -> set[torch.dtype]:
    names = ("int8", "uint8", "qint8", "quint8", "qint32")
    return {
        dtype
        for name in names
        if isinstance(dtype := getattr(torch, name, None), torch.dtype)
    }


def is_quantized_weight(weight: Any) -> bool:
    try:
        if bool(getattr(weight, "is_quantized", False)):
            return True
    except (RuntimeError, TypeError, ValueError):
        # guard-fallback-ok: unreadable quantized metadata fails closed as packed.
        return True

    dtype = getattr(weight, "dtype", None)
    if dtype in _known_quantized_dtypes():
        return True

    weight_type = type(weight)
    qualified_name = f"{weight_type.__module__}.{weight_type.__qualname__}".lower()
    return any(marker in qualified_name for marker in _QUANTIZED_CLASS_MARKERS)


def is_packed_quantized_module(module: Any) -> bool:
    weight = getattr(module, "weight", None)
    if weight is not None and is_quantized_weight(weight):
        return True

    module_type = type(module)
    qualified_name = f"{module_type.__module__}.{module_type.__qualname__}".lower()
    if any(marker in qualified_name for marker in _QUANTIZED_CLASS_MARKERS):
        return True

    for attr in _PACKED_QUANTIZED_ATTRS:
        try:
            value = getattr(module, attr, None)
        except (RuntimeError, TypeError, ValueError):
            # guard-fallback-ok: unreadable packed metadata fails closed.
            return True
        if value is not None:
            return True

    state_dict = getattr(module, "state_dict", None)
    if callable(state_dict):
        try:
            keys = set(state_dict().keys())
        except (RuntimeError, TypeError, ValueError):
            # guard-fallback-ok: unreadable packed state fails closed.
            return True
        if keys.intersection(_PACKED_QUANTIZED_ATTRS):
            return True

    return False


__all__ = ["is_packed_quantized_module", "is_quantized_weight"]
