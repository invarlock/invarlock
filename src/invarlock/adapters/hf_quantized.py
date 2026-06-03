"""Shared structural introspection for HF quantized adapter loaders."""

from __future__ import annotations

from typing import Any

from invarlock.adapters.hf_causal import HF_Causal_Adapter


def describe_causal_quantized_model(model: Any) -> dict[str, Any]:
    """Describe a quantized causal model through the standard causal spec."""

    return HF_Causal_Adapter().describe(model)


def get_causal_quantized_layer_modules(model: Any, layer_idx: int) -> dict[str, Any]:
    """Expose layer modules through the standard causal adapter contract."""

    return HF_Causal_Adapter().get_layer_modules(model, layer_idx)


__all__ = [
    "describe_causal_quantized_model",
    "get_causal_quantized_layer_modules",
]
