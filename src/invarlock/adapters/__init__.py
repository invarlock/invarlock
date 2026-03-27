"""Adapter namespace (`invarlock.adapters`) exposing built-in adapters."""

from __future__ import annotations

import importlib as _importlib
from typing import TYPE_CHECKING
from typing import Any as _Any

from invarlock.core.abi import INVARLOCK_CORE_ABI as INVARLOCK_CORE_ABI

from .base import (
    AdapterConfig,
    AdapterInterface,
    BaseAdapter,
    DeviceManager,
)
from .base import (
    PerformanceMetrics as BasePerformanceMetrics,
)
from .capabilities import (
    ModelCapabilities,
    QuantizationConfig,
    QuantizationMethod,
    detect_capabilities_from_model,
    detect_quantization_from_config,
)

_LAZY_MAP = {
    "HF_Causal_Adapter": ".hf_causal",
    "HF_MLM_Adapter": ".hf_mlm",
    "HF_Seq2Seq_Adapter": ".hf_seq2seq",
    "HF_Auto_Adapter": ".auto",
}

if TYPE_CHECKING:  # pragma: no cover - typing aid for lazy exports
    from .auto import HF_Auto_Adapter
    from .hf_causal import HF_Causal_Adapter
    from .hf_mlm import HF_MLM_Adapter
    from .hf_seq2seq import HF_Seq2Seq_Adapter


def __getattr__(name: str) -> _Any:  # pragma: no cover - simple lazy import
    mod_name = _LAZY_MAP.get(name)
    if not mod_name:
        raise AttributeError(name)
    module = _importlib.import_module(mod_name, __name__)
    try:
        return getattr(module, name)
    except AttributeError as exc:  # re-raise with module context
        raise AttributeError(f"{name} not found in {mod_name}") from exc


# Simple quality label helper used by tests
def quality_label(ratio: float) -> str:
    if ratio <= 1.10:
        return "Excellent"
    if ratio <= 1.25:
        return "Good"
    if ratio <= 1.40:
        return "Fair"
    return "Degraded"


__all__ = [
    "HF_Causal_Adapter",
    "HF_MLM_Adapter",
    "HF_Seq2Seq_Adapter",
    "HF_Auto_Adapter",
    "BaseAdapter",
    "AdapterConfig",
    "AdapterInterface",
    "DeviceManager",
    "BasePerformanceMetrics",
    "quality_label",
    "INVARLOCK_CORE_ABI",
    # Capabilities
    "ModelCapabilities",
    "QuantizationConfig",
    "QuantizationMethod",
    "detect_capabilities_from_model",
    "detect_quantization_from_config",
]
