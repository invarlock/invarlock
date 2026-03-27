"""Adapter namespace (`invarlock.adapters`) exposing safe built-in base types."""

from __future__ import annotations

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
