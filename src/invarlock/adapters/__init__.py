"""Adapter namespace (`invarlock.adapters`) exposing safe built-in base types."""

from __future__ import annotations

from invarlock.core import INVARLOCK_CORE_ABI as INVARLOCK_CORE_ABI

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

__all__ = [
    "BaseAdapter",
    "AdapterConfig",
    "AdapterInterface",
    "DeviceManager",
    "BasePerformanceMetrics",
    "INVARLOCK_CORE_ABI",
    # Capabilities
    "ModelCapabilities",
    "QuantizationConfig",
    "QuantizationMethod",
    "detect_capabilities_from_model",
    "detect_quantization_from_config",
]
