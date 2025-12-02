"""
InvarLock Adapters Base Types
=========================

Type definitions for the InvarLock adapter system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AdapterType(Enum):
    """Adapter type enumeration."""

    TRANSFORMER = "transformer"
    GENERIC = "generic"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class DeviceType(Enum):
    """Device type enumeration."""

    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class AdapterState(Enum):
    """Adapter state enumeration."""

    INITIALIZED = "initialized"
    LOADED = "loaded"
    ERROR = "error"
    READY = "ready"


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""

    operation_count: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    memory_usage_mb: float = 0.0

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return getattr(self, key, {})

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator."""
        return hasattr(self, key)


@dataclass
class CacheConfig:
    """Cache configuration."""

    enabled: bool = True
    max_size_mb: int = 1024
    ttl_seconds: int = 3600
    cache_dir: str | None = None


@dataclass
class MonitorConfig:
    """Monitor configuration."""

    enabled: bool = True
    track_performance: bool = True
    track_memory: bool = True
    log_level: str = "INFO"


# Export all types
__all__ = [
    "AdapterType",
    "DeviceType",
    "AdapterState",
    "PerformanceMetrics",
    "CacheConfig",
    "MonitorConfig",
]
