"""
InvarLock Adapters Base
===================

Base adapter interface and shared utilities for InvarLock adapters.
"""

import contextlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from invarlock.utils import get_memory_usage

TensorType = torch.Tensor
ModuleType = nn.Module


def _collect_memory_usage() -> dict[str, float]:
    """Collect process (and optional CUDA) memory usage in megabytes."""
    usage = get_memory_usage()
    memory_mb = float(usage.get("rss_mb", 0.0))
    result: dict[str, float] = {
        "memory_mb": memory_mb,
        "rss_mb": float(usage.get("rss_mb", 0.0)),
        "vms_mb": float(usage.get("vms_mb", 0.0)),
    }

    if torch.cuda.is_available():
        result["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
        result["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)

    return result


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
    metrics: dict[str, Any] = field(default_factory=dict, repr=False)

    def __getitem__(self, key: str) -> Any:
        if key in self.metrics:
            return self.metrics[key]
        return getattr(self, key, {})

    def __contains__(self, key: str) -> bool:
        return key in self.metrics or hasattr(self, key)


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


def _resolve_snapshot_member_path(
    snapshot_dir: Path,
    filename: str,
    *,
    entry_kind: str,
    entry_name: str,
) -> Path:
    owner = f"{entry_kind}: {entry_name}"
    if not isinstance(filename, str) or not filename:
        raise TypeError(f"Invalid snapshot manifest filename for {owner}")
    if (
        Path(filename).is_absolute()
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
    ):
        raise ValueError(f"Invalid snapshot manifest filename for {owner}")

    file_path = snapshot_dir / filename
    try:
        snapshot_root = snapshot_dir.resolve(strict=True)
        resolved_file = file_path.resolve(strict=True)
    except FileNotFoundError:
        return file_path
    if not resolved_file.is_relative_to(snapshot_root):
        raise ValueError(
            f"Snapshot manifest filename escapes snapshot directory for {owner}"
        )
    return file_path


def _record_snapshot_member_filename(
    seen_filenames: dict[str, str],
    filename: str,
    *,
    entry_kind: str,
    entry_name: str,
) -> None:
    owner = f"{entry_kind}:{entry_name}"
    previous_owner = seen_filenames.get(filename)
    if previous_owner is not None:
        raise ValueError(
            f"Duplicate snapshot manifest filename {filename!r} for {owner}; "
            f"already used by {previous_owner}"
        )
    seen_filenames[filename] = owner


class AdapterInterface(ABC):
    """Abstract adapter interface."""

    @abstractmethod
    def load_model(self, model_id: str, **kwargs) -> ModuleType | Any:
        """Load a model."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text."""
        pass

    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> dict[str, Any]:
        """Tokenize text."""
        pass

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Get adapter capabilities."""
        pass


class BaseAdapter(AdapterInterface):
    """Base adapter implementation."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.state = AdapterState.INITIALIZED
        self._monitoring_enabled = False
        self._performance_metrics = PerformanceMetrics()

    def cleanup(self) -> None:
        """Cleanup adapter resources."""
        pass

    def enable_monitoring(self) -> None:
        """Enable performance monitoring."""
        self._monitoring_enabled = True

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        return self._performance_metrics

    def get_memory_usage(self) -> dict[str, float]:
        """Get memory usage information."""
        return _collect_memory_usage()

    @abstractmethod
    def load_model(self, model_id: str, **kwargs) -> ModuleType | Any:
        """Load a model."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text."""
        pass

    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> dict[str, Any]:
        """Tokenize text."""
        pass

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Get adapter capabilities."""
        pass


class AdapterConfig:
    """Adapter configuration management."""

    def __init__(
        self,
        name: str,
        adapter_type: str,
        version: str = "0.2.0",
        device: dict[str, Any] | None = None,
        cache: dict[str, Any] | None = None,
        monitoring: dict[str, Any] | None = None,
        optimization: dict[str, Any] | None = None,
    ):
        self.name = name
        self.version = version
        self.adapter_type = adapter_type
        self.device = device or {"type": "auto"}
        self.cache = cache or {"enabled": True}
        self.monitoring = monitoring or {"enabled": True}
        self.optimization = optimization or {"enabled": False}

    def validate(self) -> dict[str, Any]:
        """Validate configuration."""
        valid = True
        errors = []

        if self.device.get("memory_fraction", 0.8) > 1.0:
            valid = False
            errors.append("memory_fraction must be <= 1.0")

        return {"valid": valid, "errors": errors}

    def resolve_device(self) -> str:
        """Resolve device configuration."""
        index = int(self.device.get("index", 0))

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count <= 0:
                return "cuda:0"
            index = max(0, min(index, device_count - 1))
            return f"cuda:{index}"

        return "cpu"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "adapter_type": self.adapter_type,
            "device": self.device,
            "cache": self.cache,
            "monitoring": self.monitoring,
            "optimization": self.optimization,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "AdapterConfig":
        """Create from dictionary."""
        return cls(**config_dict)


class DeviceManager:
    """Device management utilities."""

    def __init__(self, device_config: dict[str, Any]):
        self.device_type = device_config.get("type", "auto")
        self.device_index = device_config.get("index", 0)
        self.memory_fraction = device_config.get("memory_fraction", 0.8)
        self.allow_growth = device_config.get("allow_growth", True)

    def get_available_devices(self) -> list[str]:
        """Get available devices."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        return devices

    def get_memory_info(self) -> dict[str, float]:
        """Get memory information."""
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_idx = (
                self.device_index
                if self.device_index < torch.cuda.device_count()
                else 0
            )
            return {
                "total_mb": torch.cuda.get_device_properties(device_idx).total_memory
                / (1024**2),
                "allocated_mb": torch.cuda.memory_allocated(device_idx) / (1024**2),
                "reserved_mb": torch.cuda.memory_reserved(device_idx) / (1024**2),
            }
        usage = _collect_memory_usage()
        return {
            "total_mb": usage.get("memory_mb", 0.0),
            "allocated_mb": usage.get("memory_mb", 0.0),
            "reserved_mb": usage.get("vms_mb", 0.0),
        }

    def set_memory_fraction(self, fraction: float) -> None:
        """Set memory fraction."""
        self.memory_fraction = fraction

    def set_memory_growth(self, allow: bool) -> None:
        """Set memory growth."""
        self.allow_growth = allow

    @contextlib.contextmanager
    def device_context(self, device: str):
        """Device context manager."""
        if device.startswith("cuda") and torch.cuda.is_available():
            with torch.cuda.device(device):
                yield
                return

        yield


class AdapterCache:
    """Adapter caching functionality."""

    def __init__(self, cache_config: dict[str, Any]):
        self.enabled = cache_config.get("enabled", True)
        self.max_size_mb = cache_config.get("max_size_mb", 1024)
        self.ttl_seconds = cache_config.get("ttl_seconds", 3600)
        self._cache: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}

    def put(self, key: str, value: Any):
        """Put value in cache."""
        if self.enabled:
            self._cache[key] = value
            self._timestamps[key] = time.time()

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if not self.enabled or key not in self._cache:
            return None

        # Check TTL
        if time.time() - self._timestamps[key] > self.ttl_seconds:
            del self._cache[key]
            del self._timestamps[key]
            return None

        return self._cache[key]


class PerformanceTracker:
    """Performance tracking functionality."""

    def __init__(self, monitor_config: dict[str, Any]):
        self.enabled = monitor_config.get("enabled", True)
        self.track_performance = monitor_config.get("track_performance", True)
        self.track_memory = monitor_config.get("track_memory", True)
        self._metrics: dict[str, Any] = {}

    @contextlib.contextmanager
    def time_operation(self, operation_name: str):
        """Time an operation."""
        if not self.enabled or not self.track_performance:
            yield
            return

        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if operation_name not in self._metrics:
                self._metrics[operation_name] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "durations": [],
                }

            self._metrics[operation_name]["count"] += 1
            self._metrics[operation_name]["total_duration"] += duration
            self._metrics[operation_name]["durations"].append(duration)
            self._metrics[operation_name]["duration"] = duration
            self._metrics[operation_name]["average_duration"] = (
                self._metrics[operation_name]["total_duration"]
                / self._metrics[operation_name]["count"]
            )
            self._metrics[operation_name]["min_duration"] = min(
                self._metrics[operation_name]["durations"]
            )
            self._metrics[operation_name]["max_duration"] = max(
                self._metrics[operation_name]["durations"]
            )

    def record_memory_usage(self, label: str):
        """Record memory usage."""
        if not self.enabled or not self.track_memory:
            return
        if "memory_usage" not in self._metrics:
            self._metrics["memory_usage"] = {}
        self._metrics["memory_usage"][label] = _collect_memory_usage()

    def get_metrics(self) -> dict[str, Any]:
        """Get all metrics."""
        return self._metrics

    def export_metrics(self, path: Path):
        """Export metrics to file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._metrics, f, indent=2, allow_nan=False)


class AdapterManager:
    """Adapter manager for multiple adapters."""

    def __init__(self):
        self.adapters = {}

    def register(self, name: str, adapter: BaseAdapter):
        """Register an adapter."""
        self.adapters[name] = adapter

    def get(self, name: str) -> BaseAdapter | None:
        """Get an adapter."""
        return self.adapters.get(name)

    def list_adapters(self) -> list[str]:
        """List all adapter names."""
        return list(self.adapters.keys())

    def initialize_adapter(self, name: str, model_id: str):
        """Initialize a specific adapter."""
        adapter = self.adapters.get(name)
        if adapter:
            adapter.load_model(model_id)
            adapter.state = AdapterState.LOADED

    def cleanup_adapter(self, name: str):
        """Cleanup a specific adapter."""
        adapter = self.adapters.get(name)
        if adapter:
            adapter.cleanup()

    def initialize_all(self, model_id: str):
        """Initialize all adapters."""
        for name in self.adapters:
            self.initialize_adapter(name, model_id)

    def cleanup_all(self):
        """Cleanup all adapters."""
        for name in self.adapters:
            self.cleanup_adapter(name)

    def check_adapter_health(self, name: str) -> dict[str, Any]:
        """Check adapter health."""
        adapter = self.adapters.get(name)
        if adapter:
            return {"status": "healthy", "state": adapter.state.value}
        return {"status": "not_found"}

    def check_overall_health(self) -> dict[str, Any]:
        """Check overall health."""
        adapters_health = {}
        for name in self.adapters:
            adapters_health[name] = self.check_adapter_health(name)
        return {"adapters": adapters_health}
