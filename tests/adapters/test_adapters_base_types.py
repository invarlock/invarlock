"""
Comprehensive Test Suite for InvarLock Adapters Module
==================================================

Tests covering adapter infrastructure, HuggingFace integration,
device management, caching, and performance tracking.

Target: 70% coverage for invarlock_adapters module
"""

import time
from unittest.mock import patch

import pytest

# Import invarlock adapters namespace
from invarlock.adapters.base import (
    AdapterCache,
    AdapterConfig,
    AdapterState,
    AdapterType,
    BaseAdapter,
    CacheConfig,
    DeviceManager,
    DeviceType,
    MonitorConfig,
    PerformanceMetrics,
)
from invarlock.adapters.base import PerformanceMetrics as BasePerformanceMetrics
from tests.adapters._support_base_types import (
    ConcreteAdapter,
)


class TestBaseTypes:
    """Test base type definitions."""

    def test_adapter_type_enum(self):
        """Test AdapterType enum."""
        assert AdapterType.TRANSFORMER.value == "transformer"
        assert AdapterType.GENERIC.value == "generic"
        assert AdapterType.HUGGINGFACE.value == "huggingface"
        assert AdapterType.OPENAI.value == "openai"

        # Test enum iteration
        types = list(AdapterType)
        assert len(types) == 4
        assert AdapterType.TRANSFORMER in types

    def test_device_type_enum(self):
        """Test DeviceType enum."""
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.AUTO.value == "auto"

    def test_adapter_state_enum(self):
        """Test AdapterState enum."""
        assert AdapterState.INITIALIZED.value == "initialized"
        assert AdapterState.LOADED.value == "loaded"
        assert AdapterState.ERROR.value == "error"
        assert AdapterState.READY.value == "ready"

    def test_performance_metrics_dataclass(self):
        """Test PerformanceMetrics dataclass."""
        metrics = PerformanceMetrics()

        assert metrics.operation_count == 0
        assert metrics.total_duration == 0.0
        assert metrics.average_duration == 0.0
        assert metrics.memory_usage_mb == 0.0

        # Test dict-like access
        assert metrics["operation_count"] == 0
        assert "total_duration" in metrics
        assert "nonexistent" not in metrics

        # Test with custom values
        custom_metrics = PerformanceMetrics(
            operation_count=5,
            total_duration=10.5,
            average_duration=2.1,
            memory_usage_mb=256.0,
        )
        assert custom_metrics.operation_count == 5
        assert custom_metrics["total_duration"] == 10.5

    def test_cache_config_dataclass(self):
        """Test CacheConfig dataclass."""
        config = CacheConfig()

        assert config.enabled is True
        assert config.max_size_mb == 1024
        assert config.ttl_seconds == 3600
        assert config.cache_dir is None

        # Custom config
        custom_config = CacheConfig(
            enabled=False, max_size_mb=512, ttl_seconds=1800, cache_dir="/tmp/cache"
        )
        assert custom_config.enabled is False
        assert custom_config.cache_dir == "/tmp/cache"

    def test_monitor_config_dataclass(self):
        """Test MonitorConfig dataclass."""
        config = MonitorConfig()

        assert config.enabled is True
        assert config.track_performance is True
        assert config.track_memory is True
        assert config.log_level == "INFO"


class TestBaseAdapter:
    """Test base adapter infrastructure."""

    def test_base_adapter_creation(self):
        """Test BaseAdapter creation."""
        config = {"name": "test_adapter", "device": "cpu"}
        adapter = ConcreteAdapter(config)

        assert adapter.config == config
        assert adapter.state.value == AdapterState.INITIALIZED.value
        assert adapter._monitoring_enabled is False
        assert isinstance(
            adapter._performance_metrics, PerformanceMetrics | BasePerformanceMetrics
        )

    def test_base_adapter_monitoring(self):
        """Test adapter monitoring functionality."""
        adapter = ConcreteAdapter({})

        # Initially disabled
        assert adapter._monitoring_enabled is False

        # Enable monitoring
        adapter.enable_monitoring()
        assert adapter._monitoring_enabled is True

        # Get metrics
        metrics = adapter.get_performance_metrics()
        assert isinstance(metrics, PerformanceMetrics | BasePerformanceMetrics)

        # Get memory usage
        memory = adapter.get_memory_usage()
        assert isinstance(memory, dict)
        assert "memory_mb" in memory

    def test_base_adapter_cleanup(self):
        """Test adapter cleanup."""
        adapter = ConcreteAdapter({})
        adapter.cleanup()
        assert adapter.state == AdapterState.INITIALIZED

    def test_concrete_adapter_methods(self):
        """Test concrete adapter method implementations."""
        adapter = ConcreteAdapter({})

        # Test model loading
        result = adapter.load_model("test_model")
        assert adapter.state == AdapterState.LOADED
        assert result["model_id"] == "test_model"

        # Test generation
        response = adapter.generate("test prompt")
        assert "Generated response for: test prompt" == response

        # Test tokenization
        tokens = adapter.tokenize("hello world")
        assert tokens["tokens"] == ["hello", "world"]

        # Test capabilities
        caps = adapter.get_capabilities()
        assert caps["supports_generation"] is True

    def test_base_adapter_abstract_nature(self):
        """Test that BaseAdapter is abstract."""
        with pytest.raises(TypeError, match="abstract"):
            BaseAdapter({})


class TestAdapterConfig:
    """Test adapter configuration."""

    def test_adapter_config_creation(self):
        """Test AdapterConfig creation."""
        config = AdapterConfig(
            name="test_adapter", adapter_type="transformer", version="1.0.0"
        )

        assert config.name == "test_adapter"
        assert config.adapter_type == "transformer"
        assert config.version == "1.0.0"
        assert config.device == {"type": "auto"}
        assert config.cache == {"enabled": True}
        assert config.monitoring == {"enabled": True}
        assert config.optimization == {"enabled": False}

    def test_adapter_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = AdapterConfig("test", "transformer")
        result = config.validate()
        assert result["valid"] is True
        assert result["errors"] == []

        # Invalid config with high memory fraction
        config.device = {"memory_fraction": 1.5}
        result = config.validate()
        assert result["valid"] is False
        assert "memory_fraction" in result["errors"][0]

    @patch("torch.cuda.is_available", return_value=True)
    def test_adapter_config_device_resolution_cuda(self, mock_cuda):
        """Test device resolution with CUDA available."""
        config = AdapterConfig("test", "transformer")
        device = config.resolve_device()
        assert device == "cuda:0"

    @patch("torch.cuda.is_available", return_value=False)
    def test_adapter_config_device_resolution_cpu(self, mock_cuda):
        """Test device resolution with CUDA unavailable."""
        config = AdapterConfig("test", "transformer")
        device = config.resolve_device()
        assert device == "cpu"

    def test_adapter_config_serialization(self):
        """Test config serialization."""
        config = AdapterConfig(
            "test",
            "transformer",
            "2.0.0",
            device={"type": "cuda"},
            cache={"enabled": False},
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["name"] == "test"
        assert config_dict["version"] == "2.0.0"
        assert config_dict["device"]["type"] == "cuda"

        # Test from_dict
        new_config = AdapterConfig.from_dict(config_dict)
        assert new_config.name == config.name
        assert new_config.device == config.device


class TestDeviceManager:
    """Test device management."""

    def test_device_manager_creation(self):
        """Test DeviceManager creation."""
        device_config = {
            "type": "cuda",
            "index": 1,
            "memory_fraction": 0.5,
            "allow_growth": False,
        }

        manager = DeviceManager(device_config)
        assert manager.device_type == "cuda"
        assert manager.device_index == 1
        assert manager.memory_fraction == 0.5
        assert manager.allow_growth is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    def test_device_manager_available_devices_cuda(self, mock_count, mock_available):
        """Test available devices with CUDA."""
        manager = DeviceManager({})
        devices = manager.get_available_devices()

        assert "cpu" in devices
        assert "cuda:0" in devices
        assert "cuda:1" in devices
        assert len(devices) == 3

    @patch("torch.cuda.is_available", return_value=False)
    def test_device_manager_available_devices_cpu(self, mock_available):
        """Test available devices without CUDA."""
        manager = DeviceManager({})
        devices = manager.get_available_devices()

        assert devices == ["cpu"]

    def test_device_manager_memory_info(self):
        """Test memory information retrieval."""
        manager = DeviceManager({})
        memory_info = manager.get_memory_info()

        assert isinstance(memory_info, dict)
        assert "total_mb" in memory_info
        assert "allocated_mb" in memory_info
        assert "reserved_mb" in memory_info

    def test_device_manager_settings(self):
        """Test device settings modification."""
        manager = DeviceManager({})

        # Test memory fraction setting
        manager.set_memory_fraction(0.7)
        assert manager.memory_fraction == 0.7

        # Test memory growth setting
        manager.set_memory_growth(True)
        assert manager.allow_growth is True

    def test_device_manager_context(self):
        """Test device context manager."""
        manager = DeviceManager({})

        entered = False
        with manager.device_context("cuda:0"):
            entered = True
        assert entered is True


class TestAdapterCache:
    """Test adapter caching."""

    def test_cache_creation(self):
        """Test cache creation."""
        cache_config = {"enabled": True, "max_size_mb": 512, "ttl_seconds": 1800}

        cache = AdapterCache(cache_config)
        assert cache.enabled is True
        assert cache.max_size_mb == 512
        assert cache.ttl_seconds == 1800

    def test_cache_operations(self):
        """Test cache put/get operations."""
        cache = AdapterCache({"enabled": True, "ttl_seconds": 10})

        # Test put/get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Test non-existent key
        assert cache.get("nonexistent") is None

        # Test disabled cache
        cache.enabled = False
        cache.put("key2", "value2")
        assert cache.get("key2") is None

    def test_cache_ttl(self):
        """Test cache TTL expiration."""
        cache = AdapterCache({"enabled": True, "ttl_seconds": 1})

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Simulate time passing
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_cache_save_load(self):
        """Test cache save/load (stubs)."""
        cache = AdapterCache({"enabled": True})

        cache.save()
        cache.load()
        assert cache.get("missing") is None
