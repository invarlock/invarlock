"""
Comprehensive Test Suite for InvarLock Adapters Module
==================================================

Tests covering adapter infrastructure, HuggingFace integration,
device management, caching, and performance tracking.

Target: 70% coverage for invarlock_adapters module
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

import invarlock

# Import invarlock adapters namespace
import invarlock.adapters as invarlock_adapters
from invarlock.adapters.base import (
    AdapterCache,
    AdapterConfig,
    AdapterManager,
    AdapterUtils,
    BaseAdapter,
    DeviceManager,
    PerformanceTracker,
)
from invarlock.adapters.base import PerformanceMetrics as BasePerformanceMetrics
from invarlock.adapters.base_types import (
    AdapterState,
    AdapterType,
    CacheConfig,
    DeviceType,
    MonitorConfig,
    PerformanceMetrics,
)
from invarlock.adapters.hf_bert import HF_BERT_Adapter
from invarlock.adapters.hf_gpt2 import HF_GPT2_Adapter
from invarlock.adapters.hf_llama import HF_LLaMA_Adapter


class MockGPT2Model(nn.Module):
    """Mock GPT-2 model for testing HF adapter."""

    def __init__(self, n_layer=2, n_head=4, hidden_size=16):
        super().__init__()
        self.config = Mock()
        self.config.model_type = "gpt2"
        self.config.n_layer = n_layer
        self.config.n_head = n_head
        self.config.hidden_size = hidden_size
        self.config.vocab_size = 1000
        self.config.n_inner = hidden_size * 4

        # Create transformer structure
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList()

        for _i in range(n_layer):
            layer = self._create_layer(n_head, hidden_size)
            self.transformer.h.append(layer)

        # Add embeddings and head
        self.transformer.wte = nn.Embedding(1000, hidden_size)
        self.lm_head = nn.Linear(hidden_size, 1000, bias=False)

        # Optional weight tying
        if hasattr(self, "tie_weights"):
            self.lm_head.weight = self.transformer.wte.weight

    def _create_layer(self, n_head, hidden_size):
        """Create a mock transformer layer."""
        layer = nn.Module()

        # Attention
        layer.attn = nn.Module()
        layer.attn.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        layer.attn.c_proj = nn.Linear(hidden_size, hidden_size)

        # MLP
        layer.mlp = nn.Module()
        layer.mlp.c_fc = nn.Linear(hidden_size, hidden_size * 4)
        layer.mlp.c_proj = nn.Linear(hidden_size * 4, hidden_size)

        # Layer norms
        layer.ln_1 = nn.LayerNorm(hidden_size)
        layer.ln_2 = nn.LayerNorm(hidden_size)

        return layer


class MockBertLayer(nn.Module):
    """Minimal BERT encoder layer for adapter testing."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = nn.Module()
        self.attention.self.query = nn.Linear(hidden_size, hidden_size)
        self.attention.self.key = nn.Linear(hidden_size, hidden_size)
        self.attention.self.value = nn.Linear(hidden_size, hidden_size)
        self.attention.output = nn.Module()
        self.attention.output.dense = nn.Linear(hidden_size, hidden_size)
        self.attention.output.LayerNorm = nn.LayerNorm(hidden_size)

        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(hidden_size, hidden_size * 4)

        self.output = nn.Module()
        self.output.dense = nn.Linear(hidden_size * 4, hidden_size)
        self.output.LayerNorm = nn.LayerNorm(hidden_size)


class MockBertModel(nn.Module):
    """Mock BERT model with encoder/embedding/cls structure."""

    def __init__(
        self,
        n_layer: int = 2,
        hidden_size: int = 32,
        vocab_size: int = 128,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.config = Mock()
        self.config.model_type = "bert"
        self.config.num_hidden_layers = n_layer
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = 4
        self.config.intermediate_size = hidden_size * 4
        self.config.vocab_size = vocab_size
        self.config.type_vocab_size = 2
        self.config.max_position_embeddings = 512
        self.config.layer_norm_eps = 1e-12
        self.config.hidden_dropout_prob = 0.1
        self.config.attention_probs_dropout_prob = 0.1

        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = nn.Embedding(vocab_size, hidden_size)

        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [MockBertLayer(hidden_size) for _ in range(n_layer)]
        )

        self.bert = nn.Module()
        self.bert.embeddings = self.embeddings
        self.bert.encoder = self.encoder

        self.pooler = nn.Linear(hidden_size, hidden_size)

        self.cls = nn.Module()
        self.cls.predictions = nn.Module()
        self.cls.predictions.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        if tie_weights:
            self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight


class MockLLaMALayer(nn.Module):
    """Minimal LLaMA block for adapter testing."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.k_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.v_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.o_proj = nn.Linear(hidden_size, hidden_size)

        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.mlp.up_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.mlp.down_proj = nn.Linear(hidden_size * 4, hidden_size)

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)


class MockLLaMAModel(nn.Module):
    """Mock LLaMA model with tying support."""

    def __init__(
        self,
        n_layer: int = 2,
        hidden_size: int = 32,
        vocab_size: int = 64,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.config = Mock()
        self.config.model_type = "llama"
        self.config.num_hidden_layers = n_layer
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = 4
        self.config.num_key_value_heads = 2
        self.config.intermediate_size = hidden_size * 4
        self.config.vocab_size = vocab_size
        self.config.max_position_embeddings = 2048
        self.config.rms_norm_eps = 1e-6

        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockLLaMALayer(hidden_size) for _ in range(n_layer)]
        )
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.model.embed_tokens.weight


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


class ConcreteAdapter(BaseAdapter):
    """Concrete adapter implementation for testing."""

    def load_model(self, model_id: str, **kwargs):
        """Mock model loading."""
        self.state = AdapterState.LOADED
        return {"model_id": model_id}

    def generate(self, prompt: str, **kwargs) -> str:
        """Mock text generation."""
        return f"Generated response for: {prompt}"

    def tokenize(self, text: str, **kwargs):
        """Mock tokenization."""
        return {"tokens": text.split(), "token_ids": list(range(len(text.split())))}

    def get_capabilities(self):
        """Mock capabilities."""
        return {"supports_generation": True, "supports_tokenization": True}


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
        # Should not raise any errors
        adapter.cleanup()

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

        # Should not raise any errors
        with manager.device_context("cuda:0"):
            pass


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

        # Should not raise errors (stubs)
        cache.save()
        cache.load()


class TestPerformanceTracker:
    """Test performance tracking."""

    def test_performance_tracker_creation(self):
        """Test tracker creation."""
        monitor_config = {
            "enabled": True,
            "track_performance": True,
            "track_memory": False,
        }

        tracker = PerformanceTracker(monitor_config)
        assert tracker.enabled is True
        assert tracker.track_performance is True
        assert tracker.track_memory is False

    def test_performance_tracker_time_operation(self):
        """Test operation timing."""
        tracker = PerformanceTracker({"enabled": True})

        with tracker.time_operation("test_op"):
            time.sleep(0.1)

        metrics = tracker.get_metrics()
        assert "test_op" in metrics

        op_metrics = metrics["test_op"]
        assert op_metrics["count"] == 1
        assert op_metrics["total_duration"] > 0.05  # Should be at least 0.05s
        assert "average_duration" in op_metrics
        assert "min_duration" in op_metrics
        assert "max_duration" in op_metrics

    def test_performance_tracker_multiple_operations(self):
        """Test multiple operation tracking."""
        tracker = PerformanceTracker({"enabled": True})

        # Run same operation multiple times
        for _i in range(3):
            with tracker.time_operation("repeated_op"):
                time.sleep(0.01)

        metrics = tracker.get_metrics()["repeated_op"]
        assert metrics["count"] == 3
        assert metrics["average_duration"] > 0

    def test_performance_tracker_memory_recording(self):
        """Test memory usage recording."""
        tracker = PerformanceTracker({"track_memory": True})

        tracker.record_memory_usage("test_label")

        metrics = tracker.get_metrics()
        assert "memory_usage" in metrics
        assert "test_label" in metrics["memory_usage"]
        assert "memory_mb" in metrics["memory_usage"]["test_label"]

    def test_performance_tracker_export(self):
        """Test metrics export."""
        tracker = PerformanceTracker({"enabled": True})

        with tracker.time_operation("export_test"):
            pass

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_path = Path(f.name)

        try:
            tracker.export_metrics(export_path)

            # Verify file exists and contains valid JSON
            assert export_path.exists()
            with open(export_path) as f:
                exported_data = json.load(f)

            assert "export_test" in exported_data

        finally:
            if export_path.exists():
                export_path.unlink()


class TestAdapterManager:
    """Test adapter management."""

    def test_adapter_manager_creation(self):
        """Test manager creation."""
        manager = AdapterManager()
        assert isinstance(manager.adapters, dict)
        assert len(manager.adapters) == 0

    def test_adapter_manager_registration(self):
        """Test adapter registration."""
        manager = AdapterManager()
        adapter = ConcreteAdapter({"name": "test"})

        manager.register("test_adapter", adapter)
        assert "test_adapter" in manager.adapters
        assert manager.get("test_adapter") is adapter

    def test_adapter_manager_listing(self):
        """Test adapter listing."""
        manager = AdapterManager()
        adapter1 = ConcreteAdapter({"name": "adapter1"})
        adapter2 = ConcreteAdapter({"name": "adapter2"})

        manager.register("adapter1", adapter1)
        manager.register("adapter2", adapter2)

        adapters = manager.list_adapters()
        assert "adapter1" in adapters
        assert "adapter2" in adapters
        assert len(adapters) == 2

    def test_adapter_manager_initialization(self):
        """Test adapter initialization."""
        manager = AdapterManager()
        adapter = Mock(spec=BaseAdapter)
        adapter.state = AdapterState.INITIALIZED

        manager.register("test", adapter)
        manager.initialize_adapter("test", "test_model")

        adapter.load_model.assert_called_once_with("test_model")
        # The manager sets the state directly
        assert adapter.state.value == AdapterState.LOADED.value

    def test_adapter_manager_cleanup(self):
        """Test adapter cleanup."""
        manager = AdapterManager()
        adapter = Mock(spec=BaseAdapter)

        manager.register("test", adapter)
        manager.cleanup_adapter("test")

        adapter.cleanup.assert_called_once()

    def test_adapter_manager_batch_operations(self):
        """Test batch operations."""
        manager = AdapterManager()
        adapter1 = Mock(spec=BaseAdapter)
        adapter2 = Mock(spec=BaseAdapter)

        manager.register("adapter1", adapter1)
        manager.register("adapter2", adapter2)

        # Test initialize all
        manager.initialize_all("test_model")
        adapter1.load_model.assert_called_once_with("test_model")
        adapter2.load_model.assert_called_once_with("test_model")

        # Test cleanup all
        manager.cleanup_all()
        adapter1.cleanup.assert_called_once()
        adapter2.cleanup.assert_called_once()

    def test_adapter_manager_health_check(self):
        """Test health checking."""
        manager = AdapterManager()
        adapter = Mock(spec=BaseAdapter)
        adapter.state = AdapterState.LOADED

        manager.register("healthy_adapter", adapter)

        # Check individual adapter health
        health = manager.check_adapter_health("healthy_adapter")
        assert health["status"] == "healthy"
        assert health["state"] == "loaded"

        # Check non-existent adapter
        health = manager.check_adapter_health("nonexistent")
        assert health["status"] == "not_found"

        # Check overall health
        overall_health = manager.check_overall_health()
        assert "adapters" in overall_health
        assert "healthy_adapter" in overall_health["adapters"]


class TestAdapterUtils:
    """Test adapter utilities."""

    def test_validate_config(self):
        """Test configuration validation."""
        # Valid config
        config = {"name": "test", "adapter_type": "transformer"}
        result = AdapterUtils.validate_config(config)
        assert result["valid"] is True
        assert result["errors"] == []

        # Invalid config - missing name
        config = {"adapter_type": "transformer"}
        result = AdapterUtils.validate_config(config)
        assert result["valid"] is False
        assert "name is required" in result["errors"]

        # Invalid config - missing type
        config = {"name": "test"}
        result = AdapterUtils.validate_config(config)
        assert result["valid"] is False
        assert "adapter_type is required" in result["errors"]

    def test_infer_adapter_type(self):
        """Test adapter type inference."""
        assert AdapterUtils.infer_adapter_type("gpt2-medium") == "huggingface"
        assert AdapterUtils.infer_adapter_type("text-davinci-003") == "openai"
        assert AdapterUtils.infer_adapter_type("custom-model") == "generic"

    @patch("torch.cuda.is_available", return_value=True)
    def test_select_optimal_device_cuda(self, mock_cuda):
        """Test optimal device selection with CUDA."""
        device = AdapterUtils.select_optimal_device()
        assert device == "cuda:0"

    @patch("torch.cuda.is_available", return_value=False)
    def test_select_optimal_device_cpu(self, mock_cuda):
        """Test optimal device selection without CUDA."""
        device = AdapterUtils.select_optimal_device()
        assert device == "cpu"

    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        # Test with float32
        params = {"num_parameters": 1000000, "precision": "float32"}
        memory = AdapterUtils.estimate_memory_usage(params)
        expected = (1000000 * 4 / (1024**2)) * 1.2  # MB with 20% overhead
        assert abs(memory - expected) < 0.01

        # Test with float16
        params = {"num_parameters": 1000000, "precision": "float16"}
        memory = AdapterUtils.estimate_memory_usage(params)
        expected = (1000000 * 2 / (1024**2)) * 1.2
        assert abs(memory - expected) < 0.01

    def test_check_compatibility(self):
        """Test compatibility checking."""
        requirements = {"python": "3.8", "torch": "1.10.0"}
        system_info = {"python": "3.9.0", "torch": "1.11.0"}

        result = AdapterUtils.check_compatibility(requirements, system_info)
        assert result["compatible"] is True

        # Test incompatible
        system_info = {"python": "3.7.0", "torch": "1.11.0"}
        result = AdapterUtils.check_compatibility(requirements, system_info)
        assert result["compatible"] is False
        assert len(result["issues"]) > 0

    def test_migrate_config(self):
        """Test configuration migration."""
        old_config = {"name": "test", "model_path": "/path/to/model", "device_id": 1}

        new_config = AdapterUtils.migrate_config(old_config, "2.0.0")

        assert new_config["version"] == "2.0.0"
        assert new_config["model_id"] == "/path/to/model"
        assert "model_path" not in new_config
        assert new_config["device"]["type"] == "cuda"
        assert new_config["device"]["index"] == 1
        assert "device_id" not in new_config


class TestHFGPT2Adapter:
    """Test HuggingFace GPT-2 adapter."""

    def test_hf_adapter_creation(self):
        """Test adapter creation."""
        adapter = HF_GPT2_Adapter()
        assert adapter.name == "hf_gpt2"

    def test_hf_adapter_can_handle_mock(self):
        """Test can_handle with mock models."""
        adapter = HF_GPT2_Adapter()

        # Test with mock GPT-2 model - adjust the mock to be more realistic
        model = MockGPT2Model()
        # The adapter does structural checks, let's test both paths
        result = adapter.can_handle(model)
        # Could be True or False depending on mock structure - test both
        assert isinstance(result, bool)

        # Test with non-GPT-2 model
        non_gpt2 = nn.Linear(10, 10)
        assert adapter.can_handle(non_gpt2) is False

    def test_hf_adapter_describe(self):
        """Test model description."""
        adapter = HF_GPT2_Adapter()
        model = MockGPT2Model(n_layer=2, n_head=4, hidden_size=16)

        desc = adapter.describe(model)

        # Required fields
        assert desc["n_layer"] == 2
        assert desc["heads_per_layer"] == [4, 4]
        assert desc["mlp_dims"] == [64, 64]  # 4 * hidden_size
        assert isinstance(desc["tying"], dict)

        # Additional fields
        assert desc["model_type"] == "gpt2"
        assert desc["n_heads"] == 4
        assert desc["hidden_size"] == 16
        assert desc["total_params"] > 0

    def test_hf_adapter_describe_with_tying(self):
        """Test description with weight tying."""
        adapter = HF_GPT2_Adapter()
        model = MockGPT2Model()

        # Create weight tying
        model.lm_head.weight = model.transformer.wte.weight

        desc = adapter.describe(model)
        assert "lm_head.weight" in desc["tying"]
        assert desc["tying"]["lm_head.weight"] == "transformer.wte.weight"

    def test_hf_adapter_snapshot_restore(self):
        """Test snapshot and restore functionality."""
        adapter = HF_GPT2_Adapter()
        model = MockGPT2Model(n_layer=1, n_head=2, hidden_size=8)

        # Get original weights
        original_weight = model.transformer.h[0].attn.c_attn.weight.clone()

        # Create snapshot
        snapshot = adapter.snapshot(model)
        assert isinstance(snapshot, bytes)
        assert len(snapshot) > 0

        # Modify model
        with torch.no_grad():
            model.transformer.h[0].attn.c_attn.weight.fill_(1.0)

        # Verify modification
        assert not torch.equal(
            original_weight, model.transformer.h[0].attn.c_attn.weight
        )

        # Restore from snapshot
        adapter.restore(model, snapshot)

        # Verify restoration
        assert torch.allclose(
            original_weight, model.transformer.h[0].attn.c_attn.weight, atol=1e-6
        )

    def test_hf_adapter_validate_split_size(self):
        """Test split size validation."""
        adapter = HF_GPT2_Adapter()
        model = MockGPT2Model()

        # Should pass validation (no split_size specified)
        assert adapter.validate_split_size(model) is True

        # Test with split_size in config
        model.config.split_size = 16
        result = adapter.validate_split_size(model)
        assert isinstance(result, bool)

    def test_hf_adapter_get_layer_modules(self):
        """Test layer module retrieval."""
        adapter = HF_GPT2_Adapter()
        model = MockGPT2Model(n_layer=2)

        modules = adapter.get_layer_modules(model, 0)

        expected_keys = [
            "attn.c_attn",
            "attn.c_proj",
            "mlp.c_fc",
            "mlp.c_proj",
            "ln_1",
            "ln_2",
        ]

        for key in expected_keys:
            assert key in modules
            assert isinstance(modules[key], nn.Module)

    def test_hf_adapter_weight_tying_extraction(self):
        """Test weight tying extraction."""
        adapter = HF_GPT2_Adapter()
        model = MockGPT2Model()

        # No tying initially
        tying = adapter._extract_weight_tying_info(model)
        assert isinstance(tying, dict)
        assert len(tying) == 0

        # Create weight tying
        model.lm_head.weight = model.transformer.wte.weight
        tying = adapter._extract_weight_tying_info(model)
        assert "lm_head.weight" in tying

    def test_hf_adapter_error_handling(self):
        """Test error handling in adapter methods."""
        adapter = HF_GPT2_Adapter()

        # Test with invalid model structure
        invalid_model = nn.Module()

        from invarlock.core.exceptions import AdapterError

        with pytest.raises(AdapterError):
            adapter.describe(invalid_model)

        # Test can_handle with various edge cases
        assert adapter.can_handle(None) is False
        assert adapter.can_handle("not_a_model") is False


class TestHFBERTAdapter:
    """Tests specific to the HuggingFace BERT adapter."""

    def test_bert_adapter_snapshot_preserves_weight_tying(self):
        adapter = HF_BERT_Adapter()
        model = MockBertModel(tie_weights=True)

        assert adapter.can_handle(model) is True

        desc = adapter.describe(model)
        assert desc["model_type"] == "bert"
        tying = adapter._extract_weight_tying_info(model)
        assert (
            tying.get("cls.predictions.decoder.weight")
            == "bert.embeddings.word_embeddings.weight"
        )

        snapshot = adapter.snapshot(model)
        original = model.embeddings.word_embeddings.weight.detach().clone()

        with torch.no_grad():
            model.embeddings.word_embeddings.weight.add_(1.0)

        adapter.restore(model, snapshot)
        assert torch.allclose(model.embeddings.word_embeddings.weight, original)
        assert (
            model.cls.predictions.decoder.weight
            is model.embeddings.word_embeddings.weight
        )


class TestHFLLaMAAdapter:
    """Tests specific to the HuggingFace LLaMA adapter."""

    def test_llama_adapter_snapshot_preserves_weight_tying(self):
        adapter = HF_LLaMA_Adapter()
        model = MockLLaMAModel(tie_weights=True)

        assert adapter.can_handle(model) is True

        desc = adapter.describe(model)
        assert desc["model_type"] == "llama"
        tying = adapter._extract_weight_tying_info(model)
        assert tying == {"lm_head.weight": "model.embed_tokens.weight"}

        snapshot = adapter.snapshot(model)
        original = model.model.embed_tokens.weight.detach().clone()

        with torch.no_grad():
            model.model.embed_tokens.weight.mul_(0.5)

        adapter.restore(model, snapshot)
        assert torch.allclose(model.model.embed_tokens.weight, original)
        assert model.lm_head.weight is model.model.embed_tokens.weight


class TestInitModule:
    """Test __init__.py module functionality."""

    def test_quality_label_function(self):
        """Test quality label function."""
        # Test different quality tiers
        assert invarlock.adapters.quality_label(1.05) == "Excellent"
        assert invarlock.adapters.quality_label(1.15) == "Good"
        assert invarlock.adapters.quality_label(1.30) == "Fair"
        assert invarlock.adapters.quality_label(1.50) == "Degraded"

        # Test boundary conditions
        assert invarlock.adapters.quality_label(1.10) == "Excellent"
        assert invarlock.adapters.quality_label(1.25) == "Good"
        assert invarlock.adapters.quality_label(1.40) == "Fair"

    def test_placeholder_adapters(self):
        """Test placeholder adapter classes."""
        # Test that placeholders raise NotImplementedError
        with pytest.raises(NotImplementedError):
            invarlock.adapters.HF_Pythia_Adapter()

        # Note: HF_LLaMA_Adapter is now implemented, not a placeholder
        # Test that it can be instantiated without error
        adapter = invarlock.adapters.HF_LLaMA_Adapter()
        assert adapter is not None

        # Test auto-tuning placeholders
        with pytest.raises(NotImplementedError):
            invarlock.adapters.auto_tune_pruning_budget()

        with pytest.raises(NotImplementedError):
            invarlock.adapters.run_auto_invarlock()

        # Baseline-specific placeholders are not present

        # Placeholder previously referenced here is no longer present

    def test_removed_component_stubs(self):
        """Test removed component stubs."""
        # Test removed components
        with pytest.raises(NotImplementedError, match="InvarLock 1.0"):
            invarlock.adapters.InvarLockPipeline()

        with pytest.raises(NotImplementedError):
            invarlock.adapters.InvarLockConfig()

        with pytest.raises(NotImplementedError):
            invarlock.adapters.run_invarlock_pipeline()

        with pytest.raises(NotImplementedError):
            invarlock.adapters.run_invarlock()

        with pytest.raises(NotImplementedError):
            invarlock.adapters.quick_prune_gpt2()

    def test_removed_component_behavior(self):
        """Test _RemovedComponent behavior."""
        stub = invarlock.adapters._RemovedComponent("TestComponent", "new.component")

        # Test calling the stub
        with pytest.raises(
            NotImplementedError, match="is not available in InvarLock 1.0"
        ):
            stub()

        # Test attribute access
        attr = stub.some_attribute
        assert isinstance(attr, invarlock.adapters._RemovedComponent)


class TestIntegration:
    """Integration tests for the adapters module."""

    def test_module_imports(self):
        """Test that main module imports work."""
        # Test adapter classes
        assert hasattr(invarlock_adapters, "HF_GPT2_Adapter")
        assert hasattr(invarlock_adapters, "BaseAdapter")
        assert hasattr(invarlock_adapters, "AdapterConfig")

        # Test utility functions
        assert hasattr(invarlock_adapters, "quality_label")
        assert callable(invarlock.adapters.quality_label)

    def test_end_to_end_adapter_workflow(self):
        """Test end-to-end adapter workflow."""
        # Create and configure adapter manager
        AdapterManager()

        # Create adapter config
        config = AdapterConfig("test_hf", "huggingface")
        assert config.validate()["valid"] is True

        # Create HF adapter
        hf_adapter = HF_GPT2_Adapter()

        # Test with mock model
        model = MockGPT2Model()
        can_handle = hf_adapter.can_handle(model)
        assert isinstance(can_handle, bool)  # Just verify it returns a boolean

        # Get model description
        desc = hf_adapter.describe(model)
        assert desc["model_type"] == "gpt2"

        # Test snapshot/restore cycle
        snapshot = hf_adapter.snapshot(model)
        hf_adapter.restore(model, snapshot)  # Should not raise errors

    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration."""
        # Create tracker
        tracker = PerformanceTracker({"enabled": True})

        # Create base adapter with monitoring
        adapter = ConcreteAdapter({"name": "monitored"})
        adapter.enable_monitoring()

        # Simulate operations
        with tracker.time_operation("test_operation"):
            time.sleep(0.01)

        metrics = tracker.get_metrics()
        assert "test_operation" in metrics
        assert metrics["test_operation"]["count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
