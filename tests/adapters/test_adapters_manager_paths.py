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

import torch.nn as nn

# Import invarlock adapters namespace
from invarlock.adapters.base import (
    AdapterManager,
    AdapterUtils,
    BaseAdapter,
    PerformanceTracker,
)
from invarlock.adapters.base_types import (
    AdapterState,
)


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


class MockRopeDecoderLayer(nn.Module):
    """Minimal RoPE decoder-only block for adapter testing."""

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


class MockRopeDecoderModel(nn.Module):
    """Mock RoPE decoder-only model with tying support."""

    def __init__(
        self,
        n_layer: int = 2,
        hidden_size: int = 32,
        vocab_size: int = 64,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.config = Mock()
        self.config.model_type = "mistral"
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
            [MockRopeDecoderLayer(hidden_size) for _ in range(n_layer)]
        )
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.model.embed_tokens.weight


class MockMixtralExpert(nn.Module):
    """Minimal Mixtral MoE expert module for adapter testing."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, intermediate_size)


class MockMixtralLayer(nn.Module):
    """Minimal Mixtral block for adapter testing (MoE instead of MLP)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.k_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.v_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.o_proj = nn.Linear(hidden_size, hidden_size)

        self.block_sparse_moe = nn.Module()
        self.block_sparse_moe.experts = nn.ModuleList(
            [MockMixtralExpert(hidden_size, intermediate_size)]
        )

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)


class MockMixtralModel(nn.Module):
    """Mock Mixtral model with tying support."""

    def __init__(
        self,
        n_layer: int = 2,
        hidden_size: int = 32,
        intermediate_size: int = 128,
        vocab_size: int = 64,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.config = Mock()
        self.config.model_type = "mixtral"
        self.config.num_hidden_layers = n_layer
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = 4
        self.config.num_key_value_heads = 2
        self.config.intermediate_size = intermediate_size
        self.config.vocab_size = vocab_size
        self.config.max_position_embeddings = 32768
        self.config.rms_norm_eps = 1e-6

        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockMixtralLayer(hidden_size, intermediate_size) for _ in range(n_layer)]
        )
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.model.embed_tokens.weight


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
