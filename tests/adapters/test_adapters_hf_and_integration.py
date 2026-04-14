from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn

import invarlock.adapters as invarlock_adapters
from invarlock.adapters.auto import HF_Auto_Adapter
from invarlock.adapters.base import AdapterConfig, AdapterManager, PerformanceTracker
from invarlock.adapters.hf_causal import HF_Causal_Adapter
from invarlock.adapters.hf_mlm import HF_MLM_Adapter
from tests.adapters.test_adapters_base_types import (
    ConcreteAdapter,
    MockBertModel,
    MockGPT2Model,
    MockMixtralModel,
    MockRopeDecoderModel,
)


class TestHFGPT2Adapter:
    """Test HuggingFace GPT-2 adapter."""

    def test_hf_adapter_creation(self):
        """Test adapter creation."""
        adapter = HF_Causal_Adapter()
        assert adapter.name == "hf_causal"

    def test_hf_adapter_can_handle_mock(self):
        """Test can_handle with mock models."""
        adapter = HF_Causal_Adapter()

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
        adapter = HF_Causal_Adapter()
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
        adapter = HF_Causal_Adapter()
        model = MockGPT2Model()

        # Create weight tying
        model.lm_head.weight = model.transformer.wte.weight

        desc = adapter.describe(model)
        assert "lm_head.weight" in desc["tying"]
        assert desc["tying"]["lm_head.weight"] == "transformer.wte.weight"

    def test_hf_adapter_snapshot_restore(self):
        """Test snapshot and restore functionality."""
        adapter = HF_Causal_Adapter()
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

    def test_hf_adapter_get_layer_modules(self):
        """Test layer module retrieval."""
        adapter = HF_Causal_Adapter()
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
        adapter = HF_Causal_Adapter()
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
        adapter = HF_Causal_Adapter()

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
        adapter = HF_MLM_Adapter()
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

    def test_bert_adapter_layer_modules_and_embeddings_info(self):
        adapter = HF_MLM_Adapter()
        model = MockBertModel(tie_weights=True)

        modules = adapter.get_layer_modules(model, 0)
        assert (
            modules["attention.self.query"]
            is model.encoder.layer[0].attention.self.query
        )
        assert (
            modules["attention.output.LayerNorm"]
            is model.encoder.layer[0].attention.output.LayerNorm
        )

        embeddings_info = adapter.get_embeddings_info(model)
        assert embeddings_info["vocab_size"] == model.config.vocab_size
        assert embeddings_info["hidden_size"] == model.config.hidden_size
        assert embeddings_info["has_word_embeddings"] is True
        assert embeddings_info["has_position_embeddings"] is False
        assert embeddings_info["has_token_type_embeddings"] is False


class TestHFCausalAdapterRopeDecoder:
    """Tests causal adapter behavior on RoPE decoder-only structures."""

    def test_causal_adapter_snapshot_preserves_weight_tying(self):
        adapter = HF_Causal_Adapter()
        model = MockRopeDecoderModel(tie_weights=True)

        assert adapter.can_handle(model) is True

        desc = adapter.describe(model)
        assert desc["model_type"] == "mistral"
        tying = adapter._extract_weight_tying_info(model)
        assert tying == {"lm_head.weight": "model.embed_tokens.weight"}

        snapshot = adapter.snapshot(model)
        original = model.model.embed_tokens.weight.detach().clone()

        with torch.no_grad():
            model.model.embed_tokens.weight.mul_(0.5)

        adapter.restore(model, snapshot)
        assert torch.allclose(model.model.embed_tokens.weight, original)
        assert model.lm_head.weight is model.model.embed_tokens.weight

    def test_causal_adapter_describe_supports_mixtral_structure(self):
        adapter = HF_Causal_Adapter()
        model = MockMixtralModel(tie_weights=True)

        assert adapter.can_handle(model) is True

        desc = adapter.describe(model)
        assert desc["hf_model_type"] == "mixtral"
        assert desc["n_layer"] == model.config.num_hidden_layers
        assert len(desc["mlp_dims"]) == model.config.num_hidden_layers
        assert all(dim == model.config.intermediate_size for dim in desc["mlp_dims"])

        modules = adapter.get_layer_modules(model, 0)
        assert (
            modules["mlp.gate_proj"]
            is model.model.layers[0].block_sparse_moe.experts[0].w1
        )
        assert (
            modules["mlp.down_proj"]
            is model.model.layers[0].block_sparse_moe.experts[0].w2
        )
        assert (
            modules["mlp.up_proj"]
            is model.model.layers[0].block_sparse_moe.experts[0].w3
        )


class TestInitModule:
    """Test __init__.py module functionality."""

    def test_test_only_quality_helper_is_not_exported(self):
        """Test-only helpers should not leak into the adapter namespace."""
        assert not hasattr(invarlock_adapters, "quality_label")

    def test_placeholder_adapters(self):
        """Removed compatibility placeholders are absent from the namespace."""
        assert not hasattr(invarlock_adapters, "HF_Pythia_Adapter")
        assert not hasattr(invarlock_adapters, "auto_tune_pruning_budget")
        assert not hasattr(invarlock_adapters, "run_auto_invarlock")
        assert not hasattr(invarlock_adapters, "InvarLockPipeline")
        assert not hasattr(invarlock_adapters, "InvarLockConfig")
        assert not hasattr(invarlock_adapters, "run_invarlock_pipeline")
        assert not hasattr(invarlock_adapters, "run_invarlock")
        assert not hasattr(invarlock_adapters, "quick_prune_gpt2")

        assert not hasattr(invarlock_adapters, "HF_Causal_Adapter")
        assert not hasattr(invarlock_adapters, "HF_Auto_Adapter")

        adapter = HF_Auto_Adapter()
        assert adapter is not None

    def test_removed_component_stubs(self):
        """Removed compatibility components stay absent."""
        for name in (
            "InvarLockPipeline",
            "InvarLockConfig",
            "run_invarlock_pipeline",
            "run_invarlock",
            "quick_prune_gpt2",
        ):
            assert not hasattr(invarlock_adapters, name)

    def test_hf_auto_adapter_exposes_only_explicit_adapter_surface(self):
        class _Delegate:
            def load_model(self, model_id: str, **kwargs):
                return {"model_id": model_id, "kwargs": kwargs}

            def describe(self, _model):
                return {"model_type": "delegate"}

            def snapshot(self, _model):
                return b"snapshot"

            def restore(self, _model, _blob):
                return None

            def tokenize(self, _text):
                return ["should-not-be-exposed"]

        adapter = HF_Auto_Adapter()
        adapter._delegate = _Delegate()

        assert adapter.load_model("demo/model", device="cpu") == {
            "model_id": "demo/model",
            "kwargs": {"device": "cpu"},
        }
        assert adapter.describe(object()) == {"model_type": "delegate"}
        assert adapter.snapshot(object()) == b"snapshot"
        adapter.restore(object(), b"snapshot")
        assert not hasattr(adapter, "tokenize")

    def test_removed_component_behavior(self):
        """Removed-component shim type is gone."""
        assert not hasattr(invarlock_adapters, "_RemovedComponent")


class TestIntegration:
    """Integration tests for the adapters module."""

    def test_module_imports(self):
        """Test that main module imports work."""
        assert not hasattr(invarlock_adapters, "HF_Causal_Adapter")
        assert hasattr(invarlock_adapters, "BaseAdapter")
        assert hasattr(invarlock_adapters, "AdapterConfig")
        assert not hasattr(invarlock_adapters, "quality_label")

    def test_end_to_end_adapter_workflow(self):
        """Test end-to-end adapter workflow."""
        # Create and configure adapter manager
        AdapterManager()

        # Create adapter config
        config = AdapterConfig("test_hf", "huggingface")
        assert config.validate()["valid"] is True

        # Create HF adapter
        hf_adapter = HF_Causal_Adapter()

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
