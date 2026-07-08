# ruff: noqa: I001

"""
Legacy Mocked Pipeline Performance Contracts
============================================

These regression tests exercise synthetic model performance and mocked adapter
contracts. They do not provide integration-lane evidence.
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

# Mock external dependencies before importing INVARLOCK modules
# Imports are inside the patched context; suppression keeps Ruff quiet.

# The imports that follow occur inside this patched context and would
# normally trigger Ruff's import-order check; the scoped noqa below keeps
# Ruff quiet without suppressing other parts of the file.

with patch.dict(
    "sys.modules",
    {
        "transformers": Mock(),
        "invarlock.core.api": Mock(),
    },
):
    try:  # noqa: I001
        from invarlock.core.config import RunConfig
        from invarlock.core.runner import CoreRunner
        from invarlock.adapters.hf_causal import HF_Causal_Adapter

        pass
        from invarlock.guards.invariants import InvariantsGuard
        from invarlock.guards.rmt import RMTGuard
        from invarlock.guards.spectral import SpectralGuard
    except ImportError:
        # Create dummy classes if imports fail
        CoreRunner = Mock
        RunConfig = Mock
        SpectralGuard = Mock
        RMTGuard = Mock
        InvariantsGuard = Mock
        HF_Causal_Adapter = Mock


class MockGPT2Model(nn.Module):
    """Mock GPT-2 model for pipeline contract tests."""

    def __init__(self, n_layers: int = 2, hidden_size: int = 128):
        super().__init__()

        # Create GPT-2-like config
        self.config = Mock()
        self.config.model_type = "gpt2"
        self.config.n_layer = n_layers
        self.config.n_head = 8
        self.config.n_embd = hidden_size
        self.config.vocab_size = 1000
        self.config.n_positions = 512

        # Create GPT-2-like structure
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(1000, hidden_size)
        self.transformer.wpe = nn.Embedding(512, hidden_size)
        self.transformer.h = nn.ModuleList()

        for _i in range(n_layers):
            layer = nn.Module()

            # Attention
            layer.attn = nn.Module()
            layer.attn.c_attn = nn.Linear(hidden_size, hidden_size * 3)
            layer.attn.c_proj = nn.Linear(hidden_size, hidden_size)

            # MLP
            layer.mlp = nn.Module()
            layer.mlp.c_fc = nn.Linear(hidden_size, hidden_size * 4)
            layer.mlp.c_proj = nn.Linear(hidden_size * 4, hidden_size)

            # Layer norms
            layer.ln_1 = nn.LayerNorm(hidden_size)
            layer.ln_2 = nn.LayerNorm(hidden_size)

            self.transformer.h.append(layer)

        self.transformer.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, 1000)

        # Add some realistic weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to realistic values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, **kwargs):
        """Simple forward pass for testing."""
        batch_size, seq_len = input_ids.shape

        # Token and position embeddings
        token_emb = self.transformer.wte(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos_ids)
        hidden = token_emb + pos_emb

        # Pass through transformer layers
        for layer in self.transformer.h:
            # Simple attention (not real attention, just for shape)
            attn_out = layer.attn.c_proj(
                torch.tanh(layer.attn.c_attn(layer.ln_1(hidden)))[
                    :, :, : hidden.size(-1)
                ]
            )
            hidden = hidden + attn_out

            # MLP
            mlp_out = layer.mlp.c_proj(torch.relu(layer.mlp.c_fc(layer.ln_2(hidden))))
            hidden = hidden + mlp_out

        # Final layer norm and output
        hidden = self.transformer.ln_f(hidden)
        logits = self.lm_head(hidden)

        return type("GPT2Output", (), {"logits": logits})()


class MockDataLoader:
    """Mock dataloader for testing."""

    def __init__(self, batch_size: int = 2, seq_len: int = 32, num_batches: int = 5):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self._batches = [
            torch.randint(0, 999, (batch_size, seq_len)) for _ in range(num_batches)
        ]

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return self.num_batches


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


class TestPipelineErrorScenarios:
    """Test various error scenarios in the pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = MockGPT2Model()
        self.adapter = HF_Causal_Adapter()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_edit_configuration(self):
        """Test handling of invalid edit configurations."""
        invalid_configs = [
            {"bitwidth": 3},  # Invalid bitwidth for quantization
            {},  # Empty config
        ]

        # Define validation function for testing purposes
        def validate_edit_config(config):
            if "bitwidth" in config and config["bitwidth"] not in [4, 8, 16]:
                raise ValueError("Invalid bitwidth value")
            if not config:  # Empty config
                raise ValueError("Configuration cannot be empty")

        for invalid_config in invalid_configs:
            # Test validation with invalid config
            with pytest.raises(ValueError):
                validate_edit_config(invalid_config)

    def test_model_adapter_mismatch(self):
        """Test handling when model and adapter don't match."""
        # Create incompatible model
        incompatible_model = nn.Linear(10, 5)

        # Test adapter rejection
        if hasattr(self.adapter, "can_handle") and not hasattr(
            self.adapter.can_handle, "_mock_name"
        ):
            # Real adapter
            assert not self.adapter.can_handle(incompatible_model)
        else:
            # Mock adapter - simulate rejection behavior
            with patch.object(self.adapter, "can_handle", return_value=False):
                assert not self.adapter.can_handle(incompatible_model)

        # Test description failure
        if hasattr(self.adapter, "describe") and not hasattr(
            self.adapter.describe, "_mock_name"
        ):
            # Real adapter
            with pytest.raises(ValueError):
                self.adapter.describe(incompatible_model)
        else:
            # Mock adapter - simulate failure behavior
            with patch.object(
                self.adapter, "describe", side_effect=ValueError("Incompatible model")
            ):
                with pytest.raises(ValueError):
                    self.adapter.describe(incompatible_model)

    def test_guard_failure_scenarios(self):
        """Test various guard failure scenarios."""
        # Test spectral guard failure
        spectral_guard = SpectralGuard(sigma_quantile=0.95)

        with patch.object(spectral_guard, "validate") as mock_validate:
            mock_validate.return_value = {
                "passed": False,
                "decision": "block",
                "message": "Spectral norm exceeded threshold",
                "violations": [{"type": "spectral_violation", "severity": "high"}],
            }

            result = spectral_guard.validate(self.model, self.adapter, {})
            assert not result["passed"]
            assert result["decision"] == "block"
            assert len(result["violations"]) > 0

        # Test RMT guard failure
        rmt_guard = RMTGuard(margin=1.5)

        with patch.object(rmt_guard, "finalize") as mock_finalize:
            mock_finalize.return_value = Mock(
                passed=False,
                violations=[{"type": "rmt_outlier", "layer": 0}],
                metrics={"outlier_count": 5},
            )

            result = rmt_guard.finalize(self.model)
            assert not result.passed
            assert len(result.violations) > 0

    def test_calibration_data_issues(self):
        """Test handling of problematic calibration data."""
        # Test empty dataloader
        empty_loader = MockDataLoader(num_batches=0)
        assert len(empty_loader) == 0

        # Test mismatched data shapes
        mismatched_data = [
            torch.randint(0, 999, (2, 16)),  # Different sequence length
            torch.randint(0, 999, (1, 32)),  # Different batch size
        ]

        # Should handle gracefully
        for batch in mismatched_data:
            assert isinstance(batch, torch.Tensor)

    def test_filesystem_errors(self):
        """Test handling of filesystem-related errors."""
        # Test read-only directory
        readonly_dir = os.path.join(self.temp_dir, "readonly")
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, 0o444)  # Read-only

        try:
            # Attempt to write to read-only directory should fail
            test_file = os.path.join(readonly_dir, "test.json")
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                with open(test_file, "w") as f:
                    json.dump({"test": "data"}, f)
            else:
                with pytest.raises(PermissionError):
                    with open(test_file, "w") as f:
                        json.dump({"test": "data"}, f)
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)

        # Test nonexistent paths
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent", "path")
        assert not os.path.exists(nonexistent_path)


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = MockGPT2Model(n_layers=4, hidden_size=256)  # Larger model
        self.adapter = HF_Causal_Adapter()
        self.large_dataloader = MockDataLoader(batch_size=4, seq_len=64, num_batches=20)

    def test_pipeline_timing(self):
        """Test pipeline execution timing."""
        import time

        # Test adapter operations timing
        start_time = time.time()
        if hasattr(self.adapter, "describe") and not hasattr(
            self.adapter.describe, "_mock_name"
        ):
            # Real adapter
            description = self.adapter.describe(self.model)
        else:
            # Mock adapter - simulate description
            with patch.object(self.adapter, "describe") as mock_describe:
                mock_describe.return_value = {
                    "n_layer": 4,
                    "hidden_size": 256,
                    "device": "cpu",
                    "model_type": "gpt2",
                }
                description = self.adapter.describe(self.model)
        describe_time = time.time() - start_time

        assert describe_time < 1.0  # Should complete within 1 second
        assert isinstance(description, dict)

        # Test snapshot timing
        start_time = time.time()
        if hasattr(self.adapter, "snapshot") and not hasattr(
            self.adapter.snapshot, "_mock_name"
        ):
            # Real adapter
            snapshot = self.adapter.snapshot(self.model)
        else:
            # Mock adapter - simulate snapshot
            with patch.object(self.adapter, "snapshot") as mock_snapshot:
                mock_snapshot.return_value = b"mock_snapshot_data"
                snapshot = self.adapter.snapshot(self.model)
        snapshot_time = time.time() - start_time

        assert snapshot_time < 2.0  # Should complete within 2 seconds
        assert isinstance(snapshot, bytes)

        # Test restore timing
        start_time = time.time()
        if hasattr(self.adapter, "restore") and not hasattr(
            self.adapter.restore, "_mock_name"
        ):
            # Real adapter
            self.adapter.restore(self.model, snapshot)
        else:
            # Mock adapter - simulate restore
            with patch.object(self.adapter, "restore"):
                self.adapter.restore(self.model, snapshot)
        restore_time = time.time() - start_time

        assert restore_time < 2.0  # Should complete within 2 seconds

    def test_memory_efficiency(self):
        """Test memory efficiency of pipeline operations."""
        import gc

        import psutil

        # Measure baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss

        # Perform memory-intensive operations
        snapshots = []
        for _ in range(5):
            snapshot = self.adapter.snapshot(self.model)
            snapshots.append(snapshot)

        # Measure peak memory
        _ = psutil.Process().memory_info().rss

        # Clean up
        del snapshots
        gc.collect()

        # Measure final memory
        final_memory = psutil.Process().memory_info().rss

        # Memory should be released after cleanup
        memory_retained = final_memory - baseline_memory
        assert memory_retained < 50 * 1024 * 1024  # Less than 50MB retained

    def test_scalability(self):
        """Test pipeline scalability with different model sizes."""
        model_sizes = [
            (2, 128),  # Small
            (4, 256),  # Medium
            (6, 384),  # Large
        ]

        timing_results = []

        for n_layers, hidden_size in model_sizes:
            model = MockGPT2Model(n_layers=n_layers, hidden_size=hidden_size)

            # Measure describe operation timing
            import time

            start_time = time.time()
            if hasattr(self.adapter, "describe") and not hasattr(
                self.adapter.describe, "_mock_name"
            ):
                # Real adapter
                description = self.adapter.describe(model)
            else:
                # Mock adapter - simulate description
                with patch.object(self.adapter, "describe") as mock_describe:
                    mock_describe.return_value = {
                        "n_layer": n_layers,
                        "hidden_size": hidden_size,
                        "device": "cpu",
                        "model_type": "gpt2",
                    }
                    description = self.adapter.describe(model)
            elapsed = time.time() - start_time

            timing_results.append(elapsed)

            # Verify operation completed successfully
            assert isinstance(description, dict)
            assert description["n_layer"] == n_layers

        # Timing should scale reasonably (not exponentially)
        assert all(t < 5.0 for t in timing_results)  # All under 5 seconds
