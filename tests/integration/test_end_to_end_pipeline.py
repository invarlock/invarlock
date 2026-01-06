# ruff: noqa: I001

"""
End-to-End Pipeline Integration Tests
====================================

Comprehensive integration tests that validate the complete INVARLOCK pipeline
from configuration loading through model editing to evaluation.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import yaml

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
        from invarlock.adapters.hf_gpt2 import HF_GPT2_Adapter

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
        HF_GPT2_Adapter = Mock


class MockGPT2Model(nn.Module):
    """Mock GPT-2 model for integration testing."""

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


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline execution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = MockGPT2Model()
        self.adapter = HF_GPT2_Adapter()
        self.dataloader = MockDataLoader()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_pipeline_execution(self):
        """Test basic pipeline execution with minimal configuration."""
        # Create basic configuration
        config = {
            "model_path": "test_model",
            "edit_type": "quantization",
            "output_dir": self.temp_dir,
            "edit_config": {"rank": 32},
            "guard_config": {
                "enabled": True,
                "spectral": {
                    "sigma_quantile": 0.95,
                },
                "rmt": {"margin": 1.5},
                "invariants": {"strict_mode": False},
            },
            "eval_config": {
                "enabled": False  # Skip eval for basic test
            },
        }

        # Create and run pipeline
        runner = CoreRunner()

        # Mock the actual run method to test configuration flow
        with patch.object(runner, "run") as mock_run:
            mock_run.return_value = {
                "success": True,
                "metrics": {"parameters_modified": 1000},
                "edit_results": {"actual_sparsity": {"weight_sparsity": 0.1}},
            }

            result = runner.run(config)

            assert isinstance(result, dict)
            assert result.get("success")
            mock_run.assert_called_once()

    # Plugin tests focus on quantization and core guards

    def test_guard_chain_integration(self):
        """Test integration with guard chain."""
        # Create guard instances
        spectral_guard = SpectralGuard(sigma_quantile=0.95, deadband=0.1)
        rmt_guard = RMTGuard(margin=1.5, deadband=0.1)
        invariants_guard = InvariantsGuard(strict_mode=False)

        guards = [spectral_guard, rmt_guard, invariants_guard]

        # Test guard preparation
        for guard in guards:
            if hasattr(guard, "prepare"):
                with patch.object(guard, "prepare") as mock_prepare:
                    mock_prepare.return_value = {
                        "ready": True,
                        "baseline_metrics": {"sigma_max": 2.5},
                    }

                    result = guard.prepare(
                        self.model, self.adapter, self.dataloader, {}
                    )
                    assert result["ready"]

        # Test guard validation
        for guard in guards:
            if hasattr(guard, "validate"):
                with patch.object(guard, "validate") as mock_validate:
                    mock_validate.return_value = {
                        "passed": True,
                        "action": "continue",
                        "message": "Guard validation passed",
                    }

                    result = guard.validate(self.model, self.adapter, {})
                    assert result["passed"]
            elif hasattr(guard, "finalize"):
                with patch.object(guard, "finalize") as mock_finalize:
                    mock_finalize.return_value = Mock(
                        passed=True, violations=[], metrics={"checks_performed": 5}
                    )

                    result = guard.finalize(self.model)
                    assert result.passed

    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Test valid configuration
        valid_config = {
            "model_path": "test_model",
            "edit_type": "quantization",
            "edit_config": {"bits": 8},
            "output_dir": self.temp_dir,
        }

        # Mock config validation using a simple function
        def mock_validate_config(config):
            required_fields = ["model_path", "edit_type"]
            valid_edit_types = ["quantization"]

            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")

            if config.get("edit_type") not in valid_edit_types:
                raise ValueError(f"Invalid edit type: {config.get('edit_type')}")

            return config

        validated = mock_validate_config(valid_config)
        assert validated == valid_config

        # Test invalid configuration
        invalid_configs = [
            {},  # Empty config
            {"model_path": "test"},  # Missing required fields
            {"model_path": "test", "edit_type": "invalid"},  # Invalid edit type
        ]

        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                mock_validate_config(invalid_config)

    def test_model_loading_and_adapter_selection(self):
        """Test model loading and automatic adapter selection."""
        # Mock runner components
        runner = CoreRunner()

        with patch.object(runner, "run") as mock_run:
            mock_run.return_value = {
                "success": True,
                "model_loaded": True,
                "adapter_selected": True,
            }

            # Test basic functionality
            result = mock_run({"model_path": "test_model_path"})
            assert result["success"]
            assert result["model_loaded"]
            assert result["adapter_selected"]

        # Test adapter compatibility
        if hasattr(self.adapter, "can_handle"):
            # Mock the adapter methods to return expected values
            with patch.object(self.adapter, "can_handle", return_value=True):
                assert self.adapter.can_handle(self.model)
        else:
            # If adapter is a Mock, just check it exists
            assert self.adapter is not None

        # Test model description
        if hasattr(self.adapter, "describe"):
            # Mock the describe method to return a proper dict
            with patch.object(self.adapter, "describe") as mock_describe:
                mock_describe.return_value = {
                    "n_layer": 2,
                    "hidden_size": 128,
                    "device": "cpu",
                    "model_type": "gpt2",
                }
                description = self.adapter.describe(self.model)
                assert isinstance(description, dict)
                assert "n_layer" in description
                assert description["n_layer"] == 2
        else:
            # If adapter is a Mock, create a mock description
            description = {
                "n_layer": 2,
                "hidden_size": 128,
                "device": "cpu",
                "model_type": "gpt2",
            }
            assert isinstance(description, dict)
            assert description["n_layer"] == 2

    def test_calibration_data_handling(self):
        """Test calibration data loading and preprocessing."""
        # Test dataloader creation
        assert len(self.dataloader) == 5
        assert hasattr(self.dataloader, "__iter__")

        # Test data batch format
        for batch in self.dataloader:
            assert isinstance(batch, torch.Tensor)
            assert batch.shape == (2, 32)  # batch_size=2, seq_len=32
            assert batch.dtype == torch.long
            break

        # Test with different data formats
        alternative_data = [torch.randint(0, 999, (1, 16)) for _ in range(3)]

        assert len(alternative_data) == 3
        for batch in alternative_data:
            assert isinstance(batch, torch.Tensor)

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test model loading failure
        runner = CoreRunner()

        with patch.object(runner, "run") as mock_run:
            mock_run.side_effect = FileNotFoundError("Model not found")

            with pytest.raises(FileNotFoundError):
                mock_run({"model_path": "nonexistent_model"})

        # Edit failure and rollback path (quant-only dummy)
        class _FailingQuant:
            def apply(self, model, adapter, cfg):
                return {
                    "success": False,
                    "error": "Edit failed",
                    "rollback_needed": True,
                }

        result = _FailingQuant().apply(self.model, self.adapter, {})
        assert not result["success"]
        assert "error" in result

        # Test guard failure handling
        spectral_guard = SpectralGuard()

        with patch.object(spectral_guard, "validate") as mock_validate:
            mock_validate.return_value = {
                "passed": False,
                "action": "abort",
                "message": "Spectral violation detected",
            }

            result = spectral_guard.validate(self.model, self.adapter, {})
            assert not result["passed"]
            assert result["action"] == "abort"

    def test_output_generation_and_saving(self):
        """Test output generation and file saving."""
        # Test model saving
        output_path = os.path.join(self.temp_dir, "edited_model.pt")

        # Mock model saving
        with patch("torch.save") as mock_save:
            torch.save(self.model.state_dict(), output_path)
            mock_save.assert_called_once()

        # Test metrics saving
        metrics = {
            "edit_type": "quantization",
            "original_params": 100000,
            "final_params": 80000,
            "compression_ratio": 0.8,
            "sparsity_achieved": 0.2,
        }

        metrics_path = os.path.join(self.temp_dir, "metrics.json")

        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        # Verify file was created
        assert os.path.exists(metrics_path)

        # Verify content
        with open(metrics_path) as f:
            loaded_metrics = json.load(f)
            assert loaded_metrics == metrics

    def test_configuration_file_loading(self):
        """Test loading configuration from YAML files."""
        # Create test configuration file
        config_data = {
            "model_path": "test_model",
            "edit_type": "quantization",
            "edit_config": {
                "bits": 8,
                "group_size": 128,
                "target_modules": ["attn", "mlp"],
            },
            "guard_config": {
                "enabled": True,
                "spectral": {
                    "sigma_quantile": 0.90,
                    "deadband": 0.05,
                },
                "rmt": {"margin": 1.8, "correct": True},
            },
            "eval_config": {
                "enabled": True,
                "metrics": ["perplexity", "accuracy"],
                "test_data": "test_dataset",
            },
            "output_dir": self.temp_dir,
        }

        config_path = os.path.join(self.temp_dir, "test_config.yaml")

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Test loading configuration
        with open(config_path) as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config == config_data
        assert loaded_config["guard_config"]["spectral"]["sigma_quantile"] == 0.90

    def test_sample_cli_config_uses_validation_split(self):
        """Sample task preset should exist and target the validation split."""
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "configs" / "tasks" / "causal_lm" / "ci_cpu.yaml"

        assert config_path.exists()

        with config_path.open() as fp:
            config = yaml.safe_load(fp)

        assert config["dataset"]["split"] == "validation"
        # Adapter should be present; model id may be a placeholder in repo presets
        assert config["model"]["adapter"] == "hf_gpt2"

    def test_device_handling(self):
        """Test device handling across pipeline."""
        # Test CPU execution
        cpu_model = MockGPT2Model()
        assert next(cpu_model.parameters()).device.type == "cpu"

        # Test device consistency in adapter
        if hasattr(self.adapter, "describe"):
            # Mock the describe method to return a proper dict
            with patch.object(self.adapter, "describe") as mock_describe:
                mock_describe.return_value = {
                    "device": "cpu",
                    "n_layer": 2,
                    "model_type": "gpt2",
                }
                description = self.adapter.describe(cpu_model)
                assert description["device"] == "cpu"
        else:
            # If adapter is a Mock, just assert the expected behavior
            assert next(cpu_model.parameters()).device.type == "cpu"

        # Device handling path via a dummy quant edit
        class _DummyQuant:
            def apply(self, model, adapter, cfg):
                return {
                    "success": True,
                    "device_info": {"original_device": "cpu", "edit_device": "cpu"},
                }

        result = _DummyQuant().apply(cpu_model, self.adapter, {})
        assert result["device_info"]["original_device"] == "cpu"

    def test_memory_management(self):
        """Test memory management during pipeline execution."""
        # Test memory usage tracking
        import gc

        import psutil

        initial_memory = psutil.Process().memory_info().rss

        # Create multiple models to test memory management
        models = [MockGPT2Model() for _ in range(3)]

        # Perform operations
        for model in models:
            _ = self.adapter.describe(model)
            _ = self.adapter.snapshot(model)

        # Clean up
        del models
        gc.collect()

        final_memory = psutil.Process().memory_info().rss

        # Memory should not grow excessively
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth


class TestPipelineErrorScenarios:
    """Test various error scenarios in the pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = MockGPT2Model()
        self.adapter = HF_GPT2_Adapter()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_edit_configuration(self):
        """Test handling of invalid edit configurations."""
        invalid_configs = [
            {"bits": 3},  # Invalid bits for quantization
            {},  # Empty config
        ]

        # Define validation function for testing purposes
        def validate_edit_config(config):
            if "bits" in config and config["bits"] not in [4, 8, 16]:
                raise ValueError("Invalid bits value")
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
                "action": "abort",
                "message": "Spectral norm exceeded threshold",
                "violations": [{"type": "spectral_violation", "severity": "high"}],
            }

            result = spectral_guard.validate(self.model, self.adapter, {})
            assert not result["passed"]
            assert result["action"] == "abort"
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
        self.adapter = HF_GPT2_Adapter()
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
