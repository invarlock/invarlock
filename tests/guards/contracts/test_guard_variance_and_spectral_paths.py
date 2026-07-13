"""
Variance and spectral guard edge-case tests.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from invarlock.guards.policies import (
    get_variance_policy,
)
from invarlock.guards.spectral import SpectralGuard
from invarlock.guards.spectral_control import (
    apply_relative_spectral_cap,
    apply_spectral_control,
)
from invarlock.guards.spectral_measurement import (
    capture_baseline_sigmas,
    compute_sigma_max,
    scan_model_gains,
)
from invarlock.guards.variance import VarianceGuard
from invarlock.guards.variance_scaling import equalise_residual_variance


class TestVarianceGuardEdgeCases:
    """Exercise variance guard edge cases and lifecycle behavior."""

    def setup_method(self):
        """Set up test fixtures."""

        policy = get_variance_policy("balanced")
        self.guard = VarianceGuard(policy)
        self.model = self._create_transformer_model()

    def _create_transformer_model(self):
        """Create a transformer model."""
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.h = nn.ModuleList()

        for _i in range(2):
            layer = nn.Module()
            layer.attn = nn.Module()
            layer.attn.c_proj = nn.Linear(64, 64)
            layer.mlp = nn.Module()
            layer.mlp.c_proj = nn.Linear(256, 64)
            model.transformer.h.append(layer)

        return model

    def test_variance_guard_comprehensive_flow(self):
        """Test complete variance guard flow with edge cases."""
        # Test prepare with different policy parameters
        policy_updates = {
            "scope": "ffn",
            "min_gain": 0.15,
            "max_calib": 100,
            "deadband": 0.08,
            "clamp": (0.7, 1.4),
            "seed": 789,
        }

        result = self.guard.prepare(
            self.model, adapter=Mock(), calib=Mock(), policy=policy_updates
        )

        # Check that policy was updated
        assert self.guard._policy["scope"] == "ffn"
        assert self.guard._policy["min_gain"] == 0.15

        # Test enable/disable cycle with different states
        if result.get("ready", False):
            # Simulate some scales
            self.guard._scales = {"test_module": 0.85}
            self.guard._target_modules = {"test_module": nn.Linear(10, 5)}

            # Test multiple enable attempts (idempotent)
            result1 = self.guard.enable(self.model)
            result2 = self.guard.enable(self.model)  # Should be idempotent

            # Test multiple disable attempts (idempotent)
            result3 = self.guard.disable(self.model)
            result4 = self.guard.disable(self.model)  # Should be idempotent

            assert isinstance(result1, bool)
            assert isinstance(result2, bool)
            assert isinstance(result3, bool)
            assert isinstance(result4, bool)

    def test_compute_variance_scales_edge_cases(self):
        """Test _compute_variance_scales with edge cases."""
        # Test with empty dataloader
        from torch.utils.data import DataLoader, TensorDataset

        empty_dataset = TensorDataset(torch.empty(0, 0))
        empty_dataloader = DataLoader(empty_dataset)

        scales = self.guard._compute_variance_scales(self.model, empty_dataloader)
        assert isinstance(scales, dict)
        assert scales == {}

    def test_finalize_comprehensive_scenarios(self):
        """Test finalize with comprehensive scenarios."""
        # Prepare guard first
        self.guard._prepared = True
        self.guard._target_modules = {"test": nn.Linear(10, 5)}

        # Test various A/B testing scenarios
        scenarios = [
            # Good improvement - should enable
            {
                "ppl_no_ve": 3.5,
                "ppl_with_ve": 3.0,
                "ratio_ci": (0.80, 0.90),
                "expected_enable": True,
            },
            # Insufficient improvement - should not enable
            {
                "ppl_no_ve": 3.5,
                "ppl_with_ve": 3.48,
                "ratio_ci": (0.98, 1.01),
                "expected_enable": False,
            },
            # Negative improvement - should not enable
            {
                "ppl_no_ve": 3.0,
                "ppl_with_ve": 3.2,
                "ratio_ci": (1.02, 1.12),
                "expected_enable": False,
            },
        ]

        for scenario in scenarios:
            # Reset state
            self.guard._enabled = False
            self.guard.set_ab_results(
                scenario["ppl_no_ve"],
                scenario["ppl_with_ve"],
                windows_used=50,
                seed_used=123,
                ratio_ci=scenario["ratio_ci"],
            )

            # Test A/B gate evaluation
            should_enable, reason = self.guard._evaluate_ab_gate()

            assert should_enable is scenario["expected_enable"], reason
            if should_enable:
                assert reason.startswith("criteria_met")
            else:
                assert not reason.startswith("criteria_met")

            # Test finalize with this state
            result = self.guard.finalize(self.model)
            assert isinstance(result, dict)
            assert "passed" in result

    def test_checkpoint_edge_cases(self):
        """Test checkpoint operations with edge cases."""
        # Test with no target modules
        self.guard._target_modules = {}
        self.guard._push_checkpoint(self.model)
        assert len(self.guard._checkpoint_stack) == 0  # Should not create checkpoint

        # Test with target modules
        test_module = nn.Linear(10, 5)
        self.guard._target_modules = {"test": test_module}

        # Push multiple checkpoints
        self.guard._push_checkpoint(self.model)
        self.guard._push_checkpoint(self.model)
        assert len(self.guard._checkpoint_stack) == 2

        # Pop one checkpoint
        result = self.guard._pop_checkpoint(self.model)
        assert result
        assert len(self.guard._checkpoint_stack) == 1

        # Commit checkpoint
        self.guard._commit_checkpoint()
        assert len(self.guard._checkpoint_stack) == 0

    def test_equalise_residual_variance_edge_cases(self):
        """Test equalise_residual_variance with edge cases."""

        # Create simple model with forward method
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList([nn.Module()])
                self.transformer.h[0].attn = nn.Module()
                self.transformer.h[0].attn.c_proj = nn.Linear(32, 32)
                self.transformer.h[0].mlp = nn.Module()
                self.transformer.h[0].mlp.c_proj = nn.Linear(64, 32)

            def forward(self, x):
                return x  # Dummy forward

        model = SimpleModel()

        # Create minimal dataloader
        data = torch.randn(3, 16)
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=1)

        # Test with various parameters
        equalise_residual_variance(
            model=model,
            dataloader=dataloader,
            windows=1,
            tol=0.01,
            scale_bias=False,
            clamp_range=(0.8, 1.2),
            allow_empty=True,
        )
        assert len(model.transformer.h) == 1


class TestSpectralGuardExceptionPaths:
    """Exercise spectral guard exception and fallback paths."""

    def test_spectral_guard_validate_exception_handling(self):
        """Unexpected validation failures should raise."""
        guard = SpectralGuard()
        guard.prepared = True
        guard._capture_sigmas = lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("Forced validation error")
        )

        with pytest.raises(RuntimeError, match="Forced validation error"):
            guard.validate(nn.Linear(10, 5), Mock(), {})

    def test_compute_sigma_max_exception_handling(self):
        """Test compute_sigma_max exception handling (lines 87-88)."""
        # Test with non-tensor input to trigger exception
        sigma = compute_sigma_max("not_a_tensor")
        assert sigma == 1.0  # Fallback value

        # Test with problematic tensor using patching
        with patch(
            "invarlock.guards.spectral_measurement.power_iter_sigma_max",
            side_effect=RuntimeError("power_iter failed"),
        ):
            real_tensor = torch.randn(5, 3)
            sigma = compute_sigma_max(real_tensor)
            assert sigma == 1.0

    def test_auto_sigma_target_exception_handling(self):
        """Test auto_sigma_target exception handling (lines 102-106)."""
        from invarlock.guards.spectral_measurement import auto_sigma_target

        # Test with a valid model - the function now computes real percentiles
        target = auto_sigma_target(nn.Linear(10, 5), percentile=0.9)
        assert isinstance(target, float)
        assert target > 0  # Should return a positive value

        # Test the narrowed percentile fallback path on supported measurement errors
        with patch("numpy.percentile", side_effect=RuntimeError("Percentile failed")):
            target = auto_sigma_target(nn.Linear(10, 5), percentile=0.9)
            assert target == 0.9  # Should fall back to percentile on exception

        # Test with empty model (no weight matrices)
        empty_model = nn.Module()
        target = auto_sigma_target(empty_model, percentile=0.9)
        assert (
            target == 0.9
        )  # Should fall back to percentile when no spectral norms found

    def test_apply_weight_rescale_behavior(self):
        """Test apply_weight_rescale behavior (lines 121-131)."""
        from invarlock.guards.spectral_control import apply_weight_rescale

        # Test the actual implementation - it really rescales weights
        model = nn.Linear(10, 5)
        original_weight = model.weight.clone()

        result = apply_weight_rescale(model, scale_factor=0.8)

        assert isinstance(result, dict)
        assert result["applied"]  # Should actually apply rescaling
        assert "message" in result
        assert result["scale_factor"] == 0.8
        assert "rescaled_modules" in result
        assert len(result["rescaled_modules"]) > 0

        # Verify the weight was actually rescaled
        assert not torch.allclose(model.weight, original_weight)
        assert torch.allclose(model.weight, original_weight * 0.8)

    def test_apply_relative_spectral_cap_behavior(self):
        """Test apply_relative_spectral_cap behavior (lines 146-156)."""

        # Test the actual implementation
        model = nn.Linear(10, 5)
        baselines = capture_baseline_sigmas(model)
        result = apply_relative_spectral_cap(
            model, cap_ratio=2.0, baseline_sigmas=baselines
        )

        assert isinstance(result, dict)
        assert "applied" in result
        assert "message" in result
        assert result["cap_ratio"] == 2.0
        assert "capped_modules" in result
        assert "failed_modules" in result

    def test_apply_spectral_control_behavior(self):
        """Test apply_spectral_control behavior (lines 171-181)."""

        # Test the actual implementation
        model = nn.Linear(10, 5)
        baselines = capture_baseline_sigmas(model)
        result = apply_spectral_control(
            model,
            {"scope": "all", "cap_ratio": 2.0, "baseline_sigmas": baselines},
        )

        assert isinstance(result, dict)
        assert "applied" in result
        assert "message" in result
        assert "policy" in result
        assert "capping_applied" in result
        assert "rescaling_applied" in result

    def test_capture_baseline_sigmas_behavior(self):
        """Test capture_baseline_sigmas behavior (lines 194-202)."""
        from invarlock.guards.spectral_measurement import capture_baseline_sigmas

        # Test the actual function behavior - it returns real sigma values
        model = nn.Linear(10, 5)
        result = capture_baseline_sigmas(model)

        assert isinstance(result, dict)
        # Should return dict with module name and actual sigma value
        assert len(result) == 1  # One module with weights
        assert "" in result  # Empty string is the module name for direct Linear module
        assert isinstance(result[""], float)  # Should be real computed sigma
        assert result[""] > 0  # Should be positive

    def test_scan_model_gains_behavior(self):
        """Test scan_model_gains behavior (lines 215-226)."""

        # Test the actual implementation
        model = nn.Linear(10, 5)
        result = scan_model_gains(model)

        assert isinstance(result, dict)
        assert "message" in result
        assert "total_layers" in result
        assert result["total_layers"] >= 1  # Should count at least the Linear layer
        assert "scanned_modules" in result
        assert "spectral_norms" in result
        assert isinstance(result["spectral_norms"], list)
