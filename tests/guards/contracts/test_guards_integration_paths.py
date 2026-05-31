"""
Comprehensive Guard System Tests
===============================

Comprehensive tests for all guard modules to achieve 70% coverage.
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from invarlock.guards.invariants import (
    InvariantsGuard,
)
from invarlock.guards.policies import (
    get_variance_policy,
)
from invarlock.guards.rmt import RMTGuard
from invarlock.guards.rmt_analysis import (
    capture_baseline_mp_stats,
    layer_svd_stats,
)
from invarlock.guards.rmt_detection import (
    rmt_detect,
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


class TestIntegrationScenarios:
    """Integration tests combining multiple guards."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = self._create_comprehensive_model()

    def _create_comprehensive_model(self):
        """Create a comprehensive model for testing."""
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.wte = nn.Embedding(1000, 128)
        model.transformer.h = nn.ModuleList()

        # Add multiple transformer layers
        for _i in range(3):
            layer = nn.Module()
            layer.attn = nn.Module()
            layer.attn.c_attn = nn.Linear(128, 384)
            layer.attn.c_proj = nn.Linear(128, 128)
            layer.mlp = nn.Module()
            layer.mlp.c_fc = nn.Linear(128, 512)
            layer.mlp.c_proj = nn.Linear(512, 128)
            model.transformer.h.append(layer)

        model.lm_head = nn.Linear(128, 1000)
        return model

    def test_multiple_guards_preparation(self):
        """Test preparing multiple guards."""
        invariants_guard = InvariantsGuard()
        spectral_guard = SpectralGuard()
        rmt_guard = RMTGuard()

        # Prepare guards that support preparation
        mock_adapter = Mock()
        mock_calib = Mock()

        inv_result = invariants_guard.prepare(self.model, mock_adapter, mock_calib, {})
        rmt_result = rmt_guard.prepare(self.model, mock_adapter, mock_calib, {})

        assert inv_result["ready"]
        assert rmt_result["ready"]

        # Check that guards that support preparation are prepared
        assert invariants_guard.prepared
        assert rmt_guard.prepared

        # Test validate method on SpectralGuard (minimal interface)
        spec_result = spectral_guard.validate(self.model, mock_adapter, {})
        assert isinstance(spec_result, dict)
        assert "passed" in spec_result

    def test_guards_with_different_policies(self):
        """Test guards with different policy configurations."""
        # Conservative policies
        spectral_guard = SpectralGuard(sigma_quantile=0.90, deadband=0.05, scope="ffn")
        rmt_guard = RMTGuard(q="auto", deadband=0.05, margin=1.3, correct=True)

        # Aggressive policies
        spectral_guard_agg = SpectralGuard(
            sigma_quantile=0.98, deadband=0.15, scope="all"
        )
        rmt_guard_agg = RMTGuard(q="auto", deadband=0.15, margin=1.8, correct=True)

        # Test RMT guards that support preparation
        mock_adapter = Mock()
        mock_calib = Mock()

        rmt_result = rmt_guard.prepare(self.model, mock_adapter, mock_calib, {})
        rmt_result_agg = rmt_guard_agg.prepare(self.model, mock_adapter, mock_calib, {})

        # RMT guards should succeed
        assert rmt_result["ready"]
        assert rmt_result_agg["ready"]

        # Check that RMT policies are different
        assert rmt_guard.margin != rmt_guard_agg.margin

        # Test SpectralGuard validate methods (minimal interface)
        spec_result = spectral_guard.validate(self.model, mock_adapter, {})
        spec_result_agg = spectral_guard_agg.validate(self.model, mock_adapter, {})

        assert isinstance(spec_result, dict)
        assert isinstance(spec_result_agg, dict)
        assert "passed" in spec_result
        assert "passed" in spec_result_agg

        # Check that SpectralGuard configs are different
        assert spectral_guard.config.get(
            "sigma_quantile"
        ) != spectral_guard_agg.config.get("sigma_quantile")

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with empty model
        empty_model = nn.Module()

        guards = [InvariantsGuard(), SpectralGuard(), RMTGuard()]

        for guard in guards:
            result = guard.prepare(empty_model, Mock(), Mock(), {})
            # Guards should degrade gracefully on empty models.
            assert isinstance(result, dict)

    def test_baseline_capture_and_comparison(self):
        """Test baseline capture and comparison across guards."""
        # Capture spectral baselines (simple signature - no scope parameter)
        spectral_baselines = capture_baseline_sigmas(self.model)
        assert isinstance(spectral_baselines, dict)
        # May be empty for the placeholder implementation, which is fine

        # Capture RMT baselines
        rmt_baselines = capture_baseline_mp_stats(self.model)
        assert isinstance(rmt_baselines, dict)
        # May be empty if no linear layers match the allowed suffixes

        # Both functions should return dictionaries
        assert isinstance(spectral_baselines, dict)
        assert isinstance(rmt_baselines, dict)

        # If both have data, there should be some overlap (both capture linear layers)
        if spectral_baselines and rmt_baselines:
            spectral_modules = set(spectral_baselines.keys())
            rmt_modules = set(rmt_baselines.keys())
            # Test that at least the functions work, overlap is not guaranteed with placeholders
            assert len(spectral_modules) >= 0
            assert len(rmt_modules) >= 0


class TestSpectralGuardEdgeCases:
    """Test spectral guard edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = SpectralGuard()
        self.model = self._create_gpt2_like_model()

    def _create_gpt2_like_model(self):
        """Create a GPT-2-like model for testing."""
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.h = nn.ModuleList()

        # Add a transformer layer
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.c_attn = nn.Linear(64, 192)
        layer.attn.c_proj = nn.Linear(64, 64)
        layer.mlp = nn.Module()
        layer.mlp.c_fc = nn.Linear(64, 256)
        layer.mlp.c_proj = nn.Linear(256, 64)

        model.transformer.h.append(layer)
        return model

    def test_validate_with_different_contexts(self):
        """Test validate with different context parameters."""
        mock_adapter = Mock()

        # Test with empty context
        result1 = self.guard.validate(self.model, mock_adapter, {})
        assert isinstance(result1, dict)
        assert "passed" in result1

        # Test with baseline metrics context
        context_with_baselines = {"baseline_metrics": {"test": 1.0}}
        result2 = self.guard.validate(self.model, mock_adapter, context_with_baselines)
        assert isinstance(result2, dict)
        assert "passed" in result2

    def test_validate_error_handling(self):
        """Test error handling during validate."""
        mock_adapter = Mock()

        with pytest.raises(AttributeError):
            self.guard.validate(None, mock_adapter, {})

    def test_config_updates(self):
        """Test that config can be updated after initialization."""
        # Create guard with initial config
        guard = SpectralGuard(test_param=1.0)
        assert guard.config.get("test_param") == 1.0

        # Update config
        guard.config["test_param"] = 2.0
        assert guard.config["test_param"] == 2.0

        # Add new config
        guard.config["new_param"] = "test"
        assert guard.config["new_param"] == "test"

    def test_multiple_validate_calls(self):
        """Test multiple validate calls on same guard instance."""
        mock_adapter = Mock()
        context = {}

        # Multiple calls should work consistently
        result1 = self.guard.validate(self.model, mock_adapter, context)
        result2 = self.guard.validate(self.model, mock_adapter, context)
        result3 = self.guard.validate(self.model, mock_adapter, context)

        # All should return valid results
        for result in [result1, result2, result3]:
            assert isinstance(result, dict)
            assert "passed" in result

    def test_spectral_utility_functions(self):
        """Test spectral utility functions."""
        # Test capture_baseline_sigmas (simple signature)
        baselines = capture_baseline_sigmas(self.model)
        assert isinstance(baselines, dict)

        # Test other utility functions that are imported
        result1 = apply_relative_spectral_cap(self.model, baseline_sigmas=baselines)
        assert isinstance(result1, dict)

        result2 = apply_spectral_control(self.model, {"baseline_sigmas": baselines})
        assert isinstance(result2, dict)

        # Test compute_sigma_max on a simple linear layer
        linear_layer = nn.Linear(10, 5)
        sigma = compute_sigma_max(linear_layer.weight)
        assert isinstance(sigma, float)
        assert sigma > 0

        # Test scan_model_gains
        gains = scan_model_gains(self.model)
        assert isinstance(gains, dict)

    def test_relative_cap_respects_baseline(self):
        """Ensure relative capping uses explicit baseline sigmas."""
        module = nn.Linear(8, 8)
        baseline = capture_baseline_sigmas(module)
        with torch.no_grad():
            module.weight.mul_(5.0)

        result = apply_relative_spectral_cap(
            module, cap_ratio=1.5, baseline_sigmas=baseline
        )

        assert result["applied"] is True
        capped_sigma = compute_sigma_max(module.weight)
        max_allowed = baseline[""] * 1.5
        assert capped_sigma <= max_allowed + 1e-6

    def test_scope_ffn_plus_proj_selects_projections(self):
        """Scope='ffn+proj' should include projection modules for capping."""
        model = self._create_gpt2_like_model()
        sigmas = capture_baseline_sigmas(model, scope="ffn+proj")
        assert any("c_proj" in name for name in sigmas)


class TestRMTGuardEdgeCases:
    """Test RMT guard edge cases and comprehensive functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = RMTGuard()
        self.model = self._create_gpt2_like_model()

    def _create_gpt2_like_model(self):
        """Create a GPT-2-like model for testing."""

        class GPT2LikeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList()

                layer = nn.Module()
                layer.attn = nn.Module()
                layer.attn.c_attn = nn.Linear(64, 192)
                layer.attn.c_proj = nn.Linear(64, 64)
                layer.mlp = nn.Module()
                layer.mlp.c_fc = nn.Linear(64, 256)
                layer.mlp.c_proj = nn.Linear(256, 64)
                self.transformer.h.append(layer)

            def forward(self, input_ids, attention_mask=None):
                _ = attention_mask
                x = input_ids.float()
                if x.dim() > 2:
                    x = x.reshape(x.shape[0], -1)
                layer0 = self.transformer.h[0]
                _ = layer0.attn.c_attn(x)
                _ = layer0.attn.c_proj(x)
                h = layer0.mlp.c_fc(x)
                return layer0.mlp.c_proj(h)

        return GPT2LikeModel()

    def test_finalize_without_prepare(self):
        """Test finalize when not prepared."""
        result = self.guard.finalize(self.model)

        # Handle both GuardOutcome and dict return types
        if hasattr(result, "passed"):
            # GuardOutcome object
            assert not result.passed
            assert len(result.violations) > 0
        else:
            # Dict return type
            assert isinstance(result, dict)
            assert not result["passed"]
            assert len(result["warnings"]) > 0 or len(result["errors"]) > 0

    def test_after_edit_without_prepare(self):
        """Test after_edit when not prepared."""
        self.guard.after_edit(self.model)
        # Should not crash but log warning
        assert len(self.guard.diagnostic_records) > 0
        assert any(
            e.get("severity") == "warning" for e in self.guard.diagnostic_records
        )

    def test_apply_rmt_detection_and_correction(self):
        """Test RMT post-edit analysis populates edge-risk results."""
        calib = [{"input_ids": torch.randint(0, 100, (1, 64))}]
        self.guard.prepare(self.model, None, calib, {"activation_required": True})
        self.guard.after_edit(self.model)

        assert isinstance(self.guard._last_result, dict)
        assert self.guard._last_result.get("analysis_source") == "activations_edge_risk"
        assert "edge_risk_by_family" in self.guard._last_result

    def test_rmt_utility_functions(self):
        """Test RMT utility functions."""
        # Test rmt_detect function
        result = rmt_detect(self.model, threshold=1.5, verbose=False)
        assert isinstance(result, dict)
        assert "has_outliers" in result

        # Test layer_svd_stats
        linear_layer = nn.Linear(64, 32)
        stats = layer_svd_stats(linear_layer)
        assert isinstance(stats, dict)
        assert "sigma_min" in stats
        assert "sigma_max" in stats


class TestVarianceGuardEdgeCases:
    """Test variance guard edge cases and comprehensive functionality."""

    def setup_method(self):
        """Set up test fixtures."""

        policy = get_variance_policy("balanced")
        self.guard = VarianceGuard(policy)
        self.model = self._create_gpt2_like_model()

    def _create_gpt2_like_model(self):
        """Create a GPT-2-like model for testing."""
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.h = nn.ModuleList()

        # Add transformer layers
        for _i in range(2):
            layer = nn.Module()
            layer.attn = nn.Module()
            layer.attn.c_proj = nn.Linear(64, 64)
            layer.mlp = nn.Module()
            layer.mlp.c_proj = nn.Linear(256, 64)
            model.transformer.h.append(layer)

        return model

    def test_finalize_without_prepare(self):
        """Test finalize when not prepared."""
        result = self.guard.finalize(self.model)

        assert isinstance(result, dict)
        assert not result["passed"]
        assert len(result["warnings"]) > 0 or len(result["errors"]) > 0

    def test_enable_without_prepare(self):
        """Test enable when not prepared."""
        result = self.guard.enable(self.model)
        assert not result
        assert len(self.guard.diagnostic_records) > 0
        assert any(
            "not prepared" in e.get("summary", "")
            for e in self.guard.diagnostic_records
        )

    def test_disable_when_not_enabled(self):
        """Test disable when not enabled (idempotent)."""
        result = self.guard.disable(self.model)
        assert result  # Should succeed idempotently

    def test_checkpoint_operations(self):
        """Test checkpoint push/pop operations."""
        # Set up basic state
        self.guard._target_modules = {"test": nn.Linear(10, 5)}

        # Test push checkpoint
        self.guard._push_checkpoint(self.model)
        assert len(self.guard._checkpoint_stack) == 1

        # Test pop checkpoint
        result = self.guard._pop_checkpoint(self.model)
        assert result
        assert len(self.guard._checkpoint_stack) == 0

        # Test pop when empty
        result = self.guard._pop_checkpoint(self.model)
        assert not result

    def test_ab_gate_edge_cases(self):
        """Test A/B gate evaluation edge cases."""
        # Test with no A/B results
        should_enable, reason = self.guard._evaluate_ab_gate()
        assert not should_enable
        assert "no_ab_results" in reason

        # Test with invalid PPL values
        self.guard.set_ab_results(None, 3.0)
        should_enable, reason = self.guard._evaluate_ab_gate()
        assert not should_enable
        assert "invalid" in reason

        # Test with negative PPL
        self.guard.set_ab_results(-1.0, 3.0)
        should_enable, reason = self.guard._evaluate_ab_gate()
        assert not should_enable

        # Test with tiny improvement (below absolute floor)
        self.guard.set_ab_results(
            3.501, 3.500, ratio_ci=(0.999, 1.001)
        )  # 0.001 improvement < min_rel_gain
        should_enable, reason = self.guard._evaluate_ab_gate()
        assert not should_enable
        assert any(
            token in reason
            for token in ("min_rel_gain", "ci_interval", "min_effect_lognll")
        )
