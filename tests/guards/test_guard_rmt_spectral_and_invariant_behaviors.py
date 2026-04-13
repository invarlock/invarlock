"""
Guard invariant, RMT, and spectral behavior tests.
"""

import pytest
import torch
import torch.nn as nn

from invarlock.core.exceptions import GuardError
from invarlock.guards.invariants import (
    assert_invariants,
    check_all_invariants,
)
from invarlock.guards.rmt import RMTGuard
from invarlock.guards.rmt_analysis import (
    capture_baseline_mp_stats,
    layer_svd_stats,
)
from invarlock.guards.rmt_detection import (
    rmt_detect,
    rmt_detect_report,
    rmt_detect_with_names,
)
from invarlock.guards.rmt_math import (
    clip_full_svd,
    mp_bulk_edge,
    mp_bulk_edges,
    rmt_growth_ratio,
    within_deadband,
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


class TestGuardUtilityBehaviors:
    """Exercise guard utility behaviors across invariant and policy helpers."""

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
        for _i in range(2):
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

    def test_invariants_with_problematic_model(self):
        """Test invariant checking with edge cases."""
        # Test with model containing NaN
        bad_model = nn.Linear(10, 5)
        bad_model.weight.data[0, 0] = float("nan")

        with pytest.raises(AssertionError):
            assert_invariants(bad_model)

        # Test check_all_invariants with same model
        outcome = check_all_invariants(bad_model)
        assert not outcome.passed
        assert len(outcome.violations) > 0
        assert any(v.get("type") == "nan_violation" for v in outcome.violations)

    def test_invariants_with_extreme_values(self):
        """Test invariant checking with extreme parameter values."""
        # Test with very large values
        large_model = nn.Linear(10, 5)
        large_model.weight.data.fill_(2000.0)  # Above threshold

        outcome = check_all_invariants(large_model, threshold=1e-6)
        assert not outcome.passed
        assert any(v.get("type") == "range_violation" for v in outcome.violations)

        # Test with very small values
        small_model = nn.Linear(10, 5)
        small_model.weight.data.fill_(1e-8)  # Below threshold

        outcome = check_all_invariants(small_model, threshold=1e-6)
        assert not outcome.passed
        assert any(v.get("type") == "range_violation" for v in outcome.violations)

    def test_spectral_functions_edge_cases(self):
        """Test spectral functions with edge cases."""
        # Test the actually available spectral functions (only imported ones)
        baselines = capture_baseline_sigmas(self.model)
        result1 = apply_relative_spectral_cap(
            self.model, cap_ratio=2.0, baseline_sigmas=baselines
        )
        assert isinstance(result1, dict)
        assert not result1.get("applied")  # Placeholder returns False

        result2 = apply_spectral_control(
            self.model, {"test": True, "baseline_sigmas": baselines}
        )
        assert isinstance(result2, dict)
        assert not result2.get("applied")  # Placeholder returns False

        # Test compute_sigma_max with different inputs
        linear_layer = nn.Linear(20, 10)
        sigma1 = compute_sigma_max(linear_layer.weight)
        assert isinstance(sigma1, float)
        assert sigma1 > 0

        # Test with non-tensor input (fallback case)
        sigma2 = compute_sigma_max("not_a_tensor")
        assert isinstance(sigma2, float)
        assert sigma2 == 1.0  # Fallback value

        # Test scan_model_gains
        gains = scan_model_gains(self.model)
        assert isinstance(gains, dict)
        if "total_layers" in gains:
            assert gains["total_layers"] >= 0

        # Test capture_baseline_sigmas
        baselines = capture_baseline_sigmas(self.model)
        assert isinstance(baselines, dict)
        # Placeholder implementation may return empty dict, which is fine

    def test_rmt_functions_comprehensive(self):
        """Test RMT functions comprehensively."""
        from invarlock.guards.rmt_analysis import analyze_weight_distribution

        # Test mp_bulk_edges
        min_edge, max_edge = mp_bulk_edges(100, 50, whitened=True)
        assert isinstance(min_edge, float)
        assert isinstance(max_edge, float)
        assert min_edge >= 0
        assert max_edge > min_edge

        # Test rmt_growth_ratio
        ratio = rmt_growth_ratio(2.0, 1.5, 1.8, 1.4)
        assert isinstance(ratio, float)
        assert ratio > 0

        # Test within_deadband
        assert within_deadband(1.05, 1.0, 0.1)
        assert not within_deadband(1.15, 1.0, 0.1)

        # Test analyze_weight_distribution
        dist_stats = analyze_weight_distribution(self.model)
        assert isinstance(dist_stats, dict)
        if dist_stats:  # May be empty for some models
            assert "mean" in dist_stats
            assert "std" in dist_stats

        # Test clip_full_svd
        W = torch.randn(10, 8)
        W_clipped = clip_full_svd(W, clip_val=1.0)
        assert isinstance(W_clipped, torch.Tensor)
        assert W_clipped.shape == W.shape

        # Test with return_components
        U, S, Vt = clip_full_svd(W, clip_val=1.0, return_components=True)
        if U is not None:  # May be None if SVD fails
            assert isinstance(U, torch.Tensor)
            assert isinstance(S, torch.Tensor)
            assert isinstance(Vt, torch.Tensor)

    def test_policy_size_based_functions(self):
        """Test policy functions that depend on model size."""
        from invarlock.guards.policies import (
            get_policy_for_model_size,
            get_rmt_policy_for_model_size,
            get_variance_policy_for_model_size,
        )

        # Test with small model
        small_policy = get_policy_for_model_size(50_000_000)  # 50M params
        assert isinstance(small_policy, dict)
        assert small_policy["sigma_quantile"] == 0.98  # Should be aggressive

        # Test with large model
        large_policy = get_policy_for_model_size(2_000_000_000)  # 2B params
        assert isinstance(large_policy, dict)
        assert large_policy["sigma_quantile"] == 0.90  # Should be conservative

        # Test RMT policies by size
        rmt_small = get_rmt_policy_for_model_size(50_000_000)
        rmt_large = get_rmt_policy_for_model_size(2_000_000_000)
        assert rmt_small["margin"] > rmt_large["margin"]  # Aggressive vs conservative

        # Test variance policies by size
        var_small = get_variance_policy_for_model_size(50_000_000)
        var_large = get_variance_policy_for_model_size(2_000_000_000)
        assert (
            var_small["min_gain"] < var_large["min_gain"]
        )  # Aggressive vs conservative

    def test_validation_gate_functions(self):
        """Test validation gate utility functions."""
        from invarlock.guards.policies import get_validation_gate

        # Test all validation gates
        for gate_name in ["strict", "standard", "permissive"]:
            gate_config = get_validation_gate(gate_name)
            assert isinstance(gate_config, dict)
            assert "max_capping_rate" in gate_config
            assert "max_ppl_degradation" in gate_config

        # Test invalid gate name
        with pytest.raises(GuardError):
            get_validation_gate("invalid_gate")


class TestRMTGuardBehaviors:
    """Exercise RMT guard detection, reporting, and finalize behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = RMTGuard()
        self.model = self._create_transformer_model()

    def _create_transformer_model(self):
        """Create a transformer model with proper structure."""
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.h = nn.ModuleList()

        # Add transformer layers
        for _i in range(2):
            layer = nn.Module()
            layer.attn = nn.Module()
            layer.attn.c_attn = nn.Linear(128, 384)
            layer.attn.c_proj = nn.Linear(128, 128)
            layer.mlp = nn.Module()
            layer.mlp.c_fc = nn.Linear(128, 512)
            layer.mlp.c_proj = nn.Linear(512, 128)
            model.transformer.h.append(layer)

        return model

    def test_rmt_detect_with_parameters(self):
        """Test rmt_detect with various parameters to hit more branches."""
        # Test with detect_only=False and correction
        result = rmt_detect(
            self.model,
            threshold=1.2,
            detect_only=False,
            correction_factor=0.9,
            verbose=True,
            max_iterations=1,
        )
        assert isinstance(result, dict)

        # Test with baseline sigmas and MP stats
        baseline_sigmas = capture_baseline_mp_stats(self.model)
        result = rmt_detect(
            self.model,
            threshold=1.5,
            baseline_sigmas={},
            baseline_mp_stats=baseline_sigmas,
            deadband=0.1,
        )
        assert isinstance(result, dict)

        # Test with layer indices filter
        result = rmt_detect(self.model, threshold=1.5, layer_indices=[0], verbose=True)
        assert isinstance(result, dict)

        # Test with target layers filter
        result = rmt_detect(
            self.model, threshold=1.5, target_layers=["transformer.h.0"], verbose=True
        )
        assert isinstance(result, dict)

    def test_rmt_detect_with_names(self):
        """Test rmt_detect_with_names function."""
        result = rmt_detect_with_names(self.model, threshold=1.5, verbose=True)
        assert isinstance(result, dict)
        assert "has_outliers" in result
        assert "per_layer" in result
        assert "outliers" in result
        assert "layers" in result

    def test_rmt_detect_report(self):
        """Test rmt_detect_report function."""
        summary, per_layer = rmt_detect_report(self.model, threshold=1.5)
        assert isinstance(summary, dict)
        assert isinstance(per_layer, list)
        assert "has_outliers" in summary
        assert "max_ratio" in summary

    def test_layer_svd_stats_comprehensive(self):
        """Test layer_svd_stats with various parameters."""
        layer = self.model.transformer.h[0]

        # Basic test
        stats = layer_svd_stats(layer)
        assert isinstance(stats, dict)

        # With baseline sigmas
        baseline_sigmas = {"test_layer": 2.0}
        stats = layer_svd_stats(layer, baseline_sigmas, None, "test_layer")
        assert isinstance(stats, dict)

        # With baseline MP stats
        baseline_mp_stats = {
            "test_layer": {
                "mp_bulk_edge_base": 1.5,
                "r_mp_base": 1.2,
                "sigma_base": 2.0,
            }
        }
        stats = layer_svd_stats(layer, baseline_sigmas, baseline_mp_stats, "test_layer")
        assert isinstance(stats, dict)

    def test_mp_bulk_functions(self):
        """Test MP bulk edge functions comprehensively."""
        # Test mp_bulk_edges with different parameters
        min_edge, max_edge = mp_bulk_edges(100, 50, whitened=False)
        assert isinstance(min_edge, float)
        assert isinstance(max_edge, float)

        min_edge_w, max_edge_w = mp_bulk_edges(100, 50, whitened=True)
        assert min_edge_w != min_edge  # Should be different

        # Test edge cases
        min_edge_zero, max_edge_zero = mp_bulk_edges(0, 50)
        assert min_edge_zero == 0.0
        assert max_edge_zero == 0.0

        # Test mp_bulk_edge single value
        edge = mp_bulk_edge(100, 50, whitened=False)
        assert edge == max_edge

        edge_zero = mp_bulk_edge(0, 0)
        assert edge_zero == 0.0

    def test_clip_full_svd_edge_cases(self):
        """Test clip_full_svd with edge cases."""
        # Test with various matrix shapes
        W = torch.randn(20, 10)
        W_clipped = clip_full_svd(W, clip_val=2.0)
        assert W_clipped.shape == W.shape

        # Test with return_components=True
        U, S, Vt = clip_full_svd(W, clip_val=2.0, return_components=True)
        if U is not None:
            assert isinstance(U, torch.Tensor)
            assert isinstance(S, torch.Tensor)
            assert isinstance(Vt, torch.Tensor)

        # Test with problematic matrix (should handle gracefully)
        bad_W = torch.zeros(5, 5)
        result = clip_full_svd(bad_W, clip_val=1.0)
        assert isinstance(result, torch.Tensor)

    def test_analyze_weight_distribution(self):
        """Test analyze_weight_distribution function comprehensively."""
        from invarlock.guards.rmt_analysis import analyze_weight_distribution

        stats = analyze_weight_distribution(self.model, n_bins=20)
        assert isinstance(stats, dict)

        if stats:  # May be empty for some models
            assert "mean" in stats
            assert "std" in stats
            assert "histogram" in stats
            assert "bin_edges" in stats

            if "singular_values" in stats:
                assert "condition_number" in stats["singular_values"]

            if "mp_edges" in stats:
                assert "min" in stats["mp_edges"]
                assert "max" in stats["mp_edges"]

    def test_guard_finalize_comprehensive(self):
        """Test RMT finalize ε-band evaluation."""
        self.guard.prepare(self.model, None, None, {})

        self.guard.baseline_edge_risk_by_family = {
            "attn": 1.0,
            "ffn": 1.0,
            "embed": 0.0,
            "other": 0.0,
        }
        self.guard.edge_risk_by_family = {
            "attn": 1.4,
            "ffn": 1.4,
            "embed": 0.0,
            "other": 0.0,
        }
        self.guard.epsilon_by_family = {
            "attn": 0.5,
            "ffn": 0.5,
            "embed": 0.0,
            "other": 0.0,
        }

        result = self.guard.finalize(self.model)
        metrics = result.metrics if hasattr(result, "metrics") else result["metrics"]
        passed = result.passed if hasattr(result, "passed") else result["passed"]
        assert passed is True
        assert metrics["epsilon_violations"] == []

        # Exceed allowance → fail
        self.guard.edge_risk_by_family["attn"] = 1.6
        result = self.guard.finalize(self.model)
        metrics = result.metrics if hasattr(result, "metrics") else result["metrics"]
        passed = result.passed if hasattr(result, "passed") else result["passed"]
        assert passed is False
        assert metrics["epsilon_violations"]


class TestSpectralGuardBehaviors:
    """Exercise spectral guard control and measurement behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = SpectralGuard()
        self.model = self._create_complex_model()

    def _create_complex_model(self):
        """Create a complex model for comprehensive testing."""
        model = nn.Module()
        model.transformer = nn.Module()
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

        return model

    def test_spectral_control_comprehensive(self):
        """Test apply_spectral_control with various parameters."""
        baselines = capture_baseline_sigmas(self.model)

        # Test with different policy parameters
        result = apply_spectral_control(
            model=self.model,
            policy={
                "sigma_quantile": 0.90,
                "scope": "ffn",
                "baseline_sigmas": baselines,
            },
        )
        assert isinstance(result, dict)
        assert not result.get("applied")  # Placeholder returns False

        # Test with different policy
        result = apply_spectral_control(
            model=self.model,
            policy={"scope": "all", "verbose": False, "baseline_sigmas": baselines},
        )
        assert isinstance(result, dict)
        assert not result.get("applied")  # Placeholder returns False

    def test_apply_relative_spectral_cap_comprehensive(self):
        """Test apply_relative_spectral_cap with various parameters."""
        baselines = capture_baseline_sigmas(self.model)

        # Test with different cap ratios
        result = apply_relative_spectral_cap(
            model=self.model, cap_ratio=1.5, baseline_sigmas=baselines
        )
        assert isinstance(result, dict)
        assert not result.get("applied")  # Placeholder returns False

        # Test with different cap ratio
        result = apply_relative_spectral_cap(
            model=self.model, cap_ratio=2.0, baseline_sigmas=baselines
        )
        assert isinstance(result, dict)
        assert not result.get("applied")  # Placeholder returns False

    def test_scan_model_gains_basic(self):
        """Test scan_model_gains basic functionality."""
        # Test basic functionality (no scope parameter in minimal implementation)
        gains = scan_model_gains(self.model)

        assert isinstance(gains, dict)
        if "total_layers" in gains:
            assert gains["total_layers"] >= 0
        if "scanned_gains" in gains:
            assert isinstance(gains["scanned_gains"], int)

    def test_capture_baseline_sigmas_comprehensive(self):
        """Test capture_baseline_sigmas with different scenarios."""
        # Test basic functionality (minimal implementation has simple signature)
        baselines = capture_baseline_sigmas(self.model)

        assert isinstance(baselines, dict)
        # Placeholder implementation may return empty dict or module->sigma mappings

        # Test multiple calls to ensure consistency
        baselines2 = capture_baseline_sigmas(self.model)
        assert isinstance(baselines2, dict)

        # Test with empty model
        empty_model = nn.Module()
        empty_baselines = capture_baseline_sigmas(empty_model)
        assert isinstance(empty_baselines, dict)

    def test_spectral_functions_consistency(self):
        """Test consistency of spectral functions."""
        # Test that functions return consistent types
        baselines = capture_baseline_sigmas(self.model)
        result1 = apply_spectral_control(self.model, {"baseline_sigmas": baselines})
        result2 = apply_relative_spectral_cap(self.model, baseline_sigmas=baselines)

        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

        # Both should have "applied" field indicating placeholder status
        assert "applied" in result1
        assert "applied" in result2

        # Test that capture_baseline_sigmas works with the model
        baselines = capture_baseline_sigmas(self.model)
        assert isinstance(baselines, dict)

        # Test compute_sigma_max with multiple layers
        for _name, module in self.model.named_modules():
            if hasattr(module, "weight") and module.weight.ndim == 2:
                sigma = compute_sigma_max(module.weight)
                assert isinstance(sigma, float)
                assert sigma > 0

    def test_spectral_error_handling(self):
        """Test error handling in spectral functions."""
        # Test with None model (should handle gracefully)
        result = apply_spectral_control(None, {})
        assert isinstance(result, dict)
        assert result.get("applied") is False

        # Test with empty policy
        baselines = capture_baseline_sigmas(self.model)
        result = apply_spectral_control(self.model, {"baseline_sigmas": baselines})
        assert isinstance(result, dict)

        # Test apply_relative_spectral_cap with edge case values
        result = apply_relative_spectral_cap(
            self.model, cap_ratio=0.1, baseline_sigmas=baselines
        )
        assert isinstance(result, dict)

        result = apply_relative_spectral_cap(
            self.model, cap_ratio=10.0, baseline_sigmas=baselines
        )
        assert isinstance(result, dict)

    def test_spectral_module_analysis(self):
        """Test spectral functions on different module types."""
        # Test with linear layers
        linear = nn.Linear(64, 32)
        sigma_linear = compute_sigma_max(linear.weight)
        assert isinstance(sigma_linear, float)
        assert sigma_linear > 0

        # Test with conv layer
        conv = nn.Conv2d(3, 16, 3)
        sigma_conv = compute_sigma_max(conv.weight.view(conv.weight.size(0), -1))
        assert isinstance(sigma_conv, float)
        assert sigma_conv > 0

        # Test model scanning
        gains = scan_model_gains(self.model)
        assert isinstance(gains, dict)

        # Test baseline capture
        baselines = capture_baseline_sigmas(self.model)
        assert isinstance(baselines, dict)
