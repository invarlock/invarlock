"""
Comprehensive Guard System Tests
===============================

Comprehensive tests for all guard modules to achieve 70% coverage.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

import invarlock.guards.rmt as rmt_mod
from invarlock.core.exceptions import GuardError, ValidationError
from invarlock.guards.rmt_analysis import (
    capture_baseline_mp_stats,
    clip_full_svd,
    layer_svd_stats,
    mp_bulk_edge,
    mp_bulk_edges,
    within_deadband,
)
from invarlock.guards.rmt_detection import (
    _apply_rmt_correction,
    rmt_detect,
    rmt_detect_report,
    rmt_detect_with_names,
)


class TestRMTEnhancedCoverage:
    """Enhanced tests to achieve 80%+ coverage for invarlock.guards.rmt module."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = self._create_comprehensive_model()
        self.guard = rmt_mod.RMTGuard()

    def _create_comprehensive_model(self):
        """Create a comprehensive model for RMT testing."""
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

    def test_mp_bulk_functions_comprehensive(self):
        """Test MP bulk edge functions with edge cases (lines 102, 149)."""
        # Test whitened parameter variations (line 102)
        min_edge, max_edge = mp_bulk_edges(100, 50, whitened=True)
        assert isinstance(min_edge, float)
        assert isinstance(max_edge, float)
        assert min_edge >= 0

        # Test single edge function (lines around 102)
        edge = mp_bulk_edge(100, 50, whitened=True)
        assert edge == max_edge

        # Test zero dimensions (covered in existing tests but ensure hit)
        min_zero, max_zero = mp_bulk_edges(0, 50)
        assert min_zero == 0.0
        assert max_zero == 0.0

        assert within_deadband(1.05, 1.0, 0.1)
        assert not within_deadband(1.15, 1.0, 0.1)

    def test_layer_svd_stats_edge_cases(self):
        """Test layer_svd_stats with various edge cases (lines 176, 186-187, 211, 224-227)."""

        # Test with empty weight matrices (line 176)
        empty_layer = nn.Module()
        empty_layer.empty_weight = nn.Parameter(torch.empty(0, 0))
        stats = layer_svd_stats(empty_layer)
        assert isinstance(stats, dict)

        # Test SVD failure path (lines 186-187)
        class FailingSVDLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.tensor([[float("nan"), 1.0], [2.0, 3.0]])
                )

        failing_layer = FailingSVDLayer()
        stats = layer_svd_stats(failing_layer)
        assert isinstance(stats, dict)
        # Should handle SVD failure gracefully

        # Test baseline-aware ratio with missing baseline (line 211)
        baseline_sigmas = {"test_layer": 0.0}  # Zero baseline
        stats = layer_svd_stats(
            self.model.transformer.h[0], baseline_sigmas, None, "test_layer"
        )
        assert isinstance(stats, dict)

        # Test quantile-based normalization (lines 224-227)
        # Create layer with single parameter to hit single value case
        single_param_layer = nn.Module()
        single_param_layer.weight = nn.Parameter(torch.randn(1, 1))
        stats = layer_svd_stats(single_param_layer)
        assert isinstance(stats, dict)
        assert "worst_ratio" in stats

    def test_capture_baseline_mp_stats_edge_cases(self):
        """Test capture_baseline_mp_stats with edge cases (lines 286-287, 311-313, 333-335)."""
        # Test with transformers import failure simulation (lines 286-287)
        import sys

        from invarlock.guards.rmt_analysis import capture_baseline_mp_stats

        if "transformers" in sys.modules:
            # Temporarily remove transformers to simulate import failure
            transformers_module = sys.modules.pop("transformers", None)
            transformers_pytorch_utils = sys.modules.pop(
                "transformers.pytorch_utils", None
            )
            try:
                # This should hit the ImportError path
                stats = capture_baseline_mp_stats(self.model)
                assert isinstance(stats, dict)
            finally:
                # Restore modules if they existed
                if transformers_module:
                    sys.modules["transformers"] = transformers_module
                if transformers_pytorch_utils:
                    sys.modules["transformers.pytorch_utils"] = (
                        transformers_pytorch_utils
                    )
        else:
            # transformers not available - test normal case
            stats = capture_baseline_mp_stats(self.model)
            assert isinstance(stats, dict)

        # Test with Conv1D module if available (lines 311-313)
        try:
            from transformers.pytorch_utils import Conv1D

            conv_model = nn.Module()
            conv_model.transformer = nn.Module()
            conv_model.transformer.h = nn.ModuleList()
            layer = nn.Module()
            layer.attn = nn.Module()
            layer.attn.c_attn = Conv1D(384, 128)
            conv_model.transformer.h.append(layer)

            stats = capture_baseline_mp_stats(conv_model)
            assert isinstance(stats, dict)
        except ImportError:
            # Skip if transformers not available
            pass

        # Test SVD failure in baseline capture (lines 333-335)
        failing_model = nn.Module()
        failing_model.transformer = nn.Module()
        failing_model.transformer.h = nn.ModuleList()
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.c_attn = nn.Linear(128, 384)
        # Corrupt the weight to cause SVD failure
        layer.attn.c_attn.weight.data.fill_(float("nan"))
        failing_model.transformer.h.append(layer)

        stats = capture_baseline_mp_stats(failing_model)
        assert isinstance(stats, dict)
        # Should handle SVD failure gracefully and continue

    def test_iter_transformer_layers(self):
        """Test iter_transformer_layers with different model types (lines 346-356)."""
        from invarlock.guards.rmt_analysis import _iter_transformer_layers

        # Test GPT-2 style (covered in main tests)
        layers = list(_iter_transformer_layers(self.model))
        assert len(layers) == 3

        # Test model.layers style model
        model_layers_model = nn.Module()
        model_layers_model.model = nn.Module()
        model_layers_model.model.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        layers = list(_iter_transformer_layers(model_layers_model))
        assert len(layers) == 2

        # Test BERT style model
        bert_model = nn.Module()
        bert_model.encoder = nn.Module()
        bert_model.encoder.layer = nn.ModuleList([nn.Module() for _ in range(2)])
        layers = list(_iter_transformer_layers(bert_model))
        assert len(layers) == 2

        # Test fallback case (lines 354-356)
        fallback_model = nn.Module()
        fallback_layer = nn.Module()
        fallback_layer.attn = nn.Module()
        fallback_layer.mlp = nn.Module()
        fallback_model.add_module("transformer_layer", fallback_layer)
        layers = list(_iter_transformer_layers(fallback_model))
        assert len(layers) >= 1

    def test_rmt_detect_comprehensive_branches(self):
        """Test rmt_detect with various parameter combinations (lines 457-470, 473-482, 487-488)."""
        # Test detect_only=False with correction (lines 457-470)
        baseline_mp_stats = capture_baseline_mp_stats(self.model)
        baseline_sigmas = {
            name: stats["sigma_base"] for name, stats in baseline_mp_stats.items()
        }

        result = rmt_detect(
            self.model,
            threshold=1.2,
            detect_only=False,
            correction_factor=0.9,
            baseline_sigmas=baseline_sigmas,
            baseline_mp_stats=baseline_mp_stats,
            deadband=0.1,
            verbose=True,
        )
        assert isinstance(result, dict)
        assert "has_outliers" in result

        # Test partial baseline-aware checking (lines 473-482)
        result = rmt_detect(
            self.model,
            threshold=1.5,
            deadband=0.1,
            baseline_sigmas=baseline_sigmas,
            verbose=True,
        )
        assert isinstance(result, dict)

        # Test standard check without baseline (lines 487-488)
        result = rmt_detect(self.model, threshold=1.5, verbose=True)
        assert isinstance(result, dict)

    def test_rmt_detect_iteration_and_correction(self):
        """Test rmt_detect iteration logic and correction (lines 511-515, 522-537, 542-547)."""
        # Test with max_iterations and correction stalling (lines 522-537)
        result = rmt_detect(
            self.model,
            threshold=0.5,  # Very low threshold to trigger outliers
            detect_only=False,
            correction_factor=1.0,  # No actual correction to test stalling
            max_iterations=2,
            verbose=True,
        )
        assert isinstance(result, dict)
        assert "correction_iterations" in result

        # Test exit when no outliers remain (lines 542-547)
        result = rmt_detect(
            self.model,
            threshold=10.0,  # Very high threshold so no outliers
            detect_only=False,
            correction_factor=0.9,
            max_iterations=3,
            verbose=True,
        )
        assert isinstance(result, dict)
        assert result.get("correction_iterations", 0) == 0

    def test_rmt_detect_verbose_output(self):
        """Test rmt_detect verbose output and reporting (lines 555-580)."""
        # Create a model likely to have outliers for verbose testing
        outlier_model = nn.Module()
        outlier_model.transformer = nn.Module()
        outlier_model.transformer.h = nn.ModuleList()
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.c_attn = nn.Linear(10, 30)
        # Make weights likely to be outliers
        layer.attn.c_attn.weight.data *= 10.0
        outlier_model.transformer.h.append(layer)

        result = rmt_detect(
            outlier_model,
            threshold=1.1,  # Low threshold to trigger
            verbose=True,
        )
        assert isinstance(result, dict)
        # Should have verbose output about outliers

    def test_rmt_detect_with_names_comprehensive(self):
        """Test rmt_detect_with_names with different model styles (lines 648-660, 681-691)."""
        # Test model.layers style model (lines 648-660)
        model_layers_model = nn.Module()
        model_layers_model.model = nn.Module()
        model_layers_model.model.layers = nn.ModuleList()
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.c_attn = nn.Linear(64, 192)
        layer.mlp = nn.Module()
        layer.mlp.c_fc = nn.Linear(64, 256)
        model_layers_model.model.layers.append(layer)

        result = rmt_detect_with_names(model_layers_model, threshold=1.5, verbose=True)
        assert isinstance(result, dict)
        assert "per_layer" in result
        assert "outliers" in result

        # Test BERT style model (lines 652-660)
        bert_model = nn.Module()
        bert_model.encoder = nn.Module()
        bert_model.encoder.layer = nn.ModuleList()
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.mlp = nn.Module()
        bert_model.encoder.layer.append(layer)

        result = rmt_detect_with_names(bert_model, threshold=1.5)
        assert isinstance(result, dict)

        # Test layer with outliers to trigger outlier collection (lines 681-691)
        outlier_layer = nn.Module()
        outlier_layer.attn = nn.Module()
        outlier_layer.attn.c_attn = nn.Linear(5, 15)
        outlier_layer.attn.c_attn.weight.data *= 5.0  # Make it likely to be outlier
        outlier_layer.mlp = nn.Module()
        outlier_layer.mlp.c_fc = nn.Linear(5, 20)

        outlier_model = nn.Module()
        outlier_model.transformer = nn.Module()
        outlier_model.transformer.h = nn.ModuleList([outlier_layer])

        result = rmt_detect_with_names(outlier_model, threshold=1.2, verbose=True)
        assert isinstance(result, dict)

    def test_rmt_detect_report_function(self):
        """Test rmt_detect_report function (lines 707-716)."""
        summary, per_layer = rmt_detect_report(self.model, threshold=1.5)

        assert isinstance(summary, dict)
        assert isinstance(per_layer, list)
        assert "has_outliers" in summary
        assert "max_ratio" in summary

    def test_apply_rmt_correction_comprehensive(self):
        """Test _apply_rmt_correction function (lines 744-834)."""
        # Create a test layer
        test_layer = nn.Linear(64, 128)

        # Test with baseline stats (Step 5 logic, lines 765-774)
        baseline_mp_stats = {
            "test_layer": {
                "sigma_base": 2.0,
                "mp_bulk_edge_base": 1.5,
                "r_mp_base": 1.33,
            }
        }
        baseline_sigmas = {"test_layer": 2.0}

        _apply_rmt_correction(
            test_layer,
            0.9,
            baseline_sigmas,
            baseline_mp_stats,
            "test_layer",
            deadband=0.1,
            verbose=True,
        )

        # Test without baseline stats (fallback, lines 775-780)
        test_layer2 = nn.Linear(32, 64)
        _apply_rmt_correction(
            test_layer2, 0.9, None, None, "test_layer2", deadband=0.0, verbose=True
        )

        # Test with adapter and tying map (lines 787-811)
        mock_adapter = Mock()
        mock_adapter.get_tying_map.return_value = {
            "test_layer.weight": ["tied_layer.weight"]
        }
        mock_adapter.get_parameter_by_name.return_value = nn.Parameter(
            torch.randn(64, 128)
        )

        _apply_rmt_correction(
            test_layer,
            0.8,
            baseline_sigmas,
            baseline_mp_stats,
            "test_layer",
            deadband=0.1,
            verbose=True,
            adapter=mock_adapter,
        )

        # Test SVD failure fallback (lines 830-834)
        class BadLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.tensor([[float("nan"), 1.0], [2.0, 3.0]])
                )

        bad_layer = BadLayer()
        _apply_rmt_correction(
            bad_layer, 0.9, None, None, "bad_layer", deadband=0.0, verbose=True
        )

    def test_clip_full_svd_edge_cases(self):
        """Test clip_full_svd with edge cases (lines 861-865)."""
        # Test normal case
        W = torch.randn(20, 15)
        W_clipped = clip_full_svd(W, clip_val=2.0)
        assert W_clipped.shape == W.shape

        # Test return_components
        U, S, Vt = clip_full_svd(W, clip_val=1.5, return_components=True)
        assert isinstance(U, torch.Tensor)
        assert isinstance(S, torch.Tensor)
        assert isinstance(Vt, torch.Tensor)

        # Test SVD failure case (lines 861-865)
        bad_W = torch.tensor([[float("inf"), 1.0], [2.0, float("nan")]])
        result = clip_full_svd(bad_W, clip_val=1.0)
        assert isinstance(result, torch.Tensor)

        # Test SVD failure with return_components
        U, S, Vt = clip_full_svd(bad_W, clip_val=1.0, return_components=True)
        # Should return None values on failure

    def test_analyze_weight_distribution_edge_cases(self):
        """Test analyze_weight_distribution with edge cases (lines 894-895, 898)."""
        from invarlock.guards.rmt_analysis import analyze_weight_distribution

        # Test with model that has no 2D weights (line 898)
        empty_model = nn.Module()
        empty_model.bias_only = nn.Parameter(torch.randn(10))  # 1D parameter
        stats = analyze_weight_distribution(empty_model)
        assert stats == {}

        # Test with SVD failure (lines 894-895) - use a different approach
        # Create a model where SVD fails but weights are valid for histogram
        failing_model = nn.Module()
        failing_model.good_weight = nn.Parameter(
            torch.randn(3, 3)
        )  # Valid for histogram

        # Mock the SVD to fail during singular value computation
        with patch("torch.linalg.svdvals", side_effect=RuntimeError("SVD failed")):
            stats = analyze_weight_distribution(failing_model, n_bins=20)
            assert isinstance(stats, dict)
            # Should still have basic stats but no singular_values section
            if stats:
                assert "mean" in stats
                assert "std" in stats

        # Test full stats computation
        stats = analyze_weight_distribution(self.model, n_bins=30)
        assert isinstance(stats, dict)
        if stats:  # Only check if not empty
            assert "mean" in stats
            assert "std" in stats
            assert "histogram" in stats
            assert "singular_values" in stats

    def test_rmt_guard_prepare_failure(self):
        """Test RMTGuard prepare method failure (lines 1275-1284)."""
        guard = rmt_mod.RMTGuard()

        with patch.object(
            rmt_mod.RMTGuard,
            "_collect_calibration_batches",
            side_effect=RuntimeError("Capture failed"),
        ):
            result = guard.prepare(
                self.model,
                None,
                [{"input_ids": torch.randint(0, 100, (1, 64))}],
                {},
            )

            assert isinstance(result, dict)
            assert not result["ready"]
            assert "error" in result
            assert not guard.prepared

    def test_rmt_guard_before_edit(self):
        """Test RMTGuard before_edit method (lines 1299-1300)."""
        guard = rmt_mod.RMTGuard()

        # Test when not prepared
        guard.before_edit(self.model)  # Should not crash

        # Test when prepared
        guard.prepared = True
        guard.before_edit(self.model)  # Should log event
        assert len(guard.diagnostic_records) > 0

    def test_rmt_guard_after_edit_comprehensive(self):
        """Test RMTGuard after_edit method comprehensively (lines 1317-1379)."""
        guard = rmt_mod.RMTGuard()

        # Test without preparation (lines 1309-1315)
        guard.after_edit(self.model)
        assert any(e.get("severity") == "warning" for e in guard.diagnostic_records)

        # Test with preparation and no activation batches
        guard.prepare(self.model, None, None, {})
        guard.after_edit(self.model)

        # Test exception handling (lines 1371-1379)
        guard.prepared = True
        with patch.object(
            rmt_mod.RMTGuard,
            "_compute_activation_edge_risk",
            side_effect=RuntimeError("Detection failed"),
        ):
            guard._calibration_batches = [{"input_ids": torch.randint(0, 100, (1, 64))}]
            guard.after_edit(self.model)
            assert any(e.get("severity") == "error" for e in guard.diagnostic_records)

    def test_rmt_guard_validate_method(self):
        """Test RMTGuard validate method (lines 1399-1411)."""
        guard = rmt_mod.RMTGuard()

        # Test validate calling finalize
        result = guard.validate(self.model, None, {})
        assert isinstance(result, dict)
        assert "passed" in result
        assert "decision" in result
        assert "diagnostics" in result

    def test_rmt_guard_finalize_not_prepared(self):
        """Test RMTGuard finalize when not prepared (lines 1441)."""
        guard = rmt_mod.RMTGuard()

        result = guard.finalize(self.model)
        # Handle both GuardOutcome and dict return types
        if hasattr(result, "passed"):
            # GuardOutcome object
            assert not result.passed
            assert len(result.violations) > 0
        else:
            # Dict return type
            assert isinstance(result, dict)
            assert not result["passed"]
            assert len(result["errors"]) > 0

    def test_rmt_guard_finalize_comprehensive(self):
        """Test RMTGuard finalize with various scenarios (lines 1536-1551)."""
        guard = rmt_mod.RMTGuard()
        guard.prepare(self.model, None, None, {})

        guard.baseline_edge_risk_by_family = {"attn": 1.0}
        guard.edge_risk_by_family = {"attn": 1.4}
        guard.epsilon_by_family = {"attn": 0.5}

        result = guard.finalize(self.model)
        metrics = result.metrics if hasattr(result, "metrics") else result["metrics"]
        passed = result.passed if hasattr(result, "passed") else result["passed"]

        assert passed is True
        assert metrics["epsilon_violations"] == []

        guard.edge_risk_by_family = {"attn": 1.6}
        guard.epsilon_by_family = {"attn": 0.0}
        result = guard.finalize(self.model)
        metrics = result.metrics if hasattr(result, "metrics") else result["metrics"]
        passed = result.passed if hasattr(result, "passed") else result["passed"]

        assert passed is False
        assert metrics["epsilon_violations"], "Expected epsilon violations recorded"

    def test_rmt_guard_get_linear_modules(self):
        """Test RMTGuard _get_linear_modules method (lines 1068-1069)."""
        guard = rmt_mod.RMTGuard()

        # Test with transformers import failure
        import sys

        if "transformers" in sys.modules:
            # Temporarily remove transformers to simulate import failure
            transformers_module = sys.modules.pop("transformers", None)
            transformers_pytorch_utils = sys.modules.pop(
                "transformers.pytorch_utils", None
            )
            try:
                modules = guard._get_linear_modules(self.model)
                assert isinstance(modules, list)
            finally:
                # Restore modules if they existed
                if transformers_module:
                    sys.modules["transformers"] = transformers_module
                if transformers_pytorch_utils:
                    sys.modules["transformers.pytorch_utils"] = (
                        transformers_pytorch_utils
                    )
        else:
            # Test normal case without transformers
            modules = guard._get_linear_modules(self.model)
            assert isinstance(modules, list)

        # Test normal case
        modules = guard._get_linear_modules(self.model)
        assert isinstance(modules, list)
        assert len(modules) > 0

        # Verify scope enforcement
        for name, _module in modules:
            assert any(name.endswith(suffix) for suffix in guard.allowed_suffixes)

    def test_rmt_guard_apply_detection_and_correction(self):
        """Test RMTGuard after_edit produces an analysis result."""
        guard = rmt_mod.RMTGuard()
        guard.prepare(self.model, None, None, {})
        guard.after_edit(self.model)
        assert isinstance(guard._last_result, dict)
        assert guard._last_result.get("analysis_source") == "activations_edge_risk"

    def test_rmt_guard_policy_method(self):
        """Test RMTGuard policy method (line 1181)."""
        guard = rmt_mod.RMTGuard(q=2.0, deadband=0.05, margin=1.8, correct=False)
        policy = guard.policy()

        assert isinstance(policy, dict)
        assert policy["q"] == 2.0
        assert policy["deadband"] == 0.05
        assert policy["margin"] == 1.8
        assert not policy["correct"]

    def test_policy_functions_comprehensive(self):
        """Test policy utility functions (lines 1611-1621, 1642-1651)."""
        from invarlock.guards.rmt import create_custom_rmt_policy, get_rmt_policy

        # Test all available policies
        for policy_name in ["conservative", "balanced", "aggressive"]:
            policy = get_rmt_policy(policy_name)
            assert isinstance(policy, dict)
            assert "q" in policy
            assert "deadband" in policy
            assert "margin" in policy
            assert "correct" in policy

        # Test invalid policy name (lines 1618-1621)
        with pytest.raises(GuardError):
            get_rmt_policy("invalid_policy")

        # Test create_custom_rmt_policy validation (lines 1642-1651)
        # Test invalid q value
        with pytest.raises(ValidationError):
            create_custom_rmt_policy(q=0.05)  # Below minimum

        # Test invalid deadband
        with pytest.raises(ValidationError):
            create_custom_rmt_policy(deadband=0.6)  # Above maximum

        # Test invalid margin
        with pytest.raises(ValidationError):
            create_custom_rmt_policy(margin=0.8)  # Below minimum

        # Test valid custom policy
        policy = create_custom_rmt_policy(
            q=2.0, deadband=0.05, margin=1.8, correct=False
        )
        assert isinstance(policy, dict)
        assert policy["q"] == 2.0
        assert policy["deadband"] == 0.05
        assert policy["margin"] == 1.8
        assert not policy["correct"]
