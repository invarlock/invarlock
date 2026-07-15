"""
Comprehensive Guard System Tests
===============================

Comprehensive tests for all guard modules to achieve 70% coverage.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from invarlock.core.exceptions import GuardError, ValidationError
from invarlock.guards.invariants import (
    assert_invariants,
    check_adapter_aware_invariants,
    check_all_invariants,
)
from invarlock.guards.policies import (
    create_custom_rmt_policy,
    create_custom_spectral_policy,
    create_custom_variance_policy,
    get_rmt_policy,
    get_spectral_policy,
    get_variance_policy,
)
from invarlock.guards.rmt_analysis import (
    capture_baseline_mp_stats,
    mp_bulk_edge,
)
from invarlock.guards.spectral_measurement import (
    compute_sigma_max,
    scan_model_gains,
)
from invarlock.guards.variance import VarianceGuard
from invarlock.guards.variance_scaling import equalise_residual_variance


class TestVarianceGuardComprehensive:
    """Comprehensive tests for VarianceGuard."""

    def setup_method(self):
        """Set up test fixtures."""
        from invarlock.guards.policies import get_variance_policy

        policy = get_variance_policy("balanced")
        self.guard = VarianceGuard(policy)
        self.model = self._create_gpt2_like_model()
        self.dataloader = self._create_mock_dataloader()

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

    def _create_mock_dataloader(self):
        """Create mock dataloader."""
        # Create tensor dataset
        data = torch.randint(0, 100, (10, 32))  # 10 batches, 32 sequence length
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(data)
        return DataLoader(dataset, batch_size=2)

    def test_guard_initialization(self):
        """Test guard initialization."""
        assert self.guard.name == "variance"
        assert isinstance(self.guard._policy, dict)
        assert "min_gain" in self.guard._policy
        assert "max_calib" in self.guard._policy
        assert "scope" in self.guard._policy
        assert self.guard._policy["mode"] in {"ci", "delta"}
        assert "min_rel_gain" in self.guard._policy
        assert "alpha" in self.guard._policy
        assert not self.guard._prepared
        assert not self.guard._enabled

    def test_prepare_method(self):
        """Test guard preparation."""
        mock_adapter = Mock()
        mock_calib = Mock()
        mock_calib.dataloader = self.dataloader
        policy = {"scope": "both", "min_gain": 0.2}

        result = self.guard.prepare(self.model, mock_adapter, mock_calib, policy)

        assert isinstance(result, dict)
        assert "ready" in result
        # Result may be True or False depending on whether target modules are found
        assert isinstance(result["ready"], bool)
        if result["ready"]:
            assert "baseline_metrics" in result
            assert self.guard._prepared
        assert self.guard._policy["scope"] == "both"
        assert self.guard._policy["min_gain"] == 0.2

    def test_resolve_target_modules(self):
        """Test _resolve_target_modules method."""
        modules = self.guard._resolve_target_modules(self.model)

        assert isinstance(modules, dict)
        # Should find modules based on scope
        if self.guard._policy["scope"] in ["both", "ffn"]:
            assert any("mlp.c_proj" in name for name in modules.keys())
        if self.guard._policy["scope"] in ["both", "attn"]:
            assert any("attn.c_proj" in name for name in modules.keys())

    def test_focus_modules_align_with_tap_patterns(self):
        """Target modules declared in policy should appear in resolved set."""
        from invarlock.guards.policies import get_variance_policy

        policy = get_variance_policy("balanced")
        target_modules = [
            "transformer.h.0.mlp.c_proj",
            "transformer.h.1.mlp.c_proj",
        ]
        policy.update(
            {
                "scope": "ffn",
                "tap": ["transformer.h.*.mlp.c_proj"],
                "target_modules": target_modules,
            }
        )
        guard = VarianceGuard(policy)
        model = self._create_gpt2_like_model()

        modules = guard._resolve_target_modules(model)

        assert set(target_modules).issubset(set(modules.keys()))
        # Focus modules should canonicalize targets
        assert guard._focus_modules == {
            "transformer.h.0.mlp.c_proj",
            "transformer.h.1.mlp.c_proj",
        }

    def test_enable_disable_methods(self):
        """Test enable and disable methods."""
        # Prepare first
        self.guard._prepared = True
        self.guard._scales = {"test_module": 0.9}
        self.guard._target_modules = {"test_module": nn.Linear(10, 5)}

        # Test enable
        result = self.guard.enable(self.model)
        assert isinstance(result, bool)

        # Test disable
        result = self.guard.disable(self.model)
        assert isinstance(result, bool)

    def test_set_ab_results(self):
        """Test set_ab_results method."""
        self.guard.set_ab_results(
            ppl_no_ve=3.5,
            ppl_with_ve=3.2,
            windows_used=50,
            seed_used=123,
            ratio_ci=(0.88, 0.94),
        )

        assert self.guard._ppl_no_ve == 3.5
        assert self.guard._ppl_with_ve == 3.2
        assert self.guard._ab_gain is not None
        assert self.guard._ab_windows_used == 50
        assert self.guard._ab_seed_used == 123
        assert self.guard._ratio_ci == (0.88, 0.94)

    def test_evaluate_ab_gate(self):
        """Test _evaluate_ab_gate method."""
        # Set up A/B results
        self.guard._policy["min_gain"] = 0.05
        self.guard.set_ab_results(
            3.5, 3.2, 50, 123, ratio_ci=(0.88, 0.94)
        )  # Good improvement with tight CI

        should_enable, reason = self.guard._evaluate_ab_gate()
        assert should_enable is True
        assert "criteria_met" in reason

        # Test with insufficient improvement
        self.guard.set_ab_results(
            3.5, 3.49, 50, 123, ratio_ci=(0.98, 1.02)
        )  # Tiny improvement, high CI
        should_enable, reason = self.guard._evaluate_ab_gate()
        assert not should_enable
        assert "min_rel_gain" in reason or "ci" in reason.lower()

    def test_policy_method(self):
        """Test policy method."""
        policy = self.guard.policy()

        assert isinstance(policy, dict)
        assert "min_gain" in policy
        assert "max_calib" in policy
        assert "scope" in policy
        assert "clamp" in policy
        assert "deadband" in policy
        assert "seed" in policy
        assert "mode" in policy
        assert "min_rel_gain" in policy
        assert "alpha" in policy

    def test_validate_sets_abort_on_errors(self):
        """validate() should request abort when finalize reports errors."""
        failure_payload = {
            "passed": False,
            "metrics": {},
            "errors": ["gate failure"],
            "warnings": [],
            "details": {"policy": self.guard._policy},
        }
        with patch.object(self.guard, "finalize", return_value=failure_payload):
            result = self.guard.validate(self.model, Mock(), {})

        assert result["passed"] is False
        assert result["decision"] == "block"
        assert result["violations"] == [
            {"type": "variance_error", "severity": "error", "message": "gate failure"}
        ]

    def test_validate_warns_when_monitor_only(self):
        """Monitor-only mode should downgrade aborts to warnings."""
        guard = VarianceGuard(self.guard._policy.copy())
        guard._monitor_only = True
        failure_payload = {
            "passed": False,
            "metrics": {},
            "errors": ["gate failure"],
            "warnings": [],
            "details": {"policy": guard._policy},
        }
        with patch.object(guard, "finalize", return_value=failure_payload):
            result = guard.validate(self.model, Mock(), {})

        assert result["decision"] == "monitor"
        assert result["passed"] is False


class TestUtilityFunctions:
    """Test utility functions from guard modules."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    def test_check_all_invariants(self):
        """Test check_all_invariants function."""
        outcome = check_all_invariants(self.model)

        assert hasattr(outcome, "name")
        assert hasattr(outcome, "passed")
        assert hasattr(outcome, "violations")
        assert hasattr(outcome, "metrics")
        assert isinstance(outcome.passed, bool)
        assert isinstance(outcome.violations, list)

    def test_assert_invariants(self):
        """Test assert_invariants function."""
        # Should not raise for a normal model
        assert_invariants(self.model)

        # Create a model with NaN parameters to test failure
        bad_model = nn.Linear(5, 2)
        bad_model.weight.data.fill_(float("nan"))

        with pytest.raises(AssertionError):
            assert_invariants(bad_model)

    def test_check_adapter_aware_invariants(self):
        """Test check_adapter_aware_invariants function."""
        passed, results = check_adapter_aware_invariants(self.model)

        assert isinstance(passed, bool)
        assert isinstance(results, dict)
        assert "adapter_type" in results
        assert "checks" in results
        assert "violations" in results

    def test_compute_sigma_max(self):
        """Test compute_sigma_max function."""
        linear_layer = nn.Linear(10, 5)
        sigma = compute_sigma_max(linear_layer)

        assert isinstance(sigma, float)
        assert sigma > 0

    def test_scan_model_gains(self):
        """Test scan_model_gains function."""
        # Create model with named modules
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.h = nn.ModuleList()

        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.c_fc = nn.Linear(64, 256)
        layer.mlp.c_proj = nn.Linear(256, 64)
        model.transformer.h.append(layer)

        gains = scan_model_gains(model)
        assert gains["scanned_modules"] == 2
        assert len(gains["spectral_norms"]) == 2
        assert set(gains["weight_statistics"]) == {
            "transformer.h.0.mlp.c_fc",
            "transformer.h.0.mlp.c_proj",
        }
        assert all(value > 0.0 for value in gains["spectral_norms"])

    def test_mp_bulk_edge(self):
        """Test mp_bulk_edge function."""
        edge = mp_bulk_edge(100, 50, whitened=False)

        assert isinstance(edge, float)
        assert edge > 0

        # Test whitened version
        edge_whitened = mp_bulk_edge(100, 50, whitened=True)
        assert isinstance(edge_whitened, float)
        assert edge_whitened > 0
        assert edge_whitened != edge

    def test_capture_baseline_mp_stats(self):
        """Test capture_baseline_mp_stats function."""
        # Create model with proper linear layers
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.h = nn.ModuleList()

        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.c_attn = nn.Linear(64, 192)
        layer.attn.c_proj = nn.Linear(64, 64)
        layer.mlp = nn.Module()
        layer.mlp.c_fc = nn.Linear(64, 256)
        layer.mlp.c_proj = nn.Linear(256, 64)
        model.transformer.h.append(layer)

        stats = capture_baseline_mp_stats(model)

        assert isinstance(stats, dict)
        # Should find some linear layers
        assert len(stats) > 0
        for _name, stat in stats.items():
            assert isinstance(stat, dict)
            assert "mp_bulk_edge_base" in stat
            assert "r_mp_base" in stat
            assert "sigma_base" in stat

    def test_equalise_residual_variance(self):
        """Test equalise_residual_variance function."""

        # Create transformer model with proper forward method
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList()

                layer = nn.Module()
                layer.attn = nn.Module()
                layer.attn.c_proj = nn.Linear(64, 64)
                layer.mlp = nn.Module()
                layer.mlp.c_proj = nn.Linear(256, 64)
                self.transformer.h.append(layer)

                self.embed = nn.Embedding(100, 64)

            def forward(self, input_ids):
                # Simple forward pass that uses the projection layers
                x = self.embed(input_ids)
                for layer in self.transformer.h:
                    # Simple attention-like operation
                    attn_out = layer.attn.c_proj(x)
                    # Simple MLP-like operation - create proper input tensor
                    mlp_in = torch.randn(x.size(0), x.size(1), 256, device=x.device)
                    mlp_out = layer.mlp.c_proj(mlp_in)
                    x = x + attn_out + mlp_out
                return x

        model = SimpleTransformer()

        # Create mock dataloader
        data = torch.randint(0, 99, (5, 32))  # Ensure indices are valid
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=1)
        attn_weight_before = model.transformer.h[0].attn.c_proj.weight.detach().clone()
        mlp_weight_before = model.transformer.h[0].mlp.c_proj.weight.detach().clone()

        scales = equalise_residual_variance(
            model, dataloader, windows=2, allow_empty=True
        )

        assert scales == {"block0.attn": 1.1, "block0.mlp": 1.1}
        assert torch.allclose(
            model.transformer.h[0].attn.c_proj.weight,
            attn_weight_before * 1.1,
        )
        assert torch.allclose(
            model.transformer.h[0].mlp.c_proj.weight,
            mlp_weight_before * 1.1,
        )


class TestPolicyFunctions:
    """Test policy utility functions."""

    def test_get_spectral_policy(self):
        """Test get_spectral_policy function."""
        policy = get_spectral_policy("balanced")

        assert isinstance(policy, dict)
        assert "sigma_quantile" in policy
        assert "deadband" in policy
        assert "scope" in policy
        assert policy.get("correction_enabled") is True

        # Test all available policies
        for name in ["conservative", "balanced", "aggressive", "attn_aware"]:
            policy = get_spectral_policy(name)
            assert isinstance(policy, dict)
            if name == "conservative":
                assert policy.get("correction_enabled") is True

        # Test invalid policy name
        from invarlock.core.exceptions import GuardError

        with pytest.raises(GuardError):
            get_spectral_policy("invalid")

    def test_create_custom_spectral_policy(self):
        """Test create_custom_spectral_policy function."""
        policy = create_custom_spectral_policy(
            sigma_quantile=0.90, deadband=0.05, scope="all"
        )

        assert isinstance(policy, dict)
        assert policy["sigma_quantile"] == 0.90
        assert policy["deadband"] == 0.05
        assert policy["scope"] == "all"
        assert "contraction" not in policy

        # Test validation
        with pytest.raises(ValidationError):
            create_custom_spectral_policy(sigma_quantile=1.5)  # Out of range

    def test_get_rmt_policy(self):
        """Test get_rmt_policy function."""
        policy = get_rmt_policy("balanced")

        assert isinstance(policy, dict)
        assert "q" in policy
        assert "deadband" in policy
        assert "margin" in policy
        assert "correct" in policy
        assert policy["correct"] is True

        # Test all available policies
        for name in ["conservative", "balanced", "aggressive"]:
            policy = get_rmt_policy(name)
            assert isinstance(policy, dict)
            if name == "conservative":
                assert policy["correct"] is True

        # Test invalid policy name
        with pytest.raises(GuardError):
            get_rmt_policy("invalid")

    def test_create_custom_rmt_policy(self):
        """Test create_custom_rmt_policy function."""
        policy = create_custom_rmt_policy(
            q=2.0, deadband=0.05, margin=2.0, correct=False
        )

        assert isinstance(policy, dict)
        assert policy["q"] == 2.0
        assert policy["deadband"] == 0.05
        assert policy["margin"] == 2.0
        assert not policy["correct"]

        # Test validation
        with pytest.raises(ValidationError):
            create_custom_rmt_policy(margin=0.5)  # Below minimum

    def test_get_variance_policy(self):
        """Test get_variance_policy function."""
        policy = get_variance_policy("balanced")

        assert isinstance(policy, dict)
        assert "min_gain" in policy
        assert "max_calib" in policy
        assert "scope" in policy
        assert "clamp" in policy
        assert "deadband" in policy
        assert "seed" in policy

        # Test all available policies
        for name in ["conservative", "balanced", "aggressive"]:
            policy = get_variance_policy(name)
            assert isinstance(policy, dict)

        # Test invalid policy name
        with pytest.raises(GuardError):
            get_variance_policy("invalid")

    def test_create_custom_variance_policy(self):
        """Test create_custom_variance_policy function."""
        policy = create_custom_variance_policy(
            min_gain=0.25,
            max_calib=150,
            scope="ffn",
            clamp=(0.8, 1.2),
            deadband=0.08,
            seed=456,
            mode="delta",
            min_rel_gain=0.02,
            alpha=0.1,
        )

        assert isinstance(policy, dict)
        assert policy["min_gain"] == 0.25
        assert policy["max_calib"] == 150
        assert policy["scope"] == "ffn"
        assert policy["clamp"] == (0.8, 1.2)
        assert policy["deadband"] == 0.08
        assert policy["seed"] == 456
        assert policy["mode"] == "delta"
        assert policy["min_rel_gain"] == 0.02
        assert policy["alpha"] == 0.1

        # Test validation
        with pytest.raises(ValidationError):
            create_custom_variance_policy(scope="invalid")  # Invalid scope
        with pytest.raises(ValidationError):
            create_custom_variance_policy(min_rel_gain=1.5)
