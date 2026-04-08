"""
Comprehensive Guard System Tests
===============================

Comprehensive tests for all guard modules to achieve 70% coverage.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from invarlock.guards.invariants import (
    InvariantsGuard,
)
from invarlock.guards.rmt import RMTGuard
from invarlock.guards.spectral import SpectralGuard
from invarlock.guards.spectral_measurement import (
    capture_baseline_sigmas,
)


class TestInvariantsGuardComprehensive:
    """Comprehensive tests for InvariantsGuard."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = InvariantsGuard()
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

        # Add transformer-like structure for more realistic testing
        self.transformer_model = nn.Module()
        self.transformer_model.transformer = nn.Module()
        self.transformer_model.transformer.wte = nn.Embedding(100, 64)
        self.transformer_model.lm_head = nn.Linear(64, 100)
        # Tie weights to test weight tying detection
        self.transformer_model.lm_head.weight = (
            self.transformer_model.transformer.wte.weight
        )

    def _create_bert_like_model(self, tied: bool = True) -> nn.Module:
        model = nn.Module()
        model.bert = nn.Module()
        model.bert.embeddings = nn.Module()
        model.bert.embeddings.word_embeddings = nn.Embedding(50, 32)
        model.cls = nn.Module()
        model.cls.predictions = nn.Module()
        model.cls.predictions.decoder = nn.Linear(32, 50, bias=False)
        if tied:
            model.cls.predictions.decoder.weight = (
                model.bert.embeddings.word_embeddings.weight
            )
        return model

    def _create_embed_tokens_model(self, tied: bool = True) -> nn.Module:
        model = nn.Module()
        model.model = nn.Module()
        model.model.embed_tokens = nn.Embedding(64, 32)
        model.lm_head = nn.Linear(32, 64, bias=False)
        if tied:
            model.lm_head.weight = model.model.embed_tokens.weight
        return model

    def test_guard_initialization(self):
        """Test guard initialization."""
        assert self.guard.name == "invariants"
        assert not self.guard.strict_mode
        assert self.guard.on_fail == "warn"
        assert not self.guard.prepared

        # Test custom initialization
        strict_guard = InvariantsGuard(strict_mode=True, on_fail="abort")
        assert strict_guard.strict_mode
        assert strict_guard.on_fail == "abort"

    def test_prepare_method(self):
        """Test guard preparation."""
        mock_adapter = Mock()
        mock_calib = Mock()
        policy = {"strict_mode": True}

        result = self.guard.prepare(self.model, mock_adapter, mock_calib, policy)

        assert isinstance(result, dict)
        assert "ready" in result
        assert result["ready"]
        assert "baseline_checks" in result
        assert self.guard.prepared
        assert len(self.guard.baseline_checks) > 0

    def test_before_edit_method(self):
        """Test before_edit hook."""
        # Should do nothing but not error
        result = self.guard.before_edit(self.model)
        assert result is None

    def test_after_edit_method(self):
        """Test after_edit hook."""
        # Should do nothing but not error
        result = self.guard.after_edit(self.model)
        assert result is None

    def test_finalize_method(self):
        """Test guard finalization."""
        # Prepare first
        self.guard.prepare(self.model, Mock(), Mock(), {})

        # Test finalize
        outcome = self.guard.finalize(self.model)

        assert hasattr(outcome, "name")
        assert outcome.name == "invariants"
        assert hasattr(outcome, "passed")
        assert isinstance(outcome.passed, bool)
        assert hasattr(outcome, "violations")
        assert isinstance(outcome.violations, list)
        assert hasattr(outcome, "metrics")
        assert isinstance(outcome.metrics, dict)

    def test_finalize_warn_only_violation(self):
        """Non-fatal invariant changes should emit warnings and still pass."""
        self.guard.prepare(self.model, Mock(), Mock(), {})
        current_checks = self.guard.baseline_checks.copy()
        current_checks["parameter_count"] = (
            self.guard.baseline_checks.get("parameter_count", 0) - 10
        )

        with patch.object(
            self.guard, "_capture_invariants", return_value=current_checks
        ):
            outcome = self.guard.finalize(self.model)

        assert outcome.passed is True
        assert outcome.action == "warn"
        assert outcome.metrics.get("warning_violations") == 1
        assert not outcome.metrics.get("fatal_violations")

    def test_finalize_fatal_violation_abort(self):
        """Fatal invariant violations should fail and request abort."""
        self.guard.prepare(self.model, Mock(), Mock(), {})

        with (
            patch.object(
                self.guard,
                "_capture_invariants",
                return_value=self.guard.baseline_checks,
            ),
            patch.object(
                self.guard, "_detect_non_finite", return_value=["parameter::w"]
            ),
        ):
            outcome = self.guard.finalize(self.model)

        assert outcome.passed is False
        assert outcome.action in {"abort", "rollback"}
        assert outcome.metrics.get("fatal_violations") == 1
        assert outcome.metrics.get("violations_found") == 1

    def test_finalize_without_prepare(self):
        """Test finalize when not prepared."""
        outcome = self.guard.finalize(self.model)

        assert not outcome.passed
        assert len(outcome.violations) > 0
        assert any(v.get("type") == "not_prepared" for v in outcome.violations)

    def test_capture_invariants_basic_model(self):
        """Test _capture_invariants on basic model."""
        invariants = self.guard._capture_invariants(self.model, None)

        assert isinstance(invariants, dict)
        assert "parameter_count" in invariants
        assert invariants["parameter_count"] > 0
        assert "structure_hash" in invariants

    def test_capture_invariants_transformer_model(self):
        """Test _capture_invariants on transformer model with weight tying."""
        invariants = self.guard._capture_invariants(self.transformer_model, None)

        assert isinstance(invariants, dict)
        assert "parameter_count" in invariants
        assert "weight_tying" in invariants
        assert invariants["weight_tying"]  # Should detect tied weights
        arch_flags = invariants.get("weight_tying_arches", {})
        assert arch_flags.get("gpt2") is True
        assert "structure_hash" in invariants

    def test_capture_invariants_bert_weight_tying(self):
        """Weight tying detection should cover BERT-style architectures."""
        bert_model = self._create_bert_like_model(tied=True)
        invariants = self.guard._capture_invariants(bert_model, None)

        assert invariants["weight_tying"] is True
        arch_flags = invariants.get("weight_tying_arches", {})
        assert arch_flags.get("bert") is True

        untied = self._create_bert_like_model(tied=False)
        untied_invariants = self.guard._capture_invariants(untied, None)
        arch_flags_untied = untied_invariants.get("weight_tying_arches", {})
        assert arch_flags_untied.get("bert") is False
        assert untied_invariants["weight_tying"] in {False, None}

    def test_capture_invariants_embed_tokens_weight_tying(self):
        """Weight tying detection should cover embed_tokens-style architectures."""
        embed_tokens_model = self._create_embed_tokens_model(tied=True)
        invariants = self.guard._capture_invariants(embed_tokens_model, None)

        assert invariants["weight_tying"] is True
        arch_flags = invariants.get("weight_tying_arches", {})
        assert arch_flags.get("embed_tokens") is True

        untied = self._create_embed_tokens_model(tied=False)
        untied_invariants = self.guard._capture_invariants(untied, None)
        arch_flags_untied = untied_invariants.get("weight_tying_arches", {})
        assert arch_flags_untied.get("embed_tokens") is False
        assert untied_invariants["weight_tying"] in {False, None}


class TestSpectralGuardComprehensive:
    """Comprehensive tests for SpectralGuard."""

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
        layer.attn.c_attn = nn.Linear(64, 192)  # 3 * 64 for Q, K, V
        layer.attn.c_proj = nn.Linear(64, 64)
        layer.mlp = nn.Module()
        layer.mlp.c_fc = nn.Linear(64, 256)
        layer.mlp.c_proj = nn.Linear(256, 64)

        model.transformer.h.append(layer)
        return model

    def test_guard_initialization(self):
        """Test guard initialization."""
        assert self.guard.name == "spectral"
        assert not self.guard.prepared
        assert hasattr(self.guard, "config")
        assert hasattr(self.guard, "baseline_metrics")
        assert isinstance(self.guard.baseline_metrics, dict)

        # Test custom initialization
        custom_guard = SpectralGuard(sigma_quantile=0.90, deadband=0.05, scope="all")
        assert custom_guard.config["sigma_quantile"] == 0.90
        assert "contraction" not in custom_guard.config
        assert custom_guard.config["deadband"] == 0.05
        assert custom_guard.config["scope"] == "all"

    def test_prepare_respects_correction_disabled(self):
        """Balanced policy should leave spectral guard in monitor-only mode."""
        mock_adapter = Mock()
        result = self.guard.prepare(
            self.model,
            mock_adapter,
            Mock(),
            {"correction_enabled": False},
        )

        assert isinstance(result, dict)
        assert self.guard.correction_enabled is False

    def test_validate_method(self):
        """Test guard validation method."""
        mock_adapter = Mock()
        context = {"baseline_metrics": {}}

        result = self.guard.validate(self.model, mock_adapter, context)

        assert isinstance(result, dict)
        assert "passed" in result
        assert "decision" in result
        assert "metrics" in result
        assert "diagnostics" in result
        assert isinstance(result["passed"], bool)

    def test_validate_aborts_when_caps_exceeded(self):
        """Spectral guard should abort when cap count exceeds configured limit."""
        guard = SpectralGuard(max_caps=0)
        guard.prepared = True
        guard.baseline_sigmas = {}
        guard.baseline_family_stats = {}
        guard.module_family_map = {}
        guard.latest_z_scores = {}
        guard.target_sigma = 1.0
        guard.family_caps = {}

        with (
            patch(
                "invarlock.guards.spectral_measurement.capture_baseline_sigmas",
                return_value={},
            ),
            patch.object(
                SpectralGuard,
                "_detect_spectral_violations",
                return_value=[
                    {
                        "type": "family_z_cap",
                        "module": "transformer.h.0.mlp.c_fc",
                        "family": "ffn",
                        "z_score": 3.0,
                        "kappa": 2.0,
                    }
                ],
            ),
        ):
            result = guard.validate(self.model, Mock(), {})

        assert result["decision"] == "block"
        assert result["metrics"]["caps_exceeded"] is True
        assert result["metrics"]["max_caps"] == 0

    def test_validate_with_error(self):
        """Test guard validation with error handling."""
        mock_adapter = Mock()

        with pytest.raises(AttributeError):
            self.guard.validate(None, mock_adapter, {})

    def test_config_storage(self):
        """Test that configuration is properly stored."""
        test_config = {"sigma_quantile": 0.85, "scope": "test"}
        guard_with_config = SpectralGuard(**test_config)

        assert guard_with_config.config["sigma_quantile"] == 0.85
        assert guard_with_config.config["scope"] == "test"
        assert guard_with_config.config["sigma_quantile"] == 0.85
        assert guard_with_config.config["scope"] == "test"

    def test_max_spectral_norm_default_is_disabled(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp_proj = nn.Linear(8, 8, bias=False)

        model = TinyModel()
        guard = SpectralGuard()
        assert guard.max_spectral_norm is None
        policy = {
            "scope": "all",
            "family_caps": {
                "ffn": {"kappa": 100.0},
                "attn": {"kappa": 100.0},
                "embed": {"kappa": 100.0},
                "other": {"kappa": 100.0},
            },
            "ignore_preview_inflation": False,
        }

        guard.prepare(model, Mock(), None, policy)

        with torch.no_grad():
            model.mlp_proj.weight.mul_(50.0)

        result = guard.validate(model, Mock(), {})
        assert all(
            violation["type"] != "max_spectral_norm"
            for violation in result["violations"]
        )

    def test_absolute_cap_disabled_when_none(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp_proj = nn.Linear(8, 8, bias=False)

        model = TinyModel()
        guard = SpectralGuard(max_spectral_norm=None)
        policy = {
            "scope": "all",
            "family_caps": {
                "ffn": {"kappa": 100.0},
                "attn": {"kappa": 100.0},
                "embed": {"kappa": 100.0},
                "other": {"kappa": 100.0},
            },
            "ignore_preview_inflation": False,
        }

        guard.prepare(model, Mock(), None, policy)

        with torch.no_grad():
            model.mlp_proj.weight.mul_(50.0)

        result = guard.validate(model, Mock(), {})
        assert all(
            violation["type"] != "max_spectral_norm"
            for violation in result["violations"]
        )

    def test_policy_serialized_in_finalize(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp_proj = nn.Linear(8, 8, bias=False)

        model = TinyModel()
        guard = SpectralGuard(
            sigma_quantile=0.93,
            deadband=0.12,
            scope="all",
            max_caps=7,
            max_spectral_norm=None,
            family_caps={
                "ffn": {"kappa": 2.7},
                "attn": {"kappa": 3.1},
            },
            multiple_testing={"method": "bh", "alpha": 0.04, "m": 4},
            correction_enabled=False,
            ignore_preview_inflation=False,
        )
        guard.prepare(model, Mock(), None, {})
        finalize = guard.finalize(model)
        assert finalize["policy"]["scope"] == "all"
        assert finalize["policy"]["sigma_quantile"] == pytest.approx(0.93)
        assert finalize["policy"]["deadband"] == pytest.approx(0.12)
        assert finalize["policy"]["max_caps"] == 7
        assert "family_caps" in finalize["policy"]
        assert finalize["policy"]["multiple_testing"]["method"] == "bh"

    def test_family_caps_zscore(self):
        """Ensure per-family z-score caps isolate violations."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_proj = nn.Linear(8, 8, bias=False)
                self.mlp_proj = nn.Linear(8, 8, bias=False)

            def forward(self, x):  # pragma: no cover - not used
                return self.mlp_proj(self.attn_proj(x))

        model = TinyModel()
        guard = SpectralGuard()
        policy = {
            "scope": "all",
            "family_caps": {
                "attn": {"kappa": 1.0},
                "ffn": {"kappa": 5.0},
                "embed": {"kappa": 5.0},
                "other": {"kappa": 5.0},
            },
            "ignore_preview_inflation": False,
        }

        guard.prepare(model, Mock(), None, policy)
        baseline = guard.validate(model, Mock(), {})
        assert baseline["passed"]

        with torch.no_grad():
            model.attn_proj.weight.mul_(6.0)

        result = guard.validate(model, Mock(), {})
        assert result["passed"] is True
        assert result["decision"] == "monitor"
        families = {violation.get("family") for violation in result["violations"]}
        assert "attn" in families
        assert "ffn" not in families

    def test_ignore_preview_inflation_masks_after_edit(self):
        """Preview-phase violations are ignored when ignore_preview_inflation is true."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_proj = nn.Linear(8, 8, bias=False)
                self.mlp_proj = nn.Linear(8, 8, bias=False)

        model = TinyModel()
        policy = {
            "family_caps": {
                "attn": {"kappa": 0.5},
                "ffn": {"kappa": 0.5},
                "embed": {"kappa": 0.5},
                "other": {"kappa": 0.5},
            },
            "ignore_preview_inflation": True,
        }

        guard = SpectralGuard()
        guard.prepare(model, Mock(), None, policy)

        with torch.no_grad():
            model.attn_proj.weight.mul_(10.0)
        inflated_metrics = capture_baseline_sigmas(model)

        preview_violations = guard._detect_spectral_violations(
            model, inflated_metrics, phase="after_edit"
        )
        assert preview_violations == []

        finalize_violations = guard._detect_spectral_violations(
            model, inflated_metrics, phase="finalize"
        )
        assert finalize_violations


class TestRMTGuardComprehensive:
    """Comprehensive tests for RMTGuard."""

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

    def test_guard_initialization(self):
        """Test guard initialization."""
        assert self.guard.name == "rmt"
        assert self.guard.q == "auto"
        assert self.guard.deadband == 0.10
        assert self.guard.margin == 1.5
        assert self.guard.correct
        assert not self.guard.prepared

        # Test custom initialization
        custom_guard = RMTGuard(q=2.0, deadband=0.05, margin=2.0, correct=False)
        assert custom_guard.q == 2.0
        assert custom_guard.deadband == 0.05
        assert custom_guard.margin == 2.0
        assert not custom_guard.correct

    def test_prepare_method(self):
        """Test guard preparation."""
        mock_adapter = Mock()
        calib = [{"input_ids": torch.randint(0, 100, (1, 64))} for _ in range(3)]
        policy = {
            "deadband": 0.05,
            "correct": True,
            "estimator": {"iters": 2, "init": "ones"},
            "activation": {
                "sampling": {"windows": {"count": 2, "indices_policy": "first"}}
            },
        }

        result = self.guard.prepare(self.model, mock_adapter, calib, policy)

        assert isinstance(result, dict)
        assert "ready" in result
        assert result["ready"]
        assert "baseline_metrics" in result
        assert self.guard.prepared
        assert isinstance(self.guard.baseline_edge_risk_by_family, dict)

        # Check that policy was applied
        assert self.guard.deadband == 0.05
        assert self.guard.correct is True
        assert self.guard.estimator["iters"] == 2
        assert self.guard.activation_sampling["windows"]["indices_policy"] == "first"

    def test_prepare_respects_correction_flag(self):
        """Balanced policy should disable automatic correction."""
        mock_adapter = Mock()
        result = self.guard.prepare(
            self.model,
            mock_adapter,
            [{"input_ids": torch.randint(0, 100, (1, 64))}],
            {"correct": False},
        )

        assert isinstance(result, dict)
        assert self.guard.correct is False

    def test_epsilon_rule_enforced_per_family(self):
        """Finalize flags epsilon-rule violations when edge-risk exceeds allowance."""

        guard = RMTGuard(
            epsilon_default=0.0,
            epsilon_by_family={"attn": 0.0, "ffn": 0.0, "embed": 0.0, "other": 0.0},
        )
        policy = {
            "epsilon_default": 0.0,
            "epsilon_by_family": {
                "attn": 0.0,
                "ffn": 0.0,
                "embed": 0.0,
                "other": 0.0,
            },
        }
        guard.prepare(
            self.model,
            Mock(),
            [{"input_ids": torch.randint(0, 100, (1, 64))}],
            policy,
        )
        base = float(guard.baseline_edge_risk_by_family.get("attn", 0.0) or 0.0)
        assert base > 0.0
        guard.edge_risk_by_family = {
            **guard.baseline_edge_risk_by_family,
            "attn": base * 2.0,
        }

        outcome = guard.finalize(self.model)
        metrics = outcome.metrics if hasattr(outcome, "metrics") else outcome["metrics"]
        passed = outcome.passed if hasattr(outcome, "passed") else outcome["passed"]

        assert metrics["epsilon_violations"], (
            "Expected epsilon violations to be recorded"
        )
        assert any(
            failure["family"] == "attn" for failure in metrics["epsilon_violations"]
        )
        assert passed is False

    def test_get_linear_modules(self):
        """Test _get_linear_modules method."""
        modules = self.guard._get_linear_modules(self.model)

        assert isinstance(modules, list)
        assert len(modules) > 0

        # Check that module names match expected patterns
        module_names = [name for name, _ in modules]
        linear_suffixes = [".attn.c_attn", ".attn.c_proj", ".mlp.c_fc", ".mlp.c_proj"]
        for name in module_names:
            assert any(name.endswith(suffix) for suffix in linear_suffixes)

    def test_policy_method(self):
        """Test policy method."""
        policy = self.guard.policy()

        assert isinstance(policy, dict)
        assert "q" in policy
        assert "deadband" in policy
        assert "margin" in policy
        assert "correct" in policy
