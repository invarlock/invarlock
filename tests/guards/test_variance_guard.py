from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from invarlock.guards.policies import create_custom_variance_policy
from invarlock.guards.variance import (
    VarianceGuard,
    _predictive_gate_outcome,
    equalise_residual_variance,
)

# Combined from: test_variance_guard_ci.py, test_variance_guard_extra.py, test_variance_guard_gate.py


class DummyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(x)


class DummyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(x)


class DummyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = DummyAttention()
        self.mlp = DummyMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = nn.ModuleList([DummyBlock()])


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = DummyTransformer()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*input_ids.shape, 4), dtype=torch.float32)


def _make_policy(**overrides):
    base = {
        "min_gain": 0.0,
        "min_rel_gain": 0.0,
        "max_calib": 200,
        "scope": "both",
        "clamp": (0.5, 2.0),
        "deadband": 0.1,
        "seed": 123,
        "mode": "ci",
        "alpha": 0.05,
        "tie_breaker_deadband": 0.001,
        "min_effect_lognll": 0.001,
        "calibration": {"windows": 8, "min_coverage": 6, "seed": 123},
    }
    base.update(overrides)
    return base


def test_equalise_residual_variance_allows_empty_dataloader() -> None:
    model = DummyModel()
    calibrations = []
    scales = equalise_residual_variance(
        model, calibrations, allow_empty=True, windows=1, tol=0.05
    )
    assert scales == {}


def test_variance_guard_ci_gate_enables_on_small_gain() -> None:
    hidden_size = 8

    class TinyAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c_attn = nn.Linear(hidden_size, hidden_size)
            self.c_proj = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.c_proj(torch.tanh(self.c_attn(x)))

    class TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c_fc = nn.Linear(hidden_size, hidden_size * 2)
            self.c_proj = nn.Linear(hidden_size * 2, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.c_proj(torch.relu(self.c_fc(x)))

    class TinyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = TinyAttention()
            self.mlp = TinyMLP()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(x)
            x = x + self.mlp(x)
            return x

    class TinyTransformer(nn.Module):
        def __init__(self, n_layer: int = 2) -> None:
            super().__init__()
            self.h = nn.ModuleList(TinyBlock() for _ in range(n_layer))

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = TinyTransformer()
            self.config = type("Cfg", (), {"n_positions": 32, "vocab_size": 128})

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            x = inputs.float()
            for block in self.transformer.h:
                x = block(x)
            return x

    model = TinyModel()
    dataloader = [torch.randn(4, hidden_size) for _ in range(8)]

    policy = create_custom_variance_policy(
        min_gain=0.01,
        max_calib=80,
        scope="both",
        clamp=(0.8, 1.25),
        deadband=0.0,
        seed=321,
        mode="ci",
        min_rel_gain=0.001,
        alpha=0.05,
    )

    guard = VarianceGuard(policy)
    calib = type("Calib", (), {"dataloader": dataloader})()

    prepare_result = guard.prepare(model, adapter=None, calib=calib)
    assert prepare_result["ready"], prepare_result

    assert guard.enable(model)

    guard.set_ab_results(
        ppl_no_ve=52.0,
        ppl_with_ve=51.0,
        windows_used=20,
        seed_used=policy["seed"],
        ratio_ci=(0.96, 0.99),
    )

    finalize = guard.finalize(model)
    assert finalize["passed"], finalize
    metrics = finalize["metrics"]
    assert metrics["ve_enabled"] is True
    assert metrics["ratio_ci"] is not None
    assert metrics["ratio_ci"][1] <= 0.999
    assert "calibration" in metrics
    assert metrics["calibration"]["coverage"] >= 0


@pytest.mark.parametrize(
    ("mean_delta", "delta_ci", "min_effect", "one_sided", "expected"),
    [
        (-0.002, (-0.003, -0.001), 0.001, True, (True, "ci_gain_met")),
        (-0.0005, (-0.002, 0.0001), 0.001, True, (False, "gain_below_threshold")),
        (-0.0005, (-0.0015, -0.0002), 0.001, True, (False, "gain_below_threshold")),
        (0.0003, (-0.001, 0.0004), 0.0, True, (False, "mean_not_negative")),
        (-0.002, (-0.0035, -0.0012), 0.001, False, (True, "ci_gain_met")),
        (-0.0008, (-0.0012, -0.0003), 0.001, False, (False, "gain_below_threshold")),
        (-0.0010, (-0.0015, 0.0002), 0.001, False, (False, "ci_contains_zero")),
    ],
)
def test_predictive_gate_outcome(mean_delta, delta_ci, min_effect, one_sided, expected):
    assert (
        _predictive_gate_outcome(mean_delta, delta_ci, min_effect, one_sided)
        == expected
    )


def test_variance_guard_focus_modules_filters_targets():
    guard = VarianceGuard(
        policy={"target_modules": ["transformer.h.0.mlp.c_proj"], "scope": "both"}
    )
    model = DummyModel()
    adapter = Mock()
    adapter.get_layer_modules.return_value = {}

    targets = guard._resolve_target_modules(model, adapter=adapter)

    assert targets
    assert all(name.endswith("mlp.c_proj") for name in targets)


def test_variance_guard_predictive_gate_state_exported_in_stats():
    guard = VarianceGuard(
        policy={
            "predictive_one_sided": True,
            "min_effect_lognll": 0.001,
            "scope": "both",
        }
    )
    guard._predictive_gate_state.update(
        {
            "evaluated": True,
            "passed": True,
            "reason": "ci_gain_met",
            "delta_ci": (-0.002, -0.001),
            "gain_ci": (0.001, 0.003),
            "mean_delta": -0.0015,
        }
    )
    guard._stats["predictive_gate"] = guard._predictive_gate_state.copy()

    gate_metrics = guard._stats["predictive_gate"]
    assert gate_metrics["evaluated"] is True
    assert gate_metrics["passed"] is True
    assert gate_metrics["reason"] == "ci_gain_met"
    assert gate_metrics["delta_ci"] == (-0.002, -0.001)


def test_variance_guard_checkpoint_cycle():
    guard = VarianceGuard(policy={"scope": "both"})
    model = DummyModel()
    guard._target_modules = {
        "transformer.h.0.mlp.c_proj": model.transformer.h[0].mlp.c_proj,
    }

    guard._push_checkpoint(model)
    # mutate weights
    guard._target_modules["transformer.h.0.mlp.c_proj"].weight.data.add_(1.0)
    restored = guard._pop_checkpoint(model)
    assert restored is True
    guard._commit_checkpoint()


@pytest.mark.parametrize(
    "setup,expected_reason",
    [
        ({}, "no_ab_results"),
        (
            {
                "_ab_gain": 0.01,
                "_ppl_no_ve": 100.0,
                "_ppl_with_ve": 99.0,
                "_ratio_ci": (0.95, 0.99),
            },
            "below_threshold_with_deadband",
        ),
        (
            {
                "_ab_gain": 0.02,
                "_ppl_no_ve": 100.0,
                "_ppl_with_ve": 97.0,
                "_ratio_ci": (0.95, 1.02),
                "_policy": {
                    "mode": "ci",
                    "min_gain": 0.0,
                    "min_rel_gain": 0.0,
                    "scope": "both",
                },
            },
            "ci_interval_too_high",
        ),
    ],
)
def test_variance_guard_evaluate_ab_gate_failures(setup, expected_reason):
    guard = VarianceGuard(policy={"scope": "both"})
    for key, value in setup.items():
        setattr(guard, key, value)
    guard._policy.update(
        {
            "min_gain": 0.01,
            "min_rel_gain": 0.01,
            "mode": "ci",
        }
    )
    decision, reason = guard._evaluate_ab_gate()
    assert decision is False
    assert expected_reason in reason


def test_variance_guard_evaluate_ab_gate_success():
    guard = VarianceGuard(
        policy={"scope": "both", "min_gain": 0.01, "min_rel_gain": 0.01}
    )
    guard._ab_gain = 0.02
    guard._ppl_no_ve = 100.0
    guard._ppl_with_ve = 98.0
    guard._ratio_ci = (0.90, 0.95)
    guard._predictive_gate_state.update(
        {"evaluated": True, "passed": True, "reason": "ci_gain_met"}
    )

    decision, reason = guard._evaluate_ab_gate()
    assert decision is True
    assert "criteria_met" in reason


def test_variance_guard_enables_when_log_effect_exceeds_threshold():
    guard = VarianceGuard(policy=_make_policy())
    guard._ab_gain = 0.002  # 0.2% relative improvement
    guard._ppl_no_ve = 50.0
    guard._ppl_with_ve = 49.94  # ~0.0012 log improvement
    guard._ratio_ci = (0.90, 0.995)

    should_enable, reason = guard._evaluate_ab_gate()
    assert should_enable, reason


def test_variance_guard_respects_conservative_tie_breaker():
    conservative_policy = _make_policy(
        min_gain=0.01,
        min_rel_gain=0.0075,
        tie_breaker_deadband=0.005,
        min_effect_lognll=0.002,
    )
    guard = VarianceGuard(policy=conservative_policy)
    guard._ab_gain = 0.002  # improvement too small for conservative policy
    guard._ppl_no_ve = 50.0
    guard._ppl_with_ve = 49.94
    guard._ratio_ci = (0.92, 0.997)

    should_enable, reason = guard._evaluate_ab_gate()
    assert not should_enable
    assert "below_threshold" in reason or "below_min" in reason
