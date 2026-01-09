from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from invarlock.guards.rmt import RMTGuard


class ActivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.norm = nn.LayerNorm(4)
        self.attn = nn.Linear(4, 4, bias=False)
        self.linear = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask=None):
        x = input_ids.float()
        return self.attn(x)


class NoMaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(3, 3, bias=False)

    def forward(self, input_ids):
        return self.attn(input_ids.float())


class DenseOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(2, 2, bias=False)


def test_rmt_context_and_family_counts():
    guard = RMTGuard()
    report = SimpleNamespace(context={"profile": "ci", "auto": {"tier": "balanced"}})
    guard.set_run_context(report)
    assert guard._require_activation is True

    report = SimpleNamespace(context=["not", "dict"])
    guard.set_run_context(report)
    assert guard._run_profile is None

    guard._set_epsilon_by_family({"bad": "nope"})

    per_layer = [
        {"module_name": "router.gate", "outlier_count": None, "has_outlier": True},
        {"module_name": "experts.block", "outlier_count": 2},
        {"module_name": "block.attn.c_proj", "outlier_count": 1},
        {"module_name": "mlp.c_fc", "outlier_count": 1},
        {"module_name": "wte", "outlier_count": 1},
        {"module_name": "misc", "outlier_count": "bad"},
        {"module_name": "misc2", "outlier_count": -1},
        {"module_name": "misc3", "outlier_count": None, "has_outlier": False},
    ]
    counts = guard._count_outliers_per_family(per_layer)
    assert counts["ffn"] == 4
    assert counts["attn"] == 1
    assert counts["embed"] == 1


def test_rmt_prepare_activation_inputs_and_token_weight():
    guard = RMTGuard()
    device = torch.device("cpu")

    ids, mask = guard._prepare_activation_inputs(
        {"input_ids": [1, 2], "attention_mask": [1, 1]}, device
    )
    assert ids.shape == (1, 2) and mask.shape == (1, 2)

    ids, mask = guard._prepare_activation_inputs(
        (torch.tensor([[1, 2]]), torch.tensor([[1, 1]])), device
    )
    assert ids.shape == (1, 2) and mask.shape == (1, 2)

    ids, mask = guard._prepare_activation_inputs(5, device)
    assert ids.shape == () and mask is None

    ids, mask = guard._prepare_activation_inputs({"input_ids": None}, device)
    assert ids is None and mask is None

    ids, _ = guard._prepare_activation_inputs({"input_ids": [1, 2]}, object())
    assert ids is not None

    ids = torch.zeros((1, 2))
    mask = torch.ones((1, 2))
    assert guard._batch_token_weight(ids, mask) == 2
    mask_zero = torch.zeros((1, 2))
    assert guard._batch_token_weight(ids, mask_zero) == 2
    assert guard._batch_token_weight(ids, None) == 2


def test_rmt_get_activation_modules_and_collect_batches():
    guard = RMTGuard()
    model = ActivationModel()

    modules = guard._get_activation_modules(model)
    names = {name for name, _module in modules}
    assert "embed" in names
    assert "norm" in names
    assert "attn" in names
    assert "linear" not in names

    assert guard._collect_calibration_batches(None, 1) == []
    assert guard._collect_calibration_batches([1, 2, 3], 2) == [1, 3]


def test_rmt_activation_svd_outliers_cases():
    guard = RMTGuard()

    assert guard._activation_svd_outliers([1, 2, 3], margin=1.0, deadband=0.0) == (
        0,
        0.0,
        0.0,
    )
    assert guard._activation_svd_outliers(
        torch.tensor([1, 2, 3]), margin=1.0, deadband=0.0
    ) == (0, 0.0, 0.0)
    assert guard._activation_svd_outliers(
        torch.empty((0, 3)), margin=1.0, deadband=0.0
    ) == (0, 0.0, 0.0)
    assert guard._activation_svd_outliers(
        torch.tensor([[float("nan"), 0.0]]), margin=1.0, deadband=0.0
    ) == (0, 0.0, 0.0)
    assert guard._activation_svd_outliers(
        torch.zeros((2, 2)), margin=1.0, deadband=0.0
    ) == (0, 0.0, 0.0)

    outliers, max_ratio, sigma_max = guard._activation_svd_outliers(
        torch.randn(2, 2), margin=1.0, deadband=0.0
    )
    assert isinstance(outliers, int)
    assert max_ratio >= 0.0 and sigma_max >= 0.0


def test_rmt_compute_activation_outliers_empty_paths():
    guard = RMTGuard()
    assert guard._compute_activation_outliers(DenseOnlyModel(), []) is None
    assert (
        guard._compute_activation_outliers(DenseOnlyModel(), [{"input_ids": None}])
        is None
    )


def test_rmt_compute_activation_outliers_branches():
    guard = RMTGuard()
    model = NoMaskModel()
    model.train()
    batches = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": torch.tensor([[1, 2, 3]])},
        {"input_ids": None},
    ]

    guard.margin = 0.0
    out = guard._compute_activation_outliers(model, batches)
    assert out and out["token_weight_total"] > 0
    assert model.training is True

    guard.margin = 1e6
    out = guard._compute_activation_outliers(model, batches)
    assert out and out["outlier_count"] >= 0


def test_rmt_after_edit_activation_required_missing():
    guard = RMTGuard()
    guard.prepared = True
    guard._require_activation = True
    guard._activation_ready = False
    guard._calibration_batches = []

    guard.after_edit(NoMaskModel())
    assert guard._activation_required_failed is True
    assert guard._last_result["analysis_source"] == "activations_edge_risk"


def test_rmt_after_edit_activation_outliers_unavailable(monkeypatch):
    guard = RMTGuard()
    guard.prepared = True
    guard._require_activation = True
    guard._activation_ready = True
    guard._calibration_batches = [{"input_ids": [1, 2, 3]}]

    monkeypatch.setattr(guard, "_compute_activation_edge_risk", lambda *_a, **_k: None)

    guard.after_edit(NoMaskModel())
    assert guard._activation_required_failed is True
    assert guard._activation_required_reason == "activation_edge_risk_unavailable"


def test_rmt_finalize_activation_required_failure():
    guard = RMTGuard()
    guard.prepared = True
    guard._require_activation = True
    guard._activation_required_failed = True
    guard._activation_required_reason = "activation_required"
    guard._activation_ready = False

    result = guard.finalize(NoMaskModel())
    passed = result.passed if hasattr(result, "passed") else result.get("passed")
    assert passed is False


def test_rmt_activation_edge_risk_rejects_invalid_inputs() -> None:
    guard = RMTGuard()

    assert guard._activation_edge_risk([1, 2, 3]) is None
    assert guard._activation_edge_risk(torch.tensor([1.0, 2.0, 3.0])) is None
    assert guard._activation_edge_risk(torch.empty((0, 3))) is None
    assert guard._activation_edge_risk(torch.tensor([[float("nan"), 0.0]])) is None

    # tuple/list input should unwrap the first element
    out = guard._activation_edge_risk(
        (torch.arange(12, dtype=torch.float32).reshape(3, 4),)
    )
    assert out is not None
    risk, sigma, mp_edge = out
    assert all(
        isinstance(x, float) and math.isfinite(x) for x in (risk, sigma, mp_edge)
    )
    assert risk > 0.0 and sigma > 0.0 and mp_edge > 0.0


def test_rmt_activation_edge_risk_is_scale_invariant() -> None:
    guard = RMTGuard()
    guard.estimator = {"type": "power_iter", "iters": 2, "init": "ones"}
    base = torch.arange(1, 1 + 16 * 8, dtype=torch.float32).reshape(16, 8)
    out1 = guard._activation_edge_risk(base)
    out2 = guard._activation_edge_risk(base * 3.0)
    assert out1 is not None and out2 is not None

    risk1, sigma1, mp1 = out1
    risk2, sigma2, mp2 = out2
    assert mp1 == pytest.approx(mp2, rel=0.0, abs=0.0)
    assert sigma1 == pytest.approx(sigma2, rel=1e-3, abs=1e-6)
    assert risk1 == pytest.approx(risk2, rel=1e-3, abs=1e-6)


def test_rmt_compute_activation_edge_risk_smoke_and_token_weight() -> None:
    guard = RMTGuard()
    guard.estimator = {"type": "power_iter", "iters": 2, "init": "ones"}
    model = NoMaskModel()
    out = guard._compute_activation_edge_risk(
        model,
        batches=[
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [1, 2, 3]},
        ],
    )
    assert out is not None
    assert out["analysis_source"] == "activations_edge_risk"
    assert out["batches_used"] == 2
    assert out["token_weight_total"] == 6
    assert "attn" in out["edge_risk_by_module"]
    assert out["edge_risk_by_family"]["attn"] == pytest.approx(
        out["edge_risk_by_module"]["attn"]
    )
    assert out["edge_risk_by_family"]["attn"] > 0.0
    assert set(out["edge_risk_by_family"].keys()) == {"attn", "ffn", "embed", "other"}


def test_rmt_policy_property_round_trips_values() -> None:
    guard = RMTGuard(epsilon_default=0.08, epsilon_by_family={"attn": 0.05})
    policy = guard.policy()
    assert policy["epsilon_default"] == pytest.approx(0.08)
    assert policy["epsilon_by_family"]["attn"] == pytest.approx(0.05)
    assert policy["margin"] == pytest.approx(guard.margin)
    assert policy["deadband"] == pytest.approx(guard.deadband)


def test_rmt_finalize_enforces_epsilon_band_and_action() -> None:
    guard = RMTGuard(epsilon_default=0.10, epsilon_by_family={"attn": 0.10})
    guard.prepared = True

    # Baseline/current set directly so finalize does not re-run activation scoring.
    guard.baseline_edge_risk_by_family = {"attn": 10.0}
    guard.edge_risk_by_family = {"attn": 11.1}  # allowed is 11.0

    result = guard.finalize(NoMaskModel())
    passed = result.passed if hasattr(result, "passed") else result.get("passed")
    action = result.action if hasattr(result, "action") else result.get("action")
    metrics = (
        result.metrics if hasattr(result, "metrics") else result.get("metrics", {})
    )

    assert passed is False
    assert action in {"abort", "warn"}
    assert metrics["stable"] is False
    assert (
        metrics["epsilon_violations"]
        and metrics["epsilon_violations"][0]["family"] == "attn"
    )
    v0 = metrics["epsilon_violations"][0]
    assert v0["edge_base"] == pytest.approx(10.0)
    assert v0["edge_cur"] == pytest.approx(11.1)
    assert v0["allowed"] == pytest.approx(11.0)
    assert v0["delta"] == pytest.approx(0.11, rel=1e-6)
    assert v0["epsilon"] == pytest.approx(0.10)


def test_rmt_epsilon_band_boundary_allows_equal_threshold() -> None:
    guard = RMTGuard(epsilon_default=0.10, epsilon_by_family={"attn": 0.10})
    guard.baseline_edge_risk_by_family = {"attn": 10.0}
    guard.edge_risk_by_family = {"attn": 11.0}  # allowed is 11.0
    assert guard._compute_epsilon_violations() == []


def test_rmt_set_epsilon_default_and_family_bounds() -> None:
    guard = RMTGuard(epsilon_default=0.10)

    guard._set_epsilon_default(0.0)
    assert guard.epsilon_default == pytest.approx(0.0)
    guard._set_epsilon_default(float("inf"))
    assert guard.epsilon_default == pytest.approx(0.0)
    guard._set_epsilon_default(-0.1)
    assert guard.epsilon_default == pytest.approx(0.0)
    guard._set_epsilon_default("0.2")
    assert guard.epsilon_default == pytest.approx(0.2)

    guard.epsilon_by_family = {}
    guard._set_epsilon_by_family(
        {
            "attn": "0.1",
            "bad": "nope",
            "neg": -0.1,
            "inf": float("inf"),
        }
    )
    assert set(guard.epsilon_by_family.keys()) == {"attn"}
    assert guard.epsilon_by_family.get("attn") == pytest.approx(0.1)
