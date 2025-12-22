from __future__ import annotations

from types import SimpleNamespace

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

    guard._set_epsilon({"bad": "nope"})

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
    assert counts["router"] == 1
    assert counts["expert_ffn"] == 2
    assert counts["attn"] == 1
    assert counts["ffn"] == 1
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
    assert guard._collect_calibration_batches([1, 2, 3], 2) == [1, 2]


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
    assert guard._last_result["analysis_source"] == "activations"


def test_rmt_after_edit_activation_outliers_unavailable(monkeypatch):
    guard = RMTGuard()
    guard.prepared = True
    guard._require_activation = True
    guard._activation_ready = True
    guard._calibration_batches = [{"input_ids": [1, 2, 3]}]

    monkeypatch.setattr(
        guard,
        "_apply_rmt_detection_and_correction",
        lambda _model: {"correction_iterations": 1, "corrected_layers": 0},
    )
    monkeypatch.setattr(guard, "_compute_activation_outliers", lambda *_a, **_k: None)

    guard.after_edit(NoMaskModel())
    assert guard._activation_required_failed is True
    assert guard._activation_required_reason == "activation_outliers_unavailable"


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
