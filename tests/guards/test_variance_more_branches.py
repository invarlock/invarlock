import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard, equalise_residual_variance


class TinyBlock(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.c_proj = nn.Linear(d, d, bias=False)


class TinyModel(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock(d)])

    def forward(self, x):
        # Trigger both projection hooks
        y1 = self.transformer.h[0].attn.c_proj(x)
        y2 = self.transformer.h[0].mlp.c_proj(x)
        return y1 + y2


def test_ab_gate_predictive_gate_failed_branch():
    g = VarianceGuard(
        policy={
            "mode": "ci",
            "min_gain": 0.0,
            "min_rel_gain": 0.0,
            "predictive_gate": True,
        }
    )
    # Valid A/B metrics and CI, but predictive gate says failed
    g._ab_gain = 0.1
    g._ppl_no_ve = 100.0
    g._ppl_with_ve = 90.0
    g._ratio_ci = (0.7, 0.9)
    g._predictive_gate_state.update(
        {"evaluated": True, "passed": False, "reason": "ci_contains_zero"}
    )
    ok, reason = g._evaluate_ab_gate()
    assert ok is False and reason.startswith("predictive_gate_failed")


def test_equalise_residual_variance_clamp_none_and_tol_skip():
    model = TinyModel()
    # Four small random batches
    batches = [torch.randn(2, 4) for _ in range(4)]
    # With very large tol, scaling should be skipped; clamp_range=None to cover that branch
    scales = equalise_residual_variance(
        model, batches, windows=4, tol=2.0, clamp_range=None
    )
    assert isinstance(scales, dict)
    # No scales applied due to high tolerance
    assert len(scales) == 0


def test_variance_guard_tensor_and_batch_helpers():
    g = VarianceGuard()
    # _ensure_tensor_value branches
    out_np = g._ensure_tensor_value(torch.tensor([1, 2, 3]).numpy())
    assert isinstance(out_np, torch.Tensor)
    out_list = g._ensure_tensor_value([1, 2, 3])
    assert isinstance(out_list, torch.Tensor)
    out_num = g._ensure_tensor_value(5.0)
    assert isinstance(out_num, torch.Tensor)

    class X:
        pass

    out_other = g._ensure_tensor_value(X())
    assert isinstance(out_other, X)

    # _materialize_batch and _tensorize_calibration_batches branches
    batch = {
        "input_ids": torch.ones(1, 2).cuda()
        if torch.cuda.is_available()
        else torch.ones(1, 2),
        "labels": [1, 2],
        "meta": {"window_id": "w1"},
    }
    mat = g._materialize_batch(batch)
    assert isinstance(mat["input_ids"], torch.Tensor) and not mat["input_ids"].is_cuda
    tensored = g._tensorize_calibration_batches([batch])
    assert isinstance(tensored[0]["labels"], torch.Tensor)
