import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


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

    def forward(self, inputs, labels=None):
        x = self.transformer.h[0].attn.c_proj(inputs)
        x = torch.tanh(x)
        x = self.transformer.h[0].mlp.c_proj(x)
        return x


def _make_batches(n=4, d=4, b=2):
    # Return a list of simple tensor batches (inputs, attention_mask)
    return [(torch.randn(b, d), torch.ones(b, d)) for _ in range(n)]


def test_prepare_runs_ab_subpass_and_populates_stats():
    model = TinyModel()
    batches = _make_batches(n=6)
    policy = {
        "scope": "both",
        "min_gain": 0.0,
        "deadband": 0.0,
        "max_calib": 20,  # scale windows = 2; calibration windows below
        "clamp": (0.8, 1.2),
        "calibration": {"windows": 3, "min_coverage": 2, "seed": 7},
        "predictive_gate": True,
        "topk_backstop": 1,
        "min_abs_adjust": 0.0,
        "max_scale_step": 0.0,
    }
    g = VarianceGuard(policy=policy)
    res = g.prepare(model, adapter=None, calib=batches, policy=None)
    assert isinstance(res, dict) and "baseline_metrics" in res
    # Expect ratio_ci or predictive gate state recorded and ab provenance present
    assert isinstance(getattr(g, "_calibration_stats", {}), dict)
    assert "ab_provenance" in g._stats
    # Either we completed or proceeded partially; both acceptable
    status = g._calibration_stats.get("status")
    assert status in {"complete", "pending", "no_scaling_required", "insufficient"}
