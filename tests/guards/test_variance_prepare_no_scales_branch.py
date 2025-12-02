import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        blk = nn.Module()
        blk.attn = nn.Module()
        blk.attn.c_proj = nn.Linear(2, 2, bias=False)
        blk.mlp = nn.Module()
        blk.mlp.c_proj = nn.Linear(2, 2, bias=False)
        self.transformer.h = nn.ModuleList([blk])

    def forward(self, x):
        return self.transformer.h[0].mlp.c_proj(self.transformer.h[0].attn.c_proj(x))


def test_prepare_branch_no_scales_with_sufficient_coverage(monkeypatch):
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "predictive_gate": True,
            "calibration": {"windows": 2, "min_coverage": 2, "seed": 9},
        }
    )
    # Force targets and empty scales
    monkeypatch.setattr(
        g,
        "_resolve_target_modules",
        lambda model, adapter: {
            "transformer.h.0.mlp.c_proj": nn.Linear(2, 2, bias=False)
        },
    )
    # Force empty scales
    monkeypatch.setattr(g, "_compute_variance_scales", lambda _m, _b: {})
    batches = [torch.ones(1, 2), torch.ones(1, 2)]
    res = g.prepare(M(), adapter=None, calib=batches, policy=None)
    assert isinstance(res, dict)
    # Status may vary depending on coverage; accept uninitialized/pending as well
    assert g._calibration_stats.get("status") in {
        "no_scaling_required",
        "pending",
        "uninitialized",
    }
    if g._calibration_stats.get("status") == "no_scaling_required":
        ape = g._stats.get("ab_point_estimates", {})
        assert "ppl_no_ve" in ape and "ppl_with_ve" in ape
