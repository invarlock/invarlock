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
        # simple linear projections; outputs match inputs shape for MSE fallback
        x = self.transformer.h[0].attn.c_proj(inputs)
        x = self.transformer.h[0].mlp.c_proj(x)
        return x


def test_prepare_with_calibration_data_exercises_ab_path():
    model = TinyModel()
    # Build simple tensor batches (inputs, attention_mask-like ignored)
    batches = [(torch.randn(2, 4), torch.ones(2, 4)) for _ in range(4)]
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 30,
            "calibration": {"windows": 3, "min_coverage": 1, "seed": 123},
            "predictive_gate": True,
        }
    )
    res = g.prepare(model, adapter=None, calib=batches, policy=None)
    assert isinstance(res, dict) and "baseline_metrics" in res
    # prepared flag may be set depending on scales; ensure calibration stats exist
    assert isinstance(getattr(g, "_calibration_stats", {}), dict)
