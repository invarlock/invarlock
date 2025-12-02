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


def _batches(n=6, b=4, d=4):
    return [(torch.randn(b, d), torch.ones(b, d)) for _ in range(n)]


def test_predictive_gate_records_delta_and_gain_ci_when_complete():
    g = VarianceGuard(
        policy={
            "scope": "both",
            "deadband": 0.0,
            "min_gain": 0.0,
            "max_calib": 60,
            "clamp": (0.8, 1.2),
            "calibration": {"windows": 4, "min_coverage": 3, "seed": 21},
            "predictive_gate": True,
        }
    )
    res = g.prepare(TinyModel(), adapter=None, calib=_batches(6), policy=None)
    assert isinstance(res, dict)
    status = g._calibration_stats.get("status")
    assert status in {"complete", "pending", "no_scaling_required", "insufficient"}
    if status == "complete":
        pg = g._stats.get("predictive_gate", {})
        assert isinstance(pg.get("delta_ci"), tuple)
        assert isinstance(pg.get("gain_ci"), tuple)
        # Values should be floats or None; at least not both Nones here
        lo, hi = pg.get("delta_ci")
        assert (lo is None and hi is None) is False


def test_finalize_copies_predictive_gate_into_metrics():
    g = VarianceGuard(
        policy={
            "scope": "both",
            "deadband": 0.0,
            "min_gain": 0.0,
            "max_calib": 0,
            "predictive_gate": False,
        }
    )
    # Simple prepare (no calibration), then finalize should add predictive_gate structure
    g.prepare(TinyModel(), adapter=None, calib=None, policy=None)
    out = g.finalize(TinyModel())
    metrics = out.get("metrics", {})
    assert "predictive_gate" in metrics
