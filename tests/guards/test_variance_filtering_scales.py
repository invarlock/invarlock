import torch.nn as nn

import invarlock.guards.variance as var_mod
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

    def forward(self, x):
        return self.transformer.h[0].mlp.c_proj(self.transformer.h[0].attn.c_proj(x))


def test_compute_variance_scales_backstop_and_limit(monkeypatch):
    g = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 10,
            "deadband": 0.002,
            "min_abs_adjust": 0.0025,
            "max_scale_step": 0.001,
            "topk_backstop": 1,
            "max_adjusted_modules": 0,
        }
    )

    # First scenario: all deltas below min_abs_adjust → backstop injects best
    def fake_eq_1(*_, **__):
        return {"block0.attn": 1.001, "block0.mlp": 1.003}

    monkeypatch.setattr(var_mod, "equalise_residual_variance", fake_eq_1)
    out = g._compute_variance_scales(TinyModel(), dataloader=[object()])
    # One scale injected by backstop, normalized key
    assert list(out.keys()) == ["block0.mlp"] or any(k.endswith("mlp") for k in out)

    # Second scenario: two large deltas, but trim to max_adjusted_modules=1
    g2 = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 10,
            "deadband": 0.0,
            "min_abs_adjust": 0.0,
            "max_scale_step": 0.0,
            "topk_backstop": 0,
            "max_adjusted_modules": 1,
        }
    )

    def fake_eq_2(*_, **__):
        return {"block0.attn": 1.02, "block0.mlp": 0.98}

    monkeypatch.setattr(var_mod, "equalise_residual_variance", fake_eq_2)
    out2 = g2._compute_variance_scales(TinyModel(), dataloader=[object()])
    assert len(out2) == 1
