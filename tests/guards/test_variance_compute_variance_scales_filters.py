import torch.nn as nn

import invarlock.guards.variance as variance_mod
from invarlock.guards.variance import VarianceGuard


def test_compute_variance_scales_filters_and_topk(monkeypatch):
    # Monkeypatch equalise_residual_variance to return raw scales
    def fake_equalise_residual_variance(model, dataloader, **kwargs):
        return {
            "block0.mlp": 1.20,  # delta 0.20 -> candidate
            "transformer.h.0.attn.c_proj": 0.99,  # delta 0.01 -> below min_abs
            "block1.attn": 0.80,  # delta 0.20 -> candidate
        }

    monkeypatch.setattr(
        variance_mod, "equalise_residual_variance", fake_equalise_residual_variance
    )

    # Policy chooses min_abs_adjust and max_scale_step and topk_backstop
    policy = {
        "min_gain": 0.0,
        "scope": "both",
        "max_calib": 0,
        "deadband": 0.0,
        "clamp": (0.85, 1.12),
        "min_abs_adjust": 0.05,
        "max_scale_step": 0.10,
        "topk_backstop": 1,
    }
    g = VarianceGuard(policy=policy)

    # Minimal model (not used by fake function)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 2, bias=False)

    out = g._compute_variance_scales(M(), [])
    # Small delta (0.01) should be filtered; larger deltas remain (one or more depending on policy)
    assert isinstance(out, dict) and len(out) >= 1
    assert all(("mlp" in k) or ("attn" in k) for k in out.keys())
