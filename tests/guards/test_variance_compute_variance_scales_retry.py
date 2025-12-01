import torch.nn as nn

import invarlock.guards.variance as variance_mod
from invarlock.guards.variance import VarianceGuard


def test_compute_variance_scales_relaxed_retry(monkeypatch):
    calls = {"n": 0}

    def fake_equalise(model, dataloader, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {}  # first attempt yields nothing
        # second attempt returns a small non‑unity scale
        return {"block0.mlp": 1.08}

    monkeypatch.setattr(variance_mod, "equalise_residual_variance", fake_equalise)

    policy = {
        "min_gain": 0.0,
        "scope": "both",
        "max_calib": 0,
        "deadband": 0.02,
        "clamp": (0.85, 1.12),
    }
    g = VarianceGuard(policy=policy)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 2, bias=False)

    out = g._compute_variance_scales(M(), [])
    assert calls["n"] >= 2
    # After retry, we should have captured a scale
    assert isinstance(out, dict) and len(out) >= 1
