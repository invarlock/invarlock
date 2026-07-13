import torch.nn as nn

import invarlock.guards.variance_scaling as variance_scaling_mod
from invarlock.guards.variance import VarianceGuard


def test_compute_variance_scales_ties_and_trim_order(monkeypatch):
    # Craft raw scales with identical absolute deltas; >=1.0 should be preferred
    # when trimming to max_adjusted_modules due to the sort key bias (+2.0).
    def fake_equalise(*_args, **_kwargs):
        return {
            "block0.mlp": 1.20,  # delta 0.20
            "block1.attn": 0.80,  # delta 0.20
            "block2.mlp": 1.20,  # delta 0.20
        }

    monkeypatch.setattr(
        variance_scaling_mod, "equalise_residual_variance", fake_equalise
    )

    policy = {
        "min_gain": 0.0,
        "scope": "both",
        "max_calib": 0,
        "deadband": 0.0,
        "clamp": (0.8, 1.2),
        "min_abs_adjust": 0.0,
        "max_adjusted_modules": 2,  # force trimming
    }
    g = VarianceGuard(policy=policy)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 2, bias=False)

    out = g._compute_variance_scales(M(), [])
    # We expect both >=1.0 entries to be kept and the <1.0 entry dropped
    assert set(out.keys()).issubset({"block0.mlp", "block2.mlp"})
    assert "block1.attn" not in out


def test_compute_variance_scales_uses_module_name_for_exact_ties(monkeypatch):
    monkeypatch.setattr(
        variance_scaling_mod,
        "equalise_residual_variance",
        lambda *_args, **_kwargs: {"z.module": 1.125, "a.module": 0.875},
    )
    guard = VarianceGuard(
        policy={
            "scope": "both",
            "max_calib": 0,
            "deadband": 0.0,
            "min_abs_adjust": 0.2,
            "max_scale_step": 0.02,
            "topk_backstop": 1,
        }
    )

    result = guard._compute_variance_scales(nn.Linear(2, 2, bias=False), [])

    assert result == {"a.module": 0.98}
