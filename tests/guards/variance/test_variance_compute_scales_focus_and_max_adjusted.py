import torch.nn as nn

import invarlock.guards.variance_scaling as variance_scaling_mod
from invarlock.guards.variance import VarianceGuard


def test_compute_scales_focus_and_max_adjusted(monkeypatch):
    def fake_equalise(model, dataloader, **kwargs):
        return {
            "block0.mlp": 1.20,  # candidate
            "block1.mlp": 0.85,  # candidate
            "block2.attn": 1.01,  # below min_abs will be filtered if min_abs high
        }

    monkeypatch.setattr(
        variance_scaling_mod, "equalise_residual_variance", fake_equalise
    )

    policy = {
        "min_gain": 0.0,
        "scope": "both",
        "max_calib": 0,
        "deadband": 0.0,
        "clamp": (0.85, 1.20),
        "min_abs_adjust": 0.05,
        "max_scale_step": 0.20,
        "topk_backstop": 0,
        "target_modules": ["transformer.h.0.mlp.c_proj", "transformer.h.1.mlp.c_proj"],
        "max_adjusted_modules": 1,
    }
    g = VarianceGuard(policy=policy)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 2, bias=False)

    out = g._compute_variance_scales(M(), [])
    # Because of focus and max_adjusted_modules=1, result contains at least one entry
    assert isinstance(out, dict) and len(out) >= 1


def test_compute_scales_monitor_only_returns_empty_result() -> None:
    guard = VarianceGuard()
    guard._monitor_only = True  # noqa: SLF001

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 2, bias=False)

    out = guard._compute_variance_scales(M(), [])
    assert out == {}
    assert guard._raw_scales == {}  # noqa: SLF001


def test_compute_scales_backstop_applies_max_scale_step(monkeypatch) -> None:
    monkeypatch.setattr(
        variance_scaling_mod,
        "equalise_residual_variance",
        lambda *_args, **_kwargs: {"block0.mlp": 1.4},
    )

    guard = VarianceGuard(
        policy={
            "min_gain": 0.0,
            "scope": "both",
            "max_calib": 0,
            "deadband": 0.0,
            "clamp": (0.85, 1.40),
            "min_abs_adjust": 0.5,
            "max_scale_step": 0.1,
            "topk_backstop": 1,
            "max_adjusted_modules": 0,
        }
    )

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 2, bias=False)

    out = guard._compute_variance_scales(M(), [])
    assert out == {"block0.mlp": 1.1}


def test_compute_scales_backstop_skips_candidate_below_threshold(monkeypatch) -> None:
    monkeypatch.setattr(
        variance_scaling_mod,
        "equalise_residual_variance",
        lambda *_args, **_kwargs: {"block0.mlp": 1.1},
    )

    guard = VarianceGuard(
        policy={
            "min_gain": 0.0,
            "scope": "both",
            "max_calib": 0,
            "deadband": 0.0,
            "clamp": (0.9, 1.2),
            "min_abs_adjust": 0.4,
            "max_scale_step": 0.0,
            "topk_backstop": 1,
            "max_adjusted_modules": 0,
        }
    )

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 2, bias=False)

    assert guard._compute_variance_scales(M(), []) == {}
