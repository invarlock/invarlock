import torch.nn as nn

import invarlock.guards.variance as variance_mod
from invarlock.guards.variance import VarianceGuard


def test_refresh_calibration_defaults_coerces_non_dict_calibration() -> None:
    g = VarianceGuard(policy={"calibration": ["bad"]})
    calibration = g._policy.get("calibration")
    assert isinstance(calibration, dict)
    assert calibration["windows"] == 6
    assert calibration["min_coverage"] == 4


def test_compute_variance_scales_filters_raw_scales_via_scale_matches_target(
    monkeypatch,
) -> None:
    def fake_equalise(*_a, **_k):
        return {"block0.attn": 1.1}

    monkeypatch.setattr(variance_mod, "equalise_residual_variance", fake_equalise)

    policy = {
        "min_gain": 0.0,
        "scope": "both",
        "max_calib": 0,
        "deadband": 0.0,
        "clamp": (0.5, 2.0),
        "min_abs_adjust": 0.0,
        "max_scale_step": 0.0,
        "topk_backstop": 0,
        # Focus includes the normalized ".c_proj" form.
        "target_modules": ["transformer.h.0.attn"],
        "max_adjusted_modules": 0,
    }
    g = VarianceGuard(policy=policy)
    # Use a slightly-mismatched key to force the fallback `_scale_matches_target` branch.
    g._target_modules = {"transformer.h.0.attn": nn.Linear(2, 2, bias=False)}
    monkeypatch.setattr(g, "_tensorize_calibration_batches", lambda batches: list(batches))

    out = g._compute_variance_scales(nn.Linear(2, 2, bias=False), [])
    assert out.get("block0.attn") == 1.1

