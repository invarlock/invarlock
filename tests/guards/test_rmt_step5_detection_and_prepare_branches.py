from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from invarlock.guards import rmt as R


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_attn = nn.Linear(2, 2, bias=False)
        self.attn.c_proj = nn.Linear(2, 2, bias=False)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(2, 2, bias=False)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _TinyBlock()


def test_rmt_apply_step5_detection_and_correction_branches(monkeypatch) -> None:
    guard = R.RMTGuard(deadband=0.1, margin=1.5, correct=True)
    model = _TinyModel()

    guard.baseline_sigmas = {"block.attn.c_attn": 1.0}
    guard.baseline_mp_stats = {
        "block.attn.c_attn": {"sigma_base": 1.0, "mp_bulk_edge_base": 1.0}
    }

    per_name_calls: dict[str, int] = {}

    def _fake_layer_svd_stats(
        module, baseline_sigmas=None, baseline_mp_stats=None, module_name=""
    ):
        _ = module, baseline_sigmas, baseline_mp_stats
        per_name_calls[module_name] = per_name_calls.get(module_name, 0) + 1

        if module_name == "block.attn.c_attn":
            if per_name_calls[module_name] == 1:
                return {"sigma_min": 0.0, "sigma_max": 10.0, "worst_ratio": 10.0}
            return {"sigma_min": 0.0, "sigma_max": 1.0, "worst_ratio": 1.0}

        if module_name == "block.attn.c_proj":
            # No baseline MP stats entry → fallback branch, outlier
            return {"sigma_min": 0.0, "sigma_max": 2.0, "worst_ratio": 2.0}

        # Fallback branch, no outlier
        return {"sigma_min": 0.0, "sigma_max": 1.0, "worst_ratio": 1.0}

    monkeypatch.setattr(R, "layer_svd_stats", _fake_layer_svd_stats)
    monkeypatch.setattr(R, "_apply_rmt_correction", lambda *_a, **_k: None)

    out = guard._apply_rmt_detection_and_correction(model)
    assert out["corrected_layers"] == 1
    assert out["n_layers_flagged"] == 1


def test_rmt_prepare_policy_parsing_and_activation_required_paths(monkeypatch) -> None:
    model = _TinyModel()

    # Exercise window_count parse failure (count not int) + estimator parsing.
    guard = R.RMTGuard()
    guard.activation_sampling["windows"]["count"] = "bad"
    out = guard.prepare(
        model,
        calib=None,
        policy={
            "q": "not-a-float",
            "margin": "not-a-float",
            "estimator": {"iters": -1, "init": "bogus"},
            "activation_required": False,
        },
    )
    assert out["ready"] is True
    assert guard.q == "auto"
    assert guard.estimator["iters"] == 1
    assert guard.estimator["init"] == "ones"

    # activation_required=True with no calibration batches returns a hard failure.
    guard_required = R.RMTGuard()
    out_required = guard_required.prepare(
        model, calib=None, policy={"activation_required": True}
    )
    assert out_required["ready"] is False

    # activation_required=True with batches but baseline unavailable returns a different failure.
    guard_baseline = R.RMTGuard()
    monkeypatch.setattr(
        guard_baseline,
        "_collect_calibration_batches",
        lambda *_a, **_k: [{"input_ids": [1]}],
    )
    monkeypatch.setattr(
        guard_baseline, "_compute_activation_edge_risk", lambda *_a, **_k: None
    )
    out_baseline = guard_baseline.prepare(
        model,
        calib=SimpleNamespace(),
        policy={"activation_required": True},
    )
    assert out_baseline["ready"] is False
    assert (
        guard_baseline._activation_required_reason == "activation_baseline_unavailable"
    )
