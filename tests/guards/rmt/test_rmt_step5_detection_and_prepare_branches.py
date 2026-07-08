from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import invarlock.guards.rmt as R
import invarlock.guards.rmt_detection as rmt_detection
from invarlock.guards.rmt_analysis import layer_svd_stats


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

    monkeypatch.setattr(R.rmt_analysis, "layer_svd_stats", _fake_layer_svd_stats)
    monkeypatch.setattr(rmt_detection, "_apply_rmt_correction", lambda *_a, **_k: None)

    out = guard._apply_rmt_detection_and_correction(model)
    assert out["corrected_layers"] == 1
    assert out["n_layers_flagged"] == 1


def test_rmt_step5_synthetic_boundary_is_strictly_greater() -> None:
    at_threshold = nn.Linear(1, 1, bias=False)
    above_threshold = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        at_threshold.weight.fill_(1.65)
        above_threshold.weight.fill_(1.650001)

    modules = [("at_threshold", at_threshold), ("above_threshold", above_threshold)]
    baseline_sigmas = {"at_threshold": 1.0, "above_threshold": 1.0}
    baseline_mp_stats = {
        name: {"sigma_base": 1.0, "mp_bulk_edge_base": 1.0} for name in baseline_sigmas
    }

    out = rmt_detection.step5_detect_and_correct_modules(
        modules,
        baseline_sigmas=baseline_sigmas,
        baseline_mp_stats=baseline_mp_stats,
        deadband=0.10,
        margin=1.5,
        correct=False,
    )

    assert out["per_layer"][0]["has_outlier"] is False
    assert out["per_layer"][0]["sigma_max"] == pytest.approx(1.65, rel=1e-6)
    assert out["per_layer"][0]["skip_reason"] == ("≤ threshold (ratio=1.65 ≤ 1.65)")
    assert out["per_layer"][1]["has_outlier"] is True
    assert out["per_layer"][1]["sigma_max"] == pytest.approx(1.650001, rel=1e-6)
    assert out["flagged_layers"] == [1]
    assert out["n_layers_flagged"] == 1


def test_rmt_no_baseline_spike_spectrum_uses_percentile_oracle() -> None:
    layer = nn.Linear(51, 51, bias=False)
    singular_values = torch.ones(51)
    singular_values[0] = 100.0
    with torch.no_grad():
        layer.weight.copy_(torch.diag(singular_values))

    stats = layer_svd_stats(layer, module_name="spike")
    details = stats["worst_details"]

    assert details["normalization"] == "98th_percentile"
    assert details["s_max"] == pytest.approx(100.0)
    assert details["s_98"] == pytest.approx(1.0)
    assert details["ratio"] == pytest.approx(100.0)
    assert details["mp_edge"] == pytest.approx(14.2828568570857)

    out = rmt_detection.step5_detect_and_correct_modules(
        [("block.attn.c_proj", layer)],
        baseline_sigmas={},
        baseline_mp_stats={},
        deadband=0.0,
        margin=50.0,
        correct=False,
    )

    assert out["has_outliers"] is True
    assert out["n_layers_flagged"] == 1
    assert out["per_layer"][0]["details"]["ratio"] == pytest.approx(100.0)


def test_rmt_prepare_policy_parsing_and_activation_required_paths(monkeypatch) -> None:
    model = _TinyModel()

    # Exercise window_count parse failure (count not int) + estimator parsing.
    guard = R.RMTGuard()
    guard.activation_sampling["windows"]["count"] = "bad"
    deadband_before = guard.deadband
    margin_before = guard.margin
    out = guard.prepare(
        model,
        calib=None,
        policy={
            "q": "auto",
            "deadband": "not-a-float",
            "margin": "not-a-float",
            "estimator": {"iters": "bad", "init": "bogus"},
            "activation": {
                "sampling": {"windows": {"count": "bad", "indices_policy": "last"}}
            },
            "activation_required": False,
        },
    )
    assert out["ready"] is True
    assert guard.q == "auto"
    assert guard.deadband == deadband_before
    assert guard.margin == margin_before
    assert guard.estimator["iters"] == 3
    assert guard.estimator["init"] == "ones"
    assert guard.activation_sampling["windows"]["count"] == "bad"
    assert guard.activation_sampling["windows"]["indices_policy"] == "last"

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

    guard_q = R.RMTGuard()
    guard_q.prepare(model, calib=None, policy={"q": object()})
    assert guard_q.q == "auto"

    guard_iters = R.RMTGuard()
    guard_iters.prepare(
        model,
        calib=None,
        policy={"estimator": {"iters": -2, "init": "e0"}},
    )
    assert guard_iters.estimator["iters"] == 1
    assert guard_iters.estimator["init"] == "e0"

    guard_sampling = R.RMTGuard()
    baseline_sampling = dict(guard_sampling.activation_sampling["windows"])
    guard_sampling.prepare(
        model, calib=None, policy={"activation": {"sampling": "bad"}}
    )
    assert guard_sampling.activation_sampling["windows"] == baseline_sampling

    guard_windows = R.RMTGuard()
    guard_windows.prepare(
        model,
        calib=None,
        policy={"activation": {"sampling": {"windows": "bad"}}},
    )
    assert guard_windows.activation_sampling["windows"]["count"] == 8
    assert (
        guard_windows.activation_sampling["windows"]["indices_policy"]
        == "evenly_spaced"
    )

    guard_indices = R.RMTGuard()
    guard_indices.prepare(
        model,
        calib=None,
        policy={"activation": {"sampling": {"windows": {"indices_policy": "last"}}}},
    )
    assert guard_indices.activation_sampling["windows"]["count"] == 8
    assert guard_indices.activation_sampling["windows"]["indices_policy"] == "last"

    guard_count = R.RMTGuard()
    guard_count.prepare(
        model,
        calib=None,
        policy={"activation": {"sampling": {"windows": {"count": 2}}}},
    )
    assert guard_count.activation_sampling["windows"]["count"] == 2
    assert (
        guard_count.activation_sampling["windows"]["indices_policy"] == "evenly_spaced"
    )
