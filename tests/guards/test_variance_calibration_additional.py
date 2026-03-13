from __future__ import annotations

import math

import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard
from invarlock.guards.variance_evaluation import (
    evaluate_calibration_pass,
    refresh_after_edit_metrics,
)


class _TinyBlock(nn.Module):
    def __init__(self, d: int = 4) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.c_proj = nn.Linear(d, d, bias=False)


class _TinyModel(nn.Module):
    def __init__(self, d: int = 4) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([_TinyBlock(d)])

    def forward(self, inputs, labels=None):
        return inputs


def _guard() -> VarianceGuard:
    return VarianceGuard(
        policy={"scope": "both", "min_gain": 0.0, "max_calib": 20, "alpha": 0.05}
    )


def test_evaluate_calibration_pass_marks_pending_when_delta_ci_is_not_finite(
    monkeypatch,
) -> None:
    guard = _guard()
    model = _TinyModel()
    guard._target_modules = guard._resolve_target_modules(model, adapter=None)
    guard._scales = {next(iter(guard._target_modules)): 0.95}
    guard._calibration_window_ids = ["0", "1"]
    guard._stats.setdefault("ab_provenance", {})

    losses_a = [math.log(2.0), math.log(4.0)]
    losses_b = [math.log(2.2), math.log(4.4)]
    calls = {"count": 0}

    def fake_compute_ppl(_model, _batches, _device, *, return_counts=False):
        calls["count"] += 1
        if calls["count"] == 1:
            return [2.0, 4.0], losses_a, []
        return [2.2, 4.4], losses_b, []

    monkeypatch.setattr(guard, "_compute_ppl_for_batches", fake_compute_ppl)
    monkeypatch.setattr(guard, "enable", lambda _model: True)
    monkeypatch.setattr(guard, "disable", lambda _model: True)
    monkeypatch.setattr(
        guard, "_bootstrap_mean_ci", lambda *_args, **_kwargs: (1.0, 1.2)
    )
    monkeypatch.setattr(guard, "_fingerprint_targets", lambda: "fingerprint")

    evaluate_calibration_pass(
        guard,
        model,
        calibration_batches=[(torch.ones(1, 4), torch.zeros(1, 4))] * 2,
        min_coverage=2,
        calib_seed=7,
        tag="unit",
        compute_paired_delta_log_ci_fn=lambda *_args, **_kwargs: (
            float("nan"),
            float("inf"),
        ),
    )

    assert guard._calibration_stats["status"] == "pending"
    assert guard._predictive_gate_state["evaluated"] is True
    assert guard._predictive_gate_state["delta_ci"] == (None, None)
    assert guard._predictive_gate_state["gain_ci"] == (None, None)


def test_evaluate_calibration_pass_records_not_evaluated_condition_b_when_coverage_is_low(
    monkeypatch,
) -> None:
    guard = _guard()
    model = _TinyModel()
    guard._target_modules = guard._resolve_target_modules(model, adapter=None)
    guard._scales = {next(iter(guard._target_modules)): 0.95}
    guard._calibration_window_ids = ["0", "1"]
    guard._stats.setdefault("ab_provenance", {})

    calls = {"count": 0}

    def fake_compute_ppl(_model, _batches, _device, *, return_counts=False):
        calls["count"] += 1
        if calls["count"] == 1:
            return [2.0], [math.log(2.0)], [4]
        return [], [], []

    monkeypatch.setattr(guard, "_compute_ppl_for_batches", fake_compute_ppl)
    monkeypatch.setattr(guard, "enable", lambda _model: True)
    monkeypatch.setattr(guard, "disable", lambda _model: True)
    monkeypatch.setattr(guard, "_fingerprint_targets", lambda: "fingerprint")

    evaluate_calibration_pass(
        guard,
        model,
        calibration_batches=[(torch.ones(1, 4), torch.zeros(1, 4))] * 2,
        min_coverage=2,
        calib_seed=11,
        tag="lowcov",
    )

    assert guard._predictive_gate_state["reason"] == "insufficient_coverage"
    assert guard._stats["ab_provenance"]["condition_b"]["status"] == "not_evaluated"


def test_refresh_after_edit_metrics_filters_focus_modules_and_recomputes_state(
    monkeypatch,
) -> None:
    guard = _guard()
    model = _TinyModel()
    guard._prepared = True
    guard._calibration_batches = [object()]
    guard._focus_modules = {"transformer.h.0.attn.c_proj"}
    guard._adapter_ref = None
    guard._raw_scales = {
        "transformer.h.0.attn.c_proj": 0.9,
        "transformer.h.0.mlp.c_proj": 1.1,
    }

    monkeypatch.setattr(
        guard,
        "_resolve_target_modules",
        lambda _model, _adapter=None: {
            "transformer.h.0.attn.c_proj": object(),
            "transformer.h.0.mlp.c_proj": object(),
        },
    )

    def fake_compute_scales(_model, _batches):
        guard._raw_scales = {
            "transformer.h.0.attn.c_proj": 0.9,
            "transformer.h.0.mlp.c_proj": 1.1,
        }
        return dict(guard._raw_scales)

    monkeypatch.setattr(guard, "_compute_variance_scales", fake_compute_scales)
    monkeypatch.setattr(
        guard, "_is_focus_match", lambda name: name.endswith("attn.c_proj")
    )
    calls: list[tuple[int, int, str]] = []
    monkeypatch.setattr(
        guard,
        "_evaluate_calibration_pass",
        lambda _model, batches, min_cov, seed, tag: calls.append(
            (len(batches), min_cov, tag)
        ),
    )

    refresh_after_edit_metrics(guard, model)

    assert guard._scales == {"transformer.h.0.attn.c_proj": 0.9}
    assert guard._stats["target_modules_post_edit"] == [
        "transformer.h.0.attn.c_proj",
        "transformer.h.0.mlp.c_proj",
    ]
    assert guard._raw_scales_post_edit == {"transformer.h.0.attn.c_proj": 0.9}
    assert calls and calls[0][2] == "post_edit"
    assert guard._post_edit_evaluated is True


def test_evaluate_calibration_pass_handles_no_valid_samples_with_zero_min_coverage(
    monkeypatch,
) -> None:
    guard = _guard()
    model = _TinyModel()
    guard._stats.setdefault("ab_provenance", {})
    guard._scales = {}

    monkeypatch.setattr(
        guard,
        "_compute_ppl_for_batches",
        lambda *_args, **_kwargs: ([], [], []),
    )
    monkeypatch.setattr(guard, "_fingerprint_targets", lambda: None)

    evaluate_calibration_pass(
        guard,
        model,
        calibration_batches=[object()],
        min_coverage=0,
        calib_seed=5,
        tag="no-valid",
    )

    assert guard._predictive_gate_state["reason"] == "no_valid_samples"
    assert guard._ratio_ci is None


def test_evaluate_calibration_pass_handles_empty_ratios_invalid_ci_and_existing_condition_b(
    monkeypatch,
) -> None:
    guard = _guard()
    model = _TinyModel()
    guard._target_modules = guard._resolve_target_modules(model, adapter=None)
    guard._scales = {next(iter(guard._target_modules)): 0.95}
    guard._calibration_window_ids = ["0", "1"]
    guard._stats.setdefault("ab_provenance", {})["condition_b"] = {"status": "existing"}

    calls = {"count": 0}

    def fake_compute_ppl(_model, _batches, _device, *, return_counts=False):
        calls["count"] += 1
        if calls["count"] == 1:
            return [0.0, 0.0], [0.1, 0.2], []
        return [0.0, 0.0], [0.1, 0.2], []

    monkeypatch.setattr(guard, "_compute_ppl_for_batches", fake_compute_ppl)
    monkeypatch.setattr(guard, "enable", lambda _model: True)
    monkeypatch.setattr(guard, "disable", lambda _model: True)
    monkeypatch.setattr(guard, "_fingerprint_targets", lambda: "fp")

    evaluate_calibration_pass(
        guard,
        model,
        calibration_batches=[object(), object()],
        min_coverage=2,
        calib_seed=13,
        tag="ratio-empty",
        compute_paired_delta_log_ci_fn=lambda *_args, **_kwargs: (
            float("nan"),
            float("nan"),
        ),
    )

    assert guard._stats["ab_provenance"]["condition_b"]["status"] == "existing"
    assert guard._predictive_gate_state["delta_ci"] == (None, None)
    assert guard._stats["ab_point_estimates"]["tag"] == "ratio-empty"


def test_evaluate_calibration_pass_preserves_disabled_reason_in_monitor_mode(
    monkeypatch,
) -> None:
    guard = VarianceGuard(
        policy={
            "scope": "both",
            "min_gain": 0.0,
            "max_calib": 20,
            "alpha": 0.05,
            "predictive_gate": False,
        }
    )
    model = _TinyModel()
    guard._target_modules = guard._resolve_target_modules(model, adapter=None)
    guard._scales = {next(iter(guard._target_modules)): 0.95}
    guard._calibration_window_ids = ["0", "1"]
    guard._stats.setdefault("ab_provenance", {})

    calls = {"count": 0}

    def fake_compute_ppl(_model, _batches, _device, *, return_counts=False):
        calls["count"] += 1
        if calls["count"] == 1:
            return [2.0], [math.log(2.0)], [4]
        return [], [], []

    monkeypatch.setattr(guard, "_compute_ppl_for_batches", fake_compute_ppl)
    monkeypatch.setattr(guard, "enable", lambda _model: True)
    monkeypatch.setattr(guard, "disable", lambda _model: True)
    monkeypatch.setattr(guard, "_fingerprint_targets", lambda: "fp")

    evaluate_calibration_pass(
        guard,
        model,
        calibration_batches=[object(), object()],
        min_coverage=2,
        calib_seed=17,
        tag="disabled-monitor",
    )

    assert guard._predictive_gate_state["reason"] == "disabled"
    assert guard._stats["ab_provenance"]["condition_b"]["status"] == "not_evaluated"


def test_refresh_after_edit_metrics_skips_scale_log_when_no_normalized_scales(
    monkeypatch,
) -> None:
    guard = _guard()
    model = _TinyModel()
    guard._prepared = True
    guard._calibration_batches = [object()]
    guard._focus_modules = {"transformer.h.0.attn.c_proj"}
    guard._adapter_ref = None
    guard._raw_scales = {}

    monkeypatch.setattr(
        guard,
        "_resolve_target_modules",
        lambda _model, _adapter=None: {"transformer.h.0.attn.c_proj": object()},
    )
    monkeypatch.setattr(
        guard,
        "_compute_variance_scales",
        lambda _model, _batches: {"transformer.h.0.attn.c_proj": 0.9},
    )
    monkeypatch.setattr(guard, "_is_focus_match", lambda _name: False)
    calls: list[str] = []
    monkeypatch.setattr(
        guard,
        "_evaluate_calibration_pass",
        lambda _model, _batches, _min_cov, _seed, tag: calls.append(tag),
    )

    refresh_after_edit_metrics(guard, model)

    assert guard._scales == {}
    assert calls == ["post_edit"]
    assert guard._post_edit_evaluated is True
