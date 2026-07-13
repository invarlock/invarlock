from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

import invarlock.core.runner_runtime.eval_metrics as rem
from tests.core._support_runner_eval_metrics import _FakeModel, _FakeRunner, _summary


def test_compute_real_metrics_uses_logloss_lists_final_weights_and_seq2seq(
    monkeypatch,
) -> None:
    preview_summary = _summary(
        ppl=2.0,
        total_tokens=0,
        weighted_log_loss=0.0,
        num_batches=1,
        log_losses=[math.log(2.0)],
        window_ids=[0],
        token_counts=[],
        actual_token_counts=[1],
    )
    final_summary = _summary(
        ppl=4.0,
        total_tokens=0,
        weighted_log_loss=0.0,
        num_batches=1,
        log_losses=[math.log(4.0)],
        window_ids=[1],
        token_counts=[3],
        actual_token_counts=[3],
    )

    def fake_slice_summary(*_args, start_idx: int, **_kwargs):
        return (preview_summary, None) if start_idx == 0 else (final_summary, None)

    monkeypatch.setattr(rem, "compute_slice_summary", fake_slice_summary)
    monkeypatch.setattr(rem, "measure_latency", lambda *_args, **_kwargs: 0.25)
    monkeypatch.setattr(
        rem,
        "compute_window_pairing_metrics",
        lambda **_kwargs: {
            "preview": {"matched": 1, "expected": 1, "reason": None},
            "final": {"matched": 1, "expected": 1, "reason": None},
            "match_fraction": 1.0,
            "overlap_fraction": 0.0,
            "duplicate_fraction": 0.0,
            "count_mismatch": False,
            "reason": None,
        },
    )
    monkeypatch.setattr(
        rem,
        "assess_bootstrap_coverage",
        lambda **_kwargs: {
            "preview_required": 4,
            "final_required": 4,
            "replicates_required": 1000,
            "preview_ok": False,
            "final_ok": True,
            "replicates_ok": True,
            "coverage": {
                "preview": {"used": 1, "required": 4, "ok": False},
                "final": {"used": 1, "required": 4, "ok": True},
                "replicates": {"used": 0, "required": 1000, "ok": True},
            },
        },
    )

    runner = _FakeRunner()
    config = SimpleNamespace(
        context={
            "eval": {
                "loss": {"type": "seq2seq"},
                "bootstrap": {"enabled": False},
            },
            "auto": {"tier": "balanced"},
        }
    )

    metrics, _eval_windows = rem.compute_real_metrics(
        runner,
        _FakeModel(),
        calibration_data=[{"input_ids": [1]}, {"input_ids": [2]}],
        adapter=object(),
        preview_n=1,
        final_n=1,
        config=config,
    )

    assert metrics["primary_metric"]["kind"] == "ppl_seq2seq"
    summary = metrics["preview_final_slice_delta_summary"]
    assert summary["basis"] == "independent_disjoint_slices"
    assert summary["paired"] is False
    assert summary["ci_method"] == "none"
    assert summary["ci_reason"] == "bootstrap_disabled"
    assert any(
        operation == "bootstrap_coverage_warning"
        for _, operation, _, _ in runner.events
    )


def test_compute_real_metrics_marks_primary_metric_invalid_on_ratio_ci_inconsistency(
    monkeypatch,
) -> None:
    preview_summary = _summary(
        ppl=2.0,
        total_tokens=2,
        weighted_log_loss=2 * math.log(2.0),
        num_batches=2,
        log_losses=[math.log(2.0), math.log(2.2)],
        window_ids=[0, 1],
        token_counts=[1, 1],
        actual_token_counts=[1, 1],
    )
    final_summary = _summary(
        ppl=3.0,
        total_tokens=2,
        weighted_log_loss=2 * math.log(3.0),
        num_batches=2,
        log_losses=[math.log(3.0), math.log(3.5)],
        window_ids=[2, 3],
        token_counts=[1, 1],
        actual_token_counts=[1, 1],
    )

    def fake_slice_summary(*_args, start_idx: int, **_kwargs):
        return (preview_summary, None) if start_idx == 0 else (final_summary, None)

    monkeypatch.setattr(rem, "compute_slice_summary", fake_slice_summary)
    monkeypatch.setattr(rem, "measure_latency", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        rem,
        "compute_window_pairing_metrics",
        lambda **_kwargs: {
            "preview": {"matched": 2, "expected": 2, "reason": None},
            "final": {"matched": 2, "expected": 2, "reason": None},
            "match_fraction": 1.0,
            "overlap_fraction": 0.0,
            "duplicate_fraction": 0.0,
            "count_mismatch": False,
            "reason": None,
        },
    )
    monkeypatch.setattr(
        rem,
        "assess_bootstrap_coverage",
        lambda **_kwargs: {
            "preview_required": 1,
            "final_required": 1,
            "replicates_required": 1,
            "preview_ok": True,
            "final_ok": True,
            "replicates_ok": True,
            "coverage": {},
        },
    )

    metrics, _eval_windows = rem.compute_real_metrics(
        _FakeRunner(),
        _FakeModel(),
        calibration_data=[{"input_ids": [1]}] * 4,
        adapter=object(),
        preview_n=2,
        final_n=2,
        config=SimpleNamespace(
            context={"eval": {"bootstrap": {"enabled": True, "replicates": 4}}}
        ),
        compute_independent_delta_log_ci_fn=lambda *_args, **_kwargs: (0.1, 0.2),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (999.0, 999.0),
    )

    assert metrics["primary_metric"]["invalid"] is True
    assert metrics["primary_metric"]["degraded_reason"] == "primary_metric_invalid"


def test_compute_real_metrics_preserves_invalid_non_finite_pm_and_delta(
    monkeypatch,
) -> None:
    preview_summary = _summary(
        ppl=float("nan"),
        total_tokens=1,
        weighted_log_loss=float("inf"),
        num_batches=1,
        log_losses=[0.1],
        window_ids=[0],
        token_counts=[1],
        actual_token_counts=[1],
    )
    final_summary = _summary(
        ppl=float("nan"),
        total_tokens=1,
        weighted_log_loss=float("inf"),
        num_batches=1,
        log_losses=[0.1],
        window_ids=[1],
        token_counts=[1],
        actual_token_counts=[1],
    )

    def fake_slice_summary(*_args, start_idx: int, **_kwargs):
        return (preview_summary, None) if start_idx == 0 else (final_summary, None)

    monkeypatch.setattr(rem, "compute_slice_summary", fake_slice_summary)
    monkeypatch.setattr(rem, "measure_latency", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        rem,
        "compute_window_pairing_metrics",
        lambda **_kwargs: {
            "preview": {"matched": 1, "expected": 1, "reason": None},
            "final": {"matched": 1, "expected": 1, "reason": None},
            "match_fraction": 1.0,
            "overlap_fraction": 0.0,
            "duplicate_fraction": 0.0,
            "count_mismatch": False,
            "reason": None,
        },
    )
    monkeypatch.setattr(
        rem,
        "assess_bootstrap_coverage",
        lambda **_kwargs: {
            "preview_required": 1,
            "final_required": 1,
            "replicates_required": 1,
            "preview_ok": True,
            "final_ok": True,
            "replicates_ok": True,
            "coverage": {},
        },
    )

    metrics, _eval_windows = rem.compute_real_metrics(
        _FakeRunner(),
        _FakeModel(),
        calibration_data=[{"input_ids": [1]}, {"input_ids": [2]}],
        adapter=object(),
        preview_n=1,
        final_n=1,
        config=SimpleNamespace(context={"eval": {"bootstrap": {"enabled": False}}}),
    )

    assert metrics["primary_metric"]["invalid"] is True
    assert metrics["primary_metric"]["preview"] is None
    assert metrics["primary_metric"]["final"] is None
    assert metrics["primary_metric"]["degraded_reason"] == "non_finite_pm"
    assert not math.isfinite(metrics["logloss_delta"])


def test_compute_real_metrics_marks_empty_slice_metrics_invalid(monkeypatch) -> None:
    preview_summary = _summary(
        ppl=float("nan"),
        total_tokens=0,
        weighted_log_loss=0.0,
        num_batches=0,
        log_losses=[],
        window_ids=[],
        token_counts=[],
        actual_token_counts=[],
    )
    final_summary = _summary(
        ppl=float("nan"),
        total_tokens=0,
        weighted_log_loss=0.0,
        num_batches=0,
        log_losses=[],
        window_ids=[],
        token_counts=[],
        actual_token_counts=[],
    )

    def fake_slice_summary(*_args, start_idx: int, **_kwargs):
        return (preview_summary, None) if start_idx == 0 else (final_summary, None)

    monkeypatch.setattr(rem, "compute_slice_summary", fake_slice_summary)
    monkeypatch.setattr(rem, "measure_latency", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        rem,
        "compute_window_pairing_metrics",
        lambda **_kwargs: {
            "preview": {"matched": 0, "expected": 0, "reason": None},
            "final": {"matched": 0, "expected": 0, "reason": None},
            "match_fraction": 1.0,
            "overlap_fraction": 0.0,
            "duplicate_fraction": 0.0,
            "count_mismatch": False,
            "reason": None,
        },
    )
    monkeypatch.setattr(
        rem,
        "assess_bootstrap_coverage",
        lambda **_kwargs: {
            "preview_required": 1,
            "final_required": 1,
            "replicates_required": 1,
            "preview_ok": True,
            "final_ok": True,
            "replicates_ok": True,
            "coverage": {},
        },
    )

    metrics, _eval_windows = rem.compute_real_metrics(
        _FakeRunner(),
        _FakeModel(),
        calibration_data=[{"input_ids": [1]}, {"input_ids": [2]}],
        adapter=object(),
        preview_n=1,
        final_n=1,
        config=SimpleNamespace(context={"eval": {"bootstrap": {"enabled": False}}}),
    )

    assert metrics["primary_metric"]["invalid"] is True
    assert metrics["primary_metric"]["degraded_reason"] == "non_finite_pm"


def test_compute_real_metrics_rejects_invalid_device_override(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            Process=lambda: SimpleNamespace(
                memory_info=lambda: SimpleNamespace(rss=0),
            )
        ),
    )

    with pytest.raises((RuntimeError, ValueError)):
        rem.compute_real_metrics(
            _FakeRunner(),
            _FakeModel(),
            calibration_data=[{"input_ids": [1]}],
            adapter=object(),
            preview_n=1,
            final_n=1,
            config=SimpleNamespace(
                context={"eval": {"device_override": "not-a-real-device"}}
            ),
        )


def test_compute_real_metrics_propagates_pairing_metric_failures(monkeypatch) -> None:
    preview_summary = _summary(
        ppl=2.0,
        total_tokens=1,
        weighted_log_loss=math.log(2.0),
        num_batches=1,
        log_losses=[math.log(2.0)],
        window_ids=[0],
        token_counts=[1],
        actual_token_counts=[1],
    )
    final_summary = _summary(
        ppl=3.0,
        total_tokens=1,
        weighted_log_loss=math.log(3.0),
        num_batches=1,
        log_losses=[math.log(3.0)],
        window_ids=[1],
        token_counts=[1],
        actual_token_counts=[1],
    )

    def fake_slice_summary(*_args, start_idx: int, **_kwargs):
        return (preview_summary, None) if start_idx == 0 else (final_summary, None)

    monkeypatch.setattr(rem, "compute_slice_summary", fake_slice_summary)
    monkeypatch.setattr(rem, "measure_latency", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        rem,
        "compute_window_pairing_metrics",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("pairing boom")),
    )

    with pytest.raises(RuntimeError, match="pairing boom"):
        rem.compute_real_metrics(
            _FakeRunner(),
            _FakeModel(),
            calibration_data=[{"input_ids": [1]}, {"input_ids": [2]}],
            adapter=object(),
            preview_n=1,
            final_n=1,
            config=SimpleNamespace(context={"eval": {"bootstrap": {"enabled": False}}}),
        )
