from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import invarlock.core.runner_runtime.eval_metrics as rem
from tests.core._support_runner_eval_metrics import _FakeModel, _FakeRunner, _summary


def test_compute_slice_metrics_filters_non_finite_losses_and_marks_ratio_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview_summary = _summary(
        ppl=2.0,
        total_tokens=1,
        weighted_log_loss=0.1,
        num_batches=2,
        log_losses=[0.1, float("inf")],
        window_ids=[0, 1],
        token_counts=[1, 1],
        actual_token_counts=[1, 1],
    )
    final_summary = _summary(
        ppl=3.0,
        total_tokens=1,
        weighted_log_loss=0.2,
        num_batches=2,
        log_losses=[0.2, float("nan")],
        window_ids=[2, 3],
        token_counts=[1, 1],
        actual_token_counts=[1, 1],
    )

    def fake_slice_summary(*_args, start_idx: int, **_kwargs):
        return (preview_summary, None) if start_idx == 0 else (final_summary, None)

    exp_values = iter([1.1, 1.2, 2.0, 3.0])
    monkeypatch.setattr(rem, "compute_slice_summary", fake_slice_summary)
    monkeypatch.setattr(rem, "resolve_limit", lambda _data, limit: limit)
    monkeypatch.setattr(rem.math, "exp", lambda _value: next(exp_values))

    runner = _FakeRunner()
    result = rem._compute_slice_metrics(
        runner,
        _FakeModel(),
        SimpleNamespace(
            preview_data=[{"input_ids": [1]}],
            final_data=[{"input_ids": [2]}],
            preview_n=2,
            final_n=2,
            device="cpu",
            resolved_loss_mode="causal",
        ),
    )

    operations = [operation for _, operation, _, _ in runner.events]
    assert result.pm_invalid is True
    assert "non_finite_preview_losses_filtered" in operations
    assert "non_finite_final_losses_filtered" in operations
    assert "primary_metric_invalid" in operations


def test_compute_slice_metrics_logs_filtered_losses_and_ratio_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview_summary = _summary(
        ppl=2.0,
        total_tokens=2,
        weighted_log_loss=0.0,
        num_batches=2,
        log_losses=[0.0, float("inf")],
        window_ids=[0, 1],
        token_counts=[1, 1],
        actual_token_counts=[1, 1],
    )
    final_summary = _summary(
        ppl=3.0,
        total_tokens=2,
        weighted_log_loss=0.0,
        num_batches=2,
        log_losses=[0.0, float("nan")],
        window_ids=[2, 3],
        token_counts=[1, 1],
        actual_token_counts=[1, 1],
    )

    def fake_slice_summary(*_args, start_idx: int, **_kwargs):
        return (preview_summary, None) if start_idx == 0 else (final_summary, None)

    exp_values = iter([1.0, 1.0, 2.0, 3.0])
    runner = _FakeRunner()
    monkeypatch.setattr(rem, "compute_slice_summary", fake_slice_summary)
    monkeypatch.setattr(rem.math, "exp", lambda _value: next(exp_values))

    result = rem._compute_slice_metrics(
        runner,
        _FakeModel(),
        SimpleNamespace(
            preview_data=[{"input_ids": [1]}],
            final_data=[{"input_ids": [2]}],
            preview_n=2,
            final_n=2,
            device="cpu",
            resolved_loss_mode="causal",
        ),
    )

    assert result.pm_invalid is True

    events = [event[1] for event in runner.events]
    assert "non_finite_preview_losses_filtered" in events
    assert "non_finite_final_losses_filtered" in events
    assert "primary_metric_invalid" in events


def test_compute_slice_metrics_uses_finite_ppl_fallback_and_payload_skips_zero_weight_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _FakeRunner()
    ppl_values = iter([2.0, 3.0])
    monkeypatch.setattr(rem, "resolve_limit", lambda _data, limit: limit)
    monkeypatch.setattr(
        rem,
        "compute_slice_summary",
        lambda *_args, start_idx, **_kwargs: (
            _summary(
                ppl=next(ppl_values),
                total_tokens=0,
                weighted_log_loss=0.0,
                num_batches=0,
                log_losses=[],
                window_ids=[],
                token_counts=[],
                actual_token_counts=[],
            ),
            None,
        ),
    )
    monkeypatch.setattr(rem, "measure_latency", lambda *_args, **_kwargs: 0.0)
    result = rem._compute_slice_metrics(
        runner,
        _FakeModel(),
        SimpleNamespace(
            preview_data=[],
            final_data=[],
            preview_n=0,
            final_n=0,
            device="cpu",
            resolved_loss_mode="causal",
        ),
    )

    metrics, _windows = rem._build_real_metrics_payload(
        _FakeModel(),
        rem._EvalRuntimeContext(
            device="cpu",
            process=SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0)),
            initial_memory=0.0,
            preview_data=[],
            final_data=[],
            preview_n=0,
            final_n=0,
            resolved_loss_mode="causal",
            bootstrap_enabled=False,
            bootstrap_method="bca",
            bootstrap_replicates=0,
            bootstrap_alpha=0.05,
            bootstrap_seed=0,
            ci_band=0.95,
            single_method="bca",
            delta_method="paired",
            profile_label="dev",
            pairing_context={},
        ),
        result,
        SimpleNamespace(
            pm_invalid=False,
            degraded_reason=None,
            preview_log_ci=(0.0, 0.0),
            final_log_ci=(0.0, 0.0),
            delta_log_ci=(0.0, 0.0),
            delta_ci_method="none",
            delta_ci_reason="bootstrap_disabled",
            degenerate_delta=False,
            degenerate_reason=None,
        ),
        SimpleNamespace(
            window_overlap_fraction=0.0,
            window_match_fraction=1.0,
            pairing_reason=None,
            preview_pair_stats={"matched": 0, "expected": 0, "reason": None},
            final_pair_stats={"matched": 0, "expected": 0, "reason": None},
            bootstrap_info={
                "preview_required": 0,
                "final_required": 0,
                "replicates_required": 0,
                "preview_ok": True,
                "final_ok": True,
                "replicates_ok": True,
                "coverage": {},
            },
        ),
    )

    assert result.pm_invalid is False
    assert result.pm_preview == 2.0
    assert result.pm_final == 3.0
    summary = metrics["preview_final_slice_delta_summary"]
    assert summary["mean"] == pytest.approx(math.log(3.0 / 2.0))
    assert summary["basis"] == "independent_disjoint_slices"
    assert summary["paired"] is False
    assert "paired_delta_summary" not in metrics
