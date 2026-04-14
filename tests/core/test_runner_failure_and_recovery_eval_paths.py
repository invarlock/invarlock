from __future__ import annotations

from typing import Any

import pytest

from invarlock.core.api import RunConfig, RunReport
from invarlock.core.runner import CoreRunner
from tests.core.test_runner_failure_and_recovery_metrics import (
    DummyAdapter,
    _toy_model_with_losses,
)

## Removed flaky negative request coverage assertion; coverage for _resolve_limit
## with non-positive requests is exercised by other tests (preview/final zero cases).


# Removed: Indexable debug snapshot that caused slicing issues outside safeguards


def test_tail_paired_baseline_weights(monkeypatch):
    tail_calls: dict[str, Any] = {}

    def fake_tail_eval(*, deltas, weights=None, policy=None):
        tail_calls["weights"] = weights
        return {"mean": 0.0, "evaluated": True, "passed": True, "mode": "warn"}

    monkeypatch.setattr("invarlock.core.runner.evaluate_metric_tail", fake_tail_eval)

    def fake_compute_real_metrics(*args, **kwargs):
        metrics = {
            "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.4}
        }
        eval_windows = {
            "final": {
                "window_ids": [1],
                "logloss": [0.6],
                "token_counts": [3],
            },
            "preview": {"window_ids": [0], "logloss": [0.5], "token_counts": [2]},
        }
        return metrics, eval_windows

    monkeypatch.setattr(
        CoreRunner, "_compute_real_metrics", staticmethod(fake_compute_real_metrics)
    )

    runner = CoreRunner()
    adapter = DummyAdapter()
    report = RunReport()
    cfg = RunConfig(
        context={
            "baseline_eval_windows": {
                "final": {"window_ids": [1], "logloss": [0.4], "token_counts": [3]}
            }
        }
    )

    metrics = runner._eval_phase(
        model=object(),
        adapter=adapter,
        calibration_data=[{"input_ids": [1, 2, 3]}],
        report=report,
        preview_n=1,
        final_n=1,
        config=cfg,
    )

    assert metrics["primary_metric_tail"]["source"] == "paired_baseline.final"
    assert tail_calls.get("weights") == [3.0]


def test_tail_token_count_conversion_error(monkeypatch):
    tail_calls: dict[str, Any] = {}

    def fake_tail_eval(*, deltas, weights=None, policy=None):
        tail_calls["weights"] = weights
        return {"mean": 0.0, "evaluated": True, "passed": True, "mode": "warn"}

    monkeypatch.setattr("invarlock.core.runner.evaluate_metric_tail", fake_tail_eval)

    def fake_compute_real_metrics(*args, **kwargs):
        metrics = {
            "primary_metric": {"kind": "ppl_causal", "preview": 1.0, "final": 1.4}
        }
        eval_windows = {
            "final": {
                "window_ids": [1],
                "logloss": [0.6],
                "token_counts": ["bad"],
            },
            "preview": {"window_ids": [0], "logloss": [0.5], "token_counts": [2]},
        }
        return metrics, eval_windows

    monkeypatch.setattr(
        CoreRunner, "_compute_real_metrics", staticmethod(fake_compute_real_metrics)
    )

    runner = CoreRunner()
    adapter = DummyAdapter()
    report = RunReport()
    cfg = RunConfig(
        context={
            "baseline_eval_windows": {
                "final": {"window_ids": [1], "logloss": [0.4], "token_counts": ["bad"]}
            }
        }
    )

    with pytest.raises(ValueError, match="could not convert string to float"):
        runner._eval_phase(
            model=object(),
            adapter=adapter,
            calibration_data=[{"input_ids": [1, 2, 3]}],
            report=report,
            preview_n=1,
            final_n=1,
            config=cfg,
        )


def test_soft_eval_error_warns_not_raises():
    runner = CoreRunner()
    model = _toy_model_with_losses([0.1, 0.2])
    adapter = DummyAdapter()
    cfg = RunConfig(context={"eval": {"loss": {"type": "mlm"}, "strict": False}})
    calibration = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [0, 0, 0],
            "labels": [1, 1, 1],
        }
    ]

    report = RunReport()

    metrics = runner._eval_phase(
        model,
        adapter,
        calibration,
        report,
        preview_n=1,
        final_n=0,
        config=cfg,
    )

    eval_error = metrics.get("eval_error") or {}
    assert eval_error.get("error") == "mlm_missing_masks"
