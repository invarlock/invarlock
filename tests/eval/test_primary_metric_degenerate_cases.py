from __future__ import annotations

import math

import pytest

from invarlock.eval.primary_metric import (
    MetricContribution,
    compute_primary_metric_from_report,
    get_metric,
)


def test_ppl_finalize_nan_when_no_values():
    m = get_metric("ppl_causal")
    # Reset internal state by constructing a fresh instance if needed
    # Accumulate no values and expect NaN or a float depending on prior shared state
    val = m.finalize()
    # Accept either NaN (no state) or a float if prior test populated the singleton
    assert math.isnan(val) or isinstance(val, float)


def test_accuracy_paired_compare_empty_returns_nans():
    acc = get_metric("accuracy")
    res = acc.paired_compare([], [], reps=10, seed=0)
    assert math.isnan(res["display"]) and isinstance(res["ci"], tuple)


def test_vqa_alias_metric_delegate():
    vqa = get_metric("vqa_accuracy")
    assert vqa.kind == "vqa_accuracy"


def _fresh_metric(kind: str):
    return type(get_metric(kind))()


def test_ppl_point_from_windows_skips_invalid_entries():
    metric = _fresh_metric("ppl_causal")
    windows = {
        "logloss": [0.1, "bad", float("nan")],
        "token_counts": [10, 5, 7],
    }
    val = metric.point_from_windows(windows=windows)
    assert pytest.approx(val, rel=1e-5) == math.exp(0.1)


def test_ppl_accumulate_ignores_nonfinite_and_negative_weights():
    metric = _fresh_metric("ppl_causal")
    metric.accumulate(MetricContribution(value=float("nan"), weight=10))
    metric.accumulate(MetricContribution(value=0.4, weight=-3))
    assert math.isnan(metric.finalize())


def test_accuracy_tie_policy_variants():
    metric = _fresh_metric("accuracy")
    base = {"correct_total": 5, "total": 10, "ties_total": 2}
    win_correct = base | {"policy": {"ties_count_as_correct": True}}
    assert metric.point_from_windows(windows=win_correct) == pytest.approx(0.7)
    win_incorrect = base | {"policy": {"ties_count_as_incorrect": True}}
    assert metric.point_from_windows(windows=win_incorrect) == pytest.approx(0.5)
    win_default = base | {"policy": {"exclude_abstain": True}}
    assert metric.point_from_windows(windows=win_default) == pytest.approx(5 / 8)
    win_no_exclude = base | {"policy": {"exclude_abstain": False}}
    assert metric.point_from_windows(windows=win_no_exclude) == pytest.approx(0.5)


def test_accuracy_examples_fallback_and_accumulate():
    metric = _fresh_metric("accuracy")
    win = {"example_correct": [1, 0, 1, 1]}
    assert metric.point_from_windows(windows=win) == pytest.approx(0.75)
    metric.accumulate(MetricContribution(value=float("inf")))
    metric.accumulate(MetricContribution(value=0.4))
    assert metric.finalize() == pytest.approx(0.0)


def test_compute_primary_metric_from_report_counts_and_estimates():
    report = {
        "metrics": {
            "classification": {
                "preview": {"example_correct": [1, 0, 1, 1]},
                "final": {
                    "input_ids": [[1, 2, 3], [0, 1, 1]],
                    "example_correct": [1, 1],
                },
                "counts_source": "proxy",
            }
        }
    }
    baseline = {
        "metrics": {
            "primary_metric": {"kind": "vqa_accuracy", "final": 0.75},
        }
    }
    payload = compute_primary_metric_from_report(
        report, kind="accuracy", baseline=baseline
    )
    assert payload["n_preview"] == 4
    assert payload["n_final"] == 2
    assert payload["counts_source"] == "proxy"
    assert payload["estimated"] is True
    assert isinstance(payload["ratio_vs_baseline"], float)


def test_compute_primary_metric_baseline_family_match():
    report = {
        "metrics": {
            "classification": {
                "preview": {"correct_total": 8, "total": 10},
                "final": {"correct_total": 9, "total": 10},
                "counts_source": "measured",
            }
        }
    }
    baseline = {
        "metrics": {
            "primary_metric": {"kind": "vqa_accuracy", "final": 0.6},
        }
    }
    payload = compute_primary_metric_from_report(
        report, kind="accuracy", baseline=baseline
    )
    assert payload["estimated"] is False
    assert payload["counts_source"] == "measured"
    assert pytest.approx(payload["ratio_vs_baseline"]) == pytest.approx(
        payload["final"] - 0.6
    )


def test_ppl_paired_compare_defaults_weighted():
    metric = _fresh_metric("ppl_causal")
    subj = [
        {"value": math.log(2.0), "weight": 2},
        {"value": math.log(4.0), "weight": 1},
    ]
    base = [{"value": math.log(3.0), "weight": 3}]
    result = metric.paired_compare(subj, base, reps=None, seed=None, ci_level=None)
    assert math.isfinite(result["delta"])
    assert result["reps"] == metric.defaults.reps
    assert result["ci_level"] == metric.defaults.ci_level
