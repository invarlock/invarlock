from __future__ import annotations

import math

import pytest

from invarlock.core.exceptions import ValidationError
from invarlock.eval.primary_metric import (
    MetricContribution,
    _Accuracy,
    _PPLCausal,
    compute_primary_metric_from_report,
    get_metric,
    infer_binary_label_from_ids,
    validate_primary_metric_block,
)


def test_ppl_causal_point_from_windows_and_finalize() -> None:
    metric = _PPLCausal()
    windows = {"logloss": [1.0, 1.5], "token_counts": [4, 6]}
    ppl = metric.point_from_windows(windows=windows)
    assert ppl == pytest.approx(math.exp((1.0 * 4 + 1.5 * 6) / 10.0))

    metric.accumulate(MetricContribution(value=1.0, weight=4))
    metric.accumulate(MetricContribution(value=1.5, weight=6))
    assert metric.finalize() == pytest.approx(ppl)


def test_ppl_causal_paired_compare_uses_defaults() -> None:
    metric = _PPLCausal()
    subj = [MetricContribution(1.0, 1.0), MetricContribution(1.5, 1.0)]
    base = [MetricContribution(1.2, 1.0), MetricContribution(1.4, 1.0)]

    out = metric.paired_compare(subj, base, reps=None, seed=None, ci_level=None)
    assert out["kind"] == "ppl_causal"
    assert out["paired"] is True
    assert out["reps"] == metric.defaults.reps
    assert out["ci_level"] == metric.defaults.ci_level
    assert "delta" in out and "ci" in out and "display_ci" in out


def test_accuracy_point_from_windows_and_policies() -> None:
    metric = _Accuracy()
    # Per-example path: 2/3 correct
    win = {"example_correct": [1, 0, 1]}
    assert metric.point_from_windows(windows=win) == pytest.approx(2.0 / 3.0)

    # Aggregate path: abstain exclusion and ties handling
    win2 = {
        "correct_total": 8,
        "total": 12,
        "abstain_total": 2,
        "ties_total": 1,
        "policy": {
            "exclude_abstain": True,
            "ties_count_as_correct": True,
        },
    }
    acc = metric.point_from_windows(windows=win2)
    # total=12, abstain=2 -> 10; ties add 1 to correct → 9/10
    assert acc == pytest.approx(0.9)


def test_accuracy_point_from_windows_handles_bad_policy_safely() -> None:
    metric = _Accuracy()
    win = {
        "correct_total": 5,
        "total": 10,
        "policy": "not-a-dict",
        "abstain_total": "bad",
        "ties_total": "bad",
    }
    acc = metric.point_from_windows(windows=win)
    assert acc == pytest.approx(0.5)


def test_compute_primary_metric_from_report_empty_windows_returns_nan() -> None:
    payload = compute_primary_metric_from_report({}, kind="ppl_causal", baseline=None)
    assert math.isnan(payload["preview"])
    assert math.isnan(payload["final"])
    assert math.isnan(payload["ratio_vs_baseline"])


def test_validate_primary_metric_block_success_and_failure() -> None:
    block = {"preview": 1.0, "final": 2.0}
    assert validate_primary_metric_block(block) is block

    with pytest.raises(ValidationError):
        validate_primary_metric_block({"preview": "nan", "final": 2.0})


def test_get_metric_unknown_kind_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_metric("does-not-exist")


def test_infer_binary_label_from_ids_handles_negative_tokens() -> None:
    # Deterministic parity path should not fail on ints
    label = infer_binary_label_from_ids([-1, 2, 3])
    assert label in {0, 1}
