from __future__ import annotations

import math

import pytest

from invarlock.eval import primary_metric as pm_mod


def test_ppl_causal_point_and_finalize():
    metric = pm_mod._PPLCausal()
    windows = {
        "logloss": [0.0, math.log(2.0)],
        "token_counts": [1, 1],
    }
    point = metric.point_from_windows(windows=windows)
    assert point == pytest.approx(math.sqrt(2.0))

    metric.accumulate(pm_mod.MetricContribution(value=0.1, weight=2))
    metric.accumulate(pm_mod.MetricContribution(value=0.2, weight=2))
    assert metric.finalize() == pytest.approx(math.exp(0.15))


def test_ppl_causal_paired_compare(monkeypatch):
    monkeypatch.setattr(
        pm_mod,
        "compute_paired_delta_log_ci",
        lambda *args, **kwargs: (0.1, 0.2),
        raising=False,
    )
    metric = pm_mod._PPLCausal()
    subject = [pm_mod.MetricContribution(value=0.3, weight=1)]
    baseline = [pm_mod.MetricContribution(value=0.1, weight=1)]
    result = metric.paired_compare(
        subject,
        baseline,
        reps=10,
        seed=1,
        ci_level=0.9,
    )
    assert result["ci"] == (0.1, 0.2)
    assert result["display_ci"][0] == pytest.approx(math.exp(0.1))


def test_ppl_mlm_prefers_masked_counts():
    metric = pm_mod._PPLMLM()
    windows = {
        "logloss": [0.0, math.log(2.0)],
        "token_counts": [1, 1000],
        "masked_token_counts": [1, 1],
    }
    point = metric.point_from_windows(windows=windows)
    assert point == pytest.approx(math.sqrt(2.0))


def test_accuracy_point_from_windows_with_policy():
    metric = pm_mod._Accuracy()
    windows = {
        "correct_total": 8,
        "total": 10,
        "abstain_total": 2,
        "policy": {"exclude_abstain": True},
    }
    assert metric.point_from_windows(windows=windows) == pytest.approx(1.0)

    windows = {
        "correct_total": 8,
        "total": 10,
        "abstain_total": 2,
        "ties_total": 1,
        "policy": {"exclude_abstain": True, "ties_count_as_correct": True},
    }
    assert metric.point_from_windows(windows=windows) == pytest.approx(9 / 8)


def test_accuracy_accumulate_and_paired_compare():
    metric = pm_mod._Accuracy()
    metric.accumulate(pm_mod.MetricContribution(value=1.0))
    metric.accumulate(pm_mod.MetricContribution(value=0.0))
    assert metric.finalize() == pytest.approx(0.5)

    subject = [
        pm_mod.MetricContribution(value=1.0),
        pm_mod.MetricContribution(value=0.0),
    ]
    baseline = [
        pm_mod.MetricContribution(value=0.0),
        pm_mod.MetricContribution(value=0.0),
    ]
    result = metric.paired_compare(subject, baseline, reps=100, seed=0, ci_level=0.9)
    assert result["delta"] >= 0
    assert result["display"] == pytest.approx(result["delta"] * 100.0)


def test_get_metric_alias_lookup():
    metric = pm_mod.get_metric("vqa_accuracy")
    assert metric.kind == "vqa_accuracy"

    with pytest.raises(KeyError):
        pm_mod.get_metric("unknown_metric")
