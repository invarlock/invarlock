from __future__ import annotations

import math

from invarlock.eval.primary_metric import MetricContribution, get_metric


def test_ppl_display_transform_overflow_to_inf():
    m = get_metric("ppl_causal")
    assert math.isinf(m.display_transform(1000.0))


def test_ppl_point_from_windows_nan_when_no_weight():
    m = get_metric("ppl_causal")
    y = m.point_from_windows(windows={"logloss": [2.0], "token_counts": [0]})
    assert math.isnan(y)


def test_ppl_mlm_uses_masked_token_counts():
    m = get_metric("ppl_mlm")
    out = m.point_from_windows(
        windows={
            "logloss": [2.302585093],
            "masked_token_counts": [1],
            "token_counts": [0],
        }
    )
    assert abs(out - 10.0) < 1e-6


def test_accuracy_point_from_windows_variants():
    acc = get_metric("accuracy")
    # Per-example flags
    out1 = acc.point_from_windows(windows={"example_correct": [1, 0, 1, 0]})
    assert abs(out1 - 0.5) < 1e-9
    # Aggregate counts with policy (ties as correct, exclude abstain)
    out2 = acc.point_from_windows(
        windows={
            "correct_total": 8,
            "total": 12,
            "abstain_total": 2,
            "ties_total": 1,
            "policy": {"exclude_abstain": True, "ties_count_as_correct": True},
        }
    )
    # Denominator 12-2=10; ties add 1 → 9/10
    assert abs(out2 - 0.9) < 1e-9


def test_accuracy_paired_compare_shapes_and_transform():
    acc = get_metric("accuracy")
    subj = [MetricContribution(1), MetricContribution(0), MetricContribution(1)]
    base = [MetricContribution(1), MetricContribution(1), MetricContribution(0)]
    res = acc.paired_compare(subj, base, reps=200, seed=0, ci_level=0.90)
    assert {"delta", "ci", "display", "display_ci"} <= set(res.keys())
    # display is in percentage points
    assert abs(res["display"] - (res["delta"] * 100.0)) < 1e-6


def test_ppl_paired_compare_shapes():
    ppl = get_metric("ppl_causal")
    subj = [MetricContribution(2.2, 5), MetricContribution(2.0, 2)]
    base = [MetricContribution(2.0, 5), MetricContribution(2.0, 2)]
    res = ppl.paired_compare(subj, base, reps=100, seed=0, ci_level=0.90)
    assert {"delta", "ci", "display", "display_ci"} <= set(res.keys())
    assert isinstance(res["ci"], tuple) and len(res["ci"]) == 2
