from __future__ import annotations

from invarlock.eval.primary_metric import MetricContribution, get_metric


def test_ppl_paired_compare_reps_and_ci_level_echo():
    ppl = get_metric("ppl_causal")
    subj = [MetricContribution(2.2, 1), MetricContribution(2.0, 1)]
    base = [MetricContribution(2.0, 1), MetricContribution(2.0, 1)]
    res = ppl.paired_compare(subj, base, reps=123, seed=7, ci_level=0.90)
    assert res["reps"] == 123 and abs(res["ci_level"] - 0.90) < 1e-12
