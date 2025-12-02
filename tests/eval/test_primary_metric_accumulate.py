from __future__ import annotations

from invarlock.eval.primary_metric import MetricContribution, get_metric


def test_ppl_accumulate_and_finalize_weighted():
    m = get_metric("ppl_causal")
    m.accumulate(MetricContribution(value=2.0, weight=2))
    m.accumulate(MetricContribution(value=1.0, weight=1))
    val = m.finalize()
    # Weighted mean logloss = (2*2 + 1*1)/3 = 5/3; exp(5/3) ≈ 5.294
    assert abs(val - 5.294) < 0.1
