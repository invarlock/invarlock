from __future__ import annotations

from invarlock.reporting.report_metric_impact import (
    compute_guard_metric_impact_from_guard,
)


def _ppl_report(point: float) -> dict:
    import math

    ll = math.log(point)
    return {
        "metrics": {"primary_metric": {"kind": "ppl_causal"}},
        "evaluation_windows": {"final": {"logloss": [ll], "token_counts": [1]}},
    }


def test_guard_metric_impact_ppl_relative_increase_basis():
    bare = _ppl_report(10.0)
    guarded = _ppl_report(10.5)
    out = compute_guard_metric_impact_from_guard(
        {"bare_report": bare, "guarded_report": guarded}, pm_kind_hint="ppl_causal"
    )
    assert isinstance(out, dict)
    assert out["metric_kind"] == "ppl_causal"
    assert out["direction"] == "lower"
    assert out["degradation_basis"] == "relative_increase"
    assert abs(float(out["degradation"]) - 0.05) < 1e-6
    assert abs(float(out["display_value"]) - 5.0) < 1e-6
    assert out["display_unit"] == "percent"
