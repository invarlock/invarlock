from __future__ import annotations

from invarlock.reporting.certificate import _compute_quality_overhead_from_guard


def _ppl_report(point: float) -> dict:
    import math

    ll = math.log(point)
    return {
        "metrics": {"primary_metric": {"kind": "ppl_causal"}},
        "evaluation_windows": {"final": {"logloss": [ll], "token_counts": [1]}},
    }


def test_quality_overhead_ppl_ratio_basis():
    bare = _ppl_report(10.0)
    guarded = _ppl_report(10.5)
    out = _compute_quality_overhead_from_guard(
        {"bare_report": bare, "guarded_report": guarded}, pm_kind_hint="ppl_causal"
    )
    assert isinstance(out, dict)
    assert out.get("basis") == "ratio"
    assert abs(float(out.get("value", 0.0)) - 1.05) < 1e-6
