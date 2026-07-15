from __future__ import annotations

from invarlock.reporting.report_metric_impact import (
    normalize_guard_metric_impact_result as _norm,
)


def test_nan_degradation_marks_not_evaluated_and_fails_closed() -> None:
    # Missing/NaN degradation is not evaluated and cannot substantiate a pass.
    payload = {
        "passed": False,
        "degradation": float("nan"),
        "display_value": None,
        "evaluated": True,
    }
    out = _norm(dict(payload))
    assert out["evaluated"] is False
    assert out["passed"] is False
