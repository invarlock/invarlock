from __future__ import annotations

from invarlock.cli.commands.run import _normalize_overhead_result as _norm


def test_overhead_nan_marks_not_evaluated_and_passes() -> None:
    # Missing/NaN overhead should be informational and not evaluated
    payload = {
        "passed": False,
        "overhead_ratio": float("nan"),
        "overhead_percent": None,
        "evaluated": True,
    }
    out = _norm(dict(payload), profile="dev")
    assert out["evaluated"] is False
    assert out["passed"] is True
