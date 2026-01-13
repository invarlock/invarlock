from __future__ import annotations

import math

from invarlock.eval.primary_metric import compute_primary_metric_from_report


def test_compute_primary_metric_flags_nonfinite_preview() -> None:
    report = {
        "evaluation_windows": {
            "preview": {"logloss": [float("nan")], "token_counts": [1]},
            "final": {"logloss": [0.0], "token_counts": [1]},
        }
    }
    baseline = {
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": math.exp(0.0)}}
    }

    payload = compute_primary_metric_from_report(
        report, kind="ppl_causal", baseline=baseline
    )

    assert payload["invalid"] is True
    assert payload["degraded"] is True
    assert payload["degraded_reason"] == "non_finite_pm"
