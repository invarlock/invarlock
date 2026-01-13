from __future__ import annotations

from invarlock.cli.commands import run as run_mod


def test_merge_primary_metric_health_prefers_core_flags() -> None:
    primary_metric = {
        "kind": "ppl_causal",
        "preview": 1.0,
        "final": 2.0,
        "ratio_vs_baseline": 2.0,
        "invalid": False,
        "degraded": False,
    }
    core_primary_metric = {
        "preview": None,
        "final": None,
        "invalid": True,
        "degraded": True,
        "degraded_reason": "non_finite_pm",
    }

    merged = run_mod._merge_primary_metric_health(primary_metric, core_primary_metric)

    assert merged["preview"] == primary_metric["preview"]
    assert merged["final"] == primary_metric["final"]
    assert merged["ratio_vs_baseline"] == primary_metric["ratio_vs_baseline"]
    assert merged["invalid"] is True
    assert merged["degraded"] is True
    assert merged["degraded_reason"] == "non_finite_pm"
