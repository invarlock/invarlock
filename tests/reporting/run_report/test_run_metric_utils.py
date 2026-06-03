from __future__ import annotations

from invarlock.reporting.run_report_metrics_contract import (
    format_debug_metric_diffs,
    merge_primary_metric_health,
)


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

    merged = merge_primary_metric_health(primary_metric, core_primary_metric)

    assert merged["preview"] == primary_metric["preview"]
    assert merged["final"] == primary_metric["final"]
    assert merged["ratio_vs_baseline"] == primary_metric["ratio_vs_baseline"]
    assert merged["invalid"] is True
    assert merged["degraded"] is True
    assert merged["degraded_reason"] == "non_finite_pm"


def test_merge_primary_metric_health_returns_empty_for_non_mapping() -> None:
    assert merge_primary_metric_health(None, {"invalid": True}) == {}


def test_format_debug_metric_diffs_includes_ratio_vs_baseline_fallback() -> None:
    pm = {"final": 12.0, "preview": 11.0, "ratio_vs_baseline": 2.4}
    metrics = {"primary_metric": {"final": 10.0, "preview": 9.0}}
    baseline = {"metrics": {"primary_metric": {"final": 5.0}}}

    out = format_debug_metric_diffs(pm, metrics, baseline)
    assert "final: v1-v1 = +2.000000000" in out
    assert "preview: v1-v1 = +2.000000000" in out
    assert "ratio_vs_baseline: v1-v1 = +0.400000000" in out


def test_format_debug_metric_diffs_skips_log_terms_on_domain_error() -> None:
    pm = {"final": -1.0, "preview": 11.0}
    metrics = {"primary_metric": {"final": 10.0, "preview": 9.0}}

    out = format_debug_metric_diffs(pm, metrics, baseline_report_data=None)
    assert "final: v1-v1 = -11.000000000" in out
    assert "Δlog(final)" not in out


def test_format_debug_metric_diffs_handles_bad_numeric_inputs() -> None:
    pm = {"final": "bad", "preview": "bad", "ratio_vs_baseline": "bad"}
    metrics = {"primary_metric": {"final": "bad", "preview": "bad"}}
    baseline = {"metrics": {"primary_metric": {"final": "bad"}}}

    assert format_debug_metric_diffs(pm, metrics, baseline) == ""


def test_format_debug_metric_diffs_handles_non_mapping_primary_metric_block() -> None:
    pm = {"final": 12.0, "preview": 11.0}
    metrics = {"primary_metric": None}

    out = format_debug_metric_diffs(pm, metrics, baseline_report_data=None)

    assert "final: v1-v1" not in out
    assert "preview: v1-v1" not in out


def test_format_debug_metric_diffs_ignores_missing_baseline_final_ratio_fallback() -> (
    None
):
    pm = {"final": 12.0, "preview": 11.0, "ratio_vs_baseline": 2.4}
    metrics = {
        "primary_metric": {
            "final": 10.0,
            "preview": 9.0,
            "ratio_vs_baseline": "bad",
        }
    }
    baseline = {"metrics": {"primary_metric": {}}}

    out = format_debug_metric_diffs(pm, metrics, baseline)

    assert "final: v1-v1 = +2.000000000" in out
    assert "preview: v1-v1 = +2.000000000" in out
    assert "ratio_vs_baseline: v1-v1" not in out
