from invarlock.reporting.guards_analysis import _extract_rmt_analysis


def test_rmt_extracts_baseline_from_baseline_metrics_when_present():
    report = {
        "meta": {"auto": {"tier": "balanced"}},
        "guards": [
            {
                "name": "rmt",
                "policy": {"deadband": 0.1},
                "metrics": {
                    "rmt_outliers": 2,
                    "outliers_per_family": {"ffn": 1},
                    "baseline_outliers_per_family": {"ffn": 1},
                    "flagged_rate": 0.1,
                    "max_mp_ratio_final": 1.2,
                    "mean_mp_ratio_final": 1.05,
                },
            }
        ],
    }
    # Baseline provides ratios via baseline.metrics.rmt
    baseline = {
        "metrics": {"rmt": {"max_mp_ratio_final": 1.1, "mean_mp_ratio_final": 1.0}}
    }
    out = _extract_rmt_analysis(report, baseline)
    # mean_deviation_ratio should reflect ratio vs baseline.mean_mp_ratio_final
    assert out["mean_deviation_ratio"] == 1.05
    assert out["max_deviation_ratio"] == 1.2 / 1.1


def test_rmt_extracts_baseline_from_guard_baseline_metrics_source3():
    report = {
        "meta": {"auto": {"tier": "balanced"}},
        "guards": [
            {
                "name": "rmt",
                "policy": {"deadband": 0.1},
                "metrics": {
                    "rmt_outliers": 1,
                    "outliers_per_family": {"ffn": 1},
                    "baseline_outliers_per_family": {"ffn": 0},
                    "flagged_rate": 0.1,
                    "max_mp_ratio_final": 1.5,
                    "mean_mp_ratio_final": 1.2,
                },
                "baseline_metrics": {"max_mp_ratio": 1.25, "mean_mp_ratio": 1.0},
            }
        ],
    }
    baseline = {}
    out = _extract_rmt_analysis(report, baseline)
    assert out["max_deviation_ratio"] == 1.5 / 1.25
    assert out["mean_deviation_ratio"] == 1.2 / 1.0
