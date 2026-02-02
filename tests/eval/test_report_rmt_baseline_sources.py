from invarlock.reporting.guards_analysis import _extract_rmt_analysis


def test_rmt_extracts_baseline_from_baseline_metrics_when_present():
    report = {
        "meta": {"auto": {"tier": "balanced"}},
        "guards": [
            {
                "name": "rmt",
                "policy": {"deadband": 0.1},
                "metrics": {
                    "edge_risk_by_family": {"ffn": 1.05},
                    "epsilon_by_family": {"ffn": 0.1},
                },
            }
        ],
    }
    baseline = {"rmt": {"edge_risk_by_family": {"ffn": 1.0}}}
    out = _extract_rmt_analysis(report, baseline)
    assert out["families"]["ffn"]["edge_base"] == 1.0
    assert out["families"]["ffn"]["edge_cur"] == 1.05


def test_rmt_extracts_baseline_from_guard_baseline_metrics_source3():
    report = {
        "meta": {"auto": {"tier": "balanced"}},
        "guards": [
            {
                "name": "rmt",
                "policy": {"deadband": 0.1},
                "metrics": {
                    "edge_risk_by_family_base": {"ffn": 1.2},
                    "edge_risk_by_family": {"ffn": 1.25},
                },
            }
        ],
    }
    baseline = {"rmt": {"edge_risk_by_family": {"ffn": 1.0}}}
    out = _extract_rmt_analysis(report, baseline)
    assert out["families"]["ffn"]["edge_base"] == 1.2
    assert out["families"]["ffn"]["edge_cur"] == 1.25
