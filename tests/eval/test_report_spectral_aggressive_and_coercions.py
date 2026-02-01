from invarlock.reporting.guards_analysis import _extract_spectral_analysis


def test_extract_spectral_analysis_aggressive_tier_and_coercions():
    # Report with spectral guard metrics and policy; tier aggressive alters defaults path
    report = {
        "meta": {"auto": {"tier": "aggressive"}},
        "guards": [
            {
                "name": "spectral",
                "policy": {
                    "family_caps": {"ffn": {"kappa": 2.7}},
                    "correction_enabled": True,
                    "max_spectral_norm": 42.0,
                },
                "metrics": {
                    "caps_applied": "2",  # coercion to int
                    "max_caps": "5",
                    "modules_checked": "10",
                    "family_z_summary": {
                        "ffn": {"max": 3.4, "violations": 2, "count": 5}
                    },
                },
            }
        ],
    }
    baseline = {
        "metrics": {
            "spectral": {
                "max_spectral_norm_final": 10.0,
                "mean_spectral_norm_final": 5.0,
            }
        }
    }
    out = _extract_spectral_analysis(report, baseline)
    assert out["caps_applied"] == 2 and out["summary"]["max_caps"] == 5
    assert out["summary"]["modules_checked"] == 10
    assert out["summary"]["status"] in {"capped", "stable"}
