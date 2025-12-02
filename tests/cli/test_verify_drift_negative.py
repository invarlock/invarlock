from invarlock.cli.commands import verify as V


def test_validate_drift_band_negative():
    cert = {
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 12.0,  # out of [0.95, 1.05] drift band
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        }
    }
    errs = V._validate_drift_band(cert)
    assert errs and "out of band" in errs[0]
