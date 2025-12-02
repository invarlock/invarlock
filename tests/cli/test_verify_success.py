from invarlock.cli.commands import verify as V


def test_verify_helpers_success_paths():
    cert = {
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "preview": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": (1.0, 1.0),
        },
        "baseline_ref": {"primary_metric": {"final": 10.0}},
        "dataset": {
            "windows": {
                "preview": 2,
                "final": 2,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
                    "paired_windows": 2,
                },
            }
        },
    }

    assert V._validate_primary_metric(cert) == []
    assert V._validate_counts(cert) == []
    assert V._validate_drift_band(cert) == []
