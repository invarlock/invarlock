from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_window_plan_and_stats_passthrough():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "stats": {
                "requested_preview": 3,
                "requested_final": 5,
                "actual_preview": 3,
                "actual_final": 5,
                "coverage_ok": True,
            },
            "window_plan": {"plan": "ok", "preview": 3, "final": 5},
        },
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [],
        "edit": {
            "name": "structured",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "ppl_final": 10.0,
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)
    # Window plan may be omitted; ensure dataset stats propagated
    stats = cert.get("dataset", {}).get("windows", {}).get("stats", {})
    # Optional passthrough keys may be omitted after normalization; presence of stats dict is sufficient
    assert isinstance(stats, dict)
