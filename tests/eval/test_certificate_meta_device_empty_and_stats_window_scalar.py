from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_meta_device_empty_and_stats_window_plan_non_dict():
    report = {
        "meta": {
            "model_id": "m",
            "seed": 1,
            "device": "",
        },  # empty device → no telemetry device
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "stats": 123,  # non-dict → branch where stats mapping is skipped
            "window_plan": 456,  # non-dict → branch where plan is skipped
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
    # Telemetry should not include device key
    assert "telemetry" not in cert or "device" not in cert.get("telemetry", {})
    # window_plan should be omitted due to non-dict
    assert "window_plan" not in cert
