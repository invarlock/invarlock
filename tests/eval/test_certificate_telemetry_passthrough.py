from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_certificate_telemetry_fields_and_device_default():
    report = {
        "meta": {"model_id": "m", "seed": 1, "device": "cpu"},
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "latency_ms_per_tok": 1.23,
            "memory_mb_peak": 456,
            "throughput_tok_per_s": 789.0,
            "preview_total_tokens": 12,
            "final_total_tokens": 34,
            "masked_tokens_total": 0,
            "masked_tokens_preview": 0,
            "masked_tokens_final": 0,
            "edge_device": {"name": "mps", "available": False},
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
    tel = cert.get("telemetry", {})
    assert tel.get("latency_ms_per_tok") == 1.23
    assert tel.get("memory_mb_peak") == 456.0
    # throughput key may be omitted
    assert tel.get("preview_total_tokens") == 12.0
    assert tel.get("final_total_tokens") == 34.0
    # masked token totals may be omitted
    # edge_device and device are surfaced in meta or omitted; do not assert under telemetry
    assert cert.get("meta", {}).get("device") == "cpu"
