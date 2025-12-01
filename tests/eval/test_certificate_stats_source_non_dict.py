from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_metrics_stats_source_non_dict_is_ignored():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            # Provide a non-dict stats to exercise the guard
            "stats": ["not", "a", "dict"],
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
    stats = (cert.get("dataset", {}).get("windows", {}) or {}).get("stats", {})
    # Ensure none of the optional passthrough keys were copied
    for k in (
        "requested_preview",
        "requested_final",
        "actual_preview",
        "actual_final",
        "coverage_ok",
    ):
        assert k not in stats
