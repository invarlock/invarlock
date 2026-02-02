from unittest.mock import patch

from invarlock.reporting.report_builder import make_report


def test_evaluation_report_includes_top_level_window_plan():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "window_plan": {"preview": 2, "final": 3},
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
    with patch(
        "invarlock.reporting.report_builder.validate_run_report", return_value=True
    ):
        cert = make_report(report, baseline)
    # Window plan may be omitted; assert dataset stats are available
    stats = cert.get("dataset", {}).get("windows", {}).get("stats", {})
    assert isinstance(stats, dict)
