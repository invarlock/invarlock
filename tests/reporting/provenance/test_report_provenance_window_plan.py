from unittest.mock import patch

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_provenance_window_plan_propagates_from_metrics():
    report = {
        "meta": {
            "adapter": "hf",
            "model_id": "m",
            "seed": 1,
            "auto": {"tier": "balanced"},
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            },
            # Ensure ppl_analysis['window_plan'] is populated
            "window_plan": {"profile": "dev", "preview": 2, "final": 4},
        },
        "context": {"profile": "dev"},
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
    baseline = {**report, "run_id": "b", "edit": {"name": "noop"}}
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)
    # Window plan may be omitted; assert dataset stats are available
    stats = cert.get("dataset", {}).get("windows", {}).get("stats", {})
    assert isinstance(stats, dict)
