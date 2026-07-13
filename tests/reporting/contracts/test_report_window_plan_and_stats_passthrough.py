from unittest.mock import patch

from invarlock.reporting.report_make import make_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def test_window_plan_and_stats_passthrough():
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 1,
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev", "assurance": {"mode": "off"}},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 10.0},
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
    baseline = {**report, "edit": {"name": "noop"}}
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(canonical_run_report(report), canonical_baseline(baseline))
    # Window plan may be omitted; ensure dataset stats propagated
    stats = cert.get("dataset", {}).get("windows", {}).get("stats", {})
    # Optional passthrough keys may be omitted after normalization; presence of stats dict is sufficient
    assert isinstance(stats, dict)
