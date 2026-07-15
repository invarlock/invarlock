from unittest.mock import patch

from invarlock.reporting.report_make import make_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def test_ppl_stats_window_pairing_fields_passthrough():
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
            "window_match_fraction": 0.75,
            "window_overlap_fraction": 0.25,
            "window_pairing_reason": "exact",
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
    stats = cert.get("dataset", {}).get("windows", {}).get("stats", {})
    assert stats.get("window_match_fraction") == 0.75
    assert stats.get("window_overlap_fraction") == 0.25
    assert stats.get("window_pairing_reason") == "exact"
