from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_evaluation_report_has_no_ppl_block_pm_only():
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 1,
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            },
        },
        "data": {
            "dataset": "dummy",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [],
        "edit": {
            "name": "structured",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    baseline = deepcopy(report)
    baseline["edit"]["name"] = "noop"
    baseline["metrics"]["primary_metric"].pop("ratio_vs_baseline")

    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)

    # PM-only: no 'ppl' key should be present in the evaluation_report
    assert "ppl" not in cert
    # Primary metric should exist
    assert isinstance(cert.get("primary_metric"), dict) and cert["primary_metric"]
