from copy import deepcopy
from unittest.mock import patch

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_spectral_summary_modules_checked_non_int_ignored():
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 1,
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            },
            "spectral": {
                "max_spectral_norm": 1.2,
                "mean_spectral_norm": 1.0,
                "modules_checked": "not-an-int",
            },
        },
        "guards": [
            {
                "name": "spectral",
                "passed": True,
                "metrics": {"max_spectral_norm": 1.2, "mean_spectral_norm": 1.0},
                "policy": {"deadband": "0.10"},  # ensure deadband float coercion path
            }
        ],
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
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
    baseline = deepcopy(report)
    baseline["run_id"] = "b"
    baseline["edit"]["name"] = "noop"
    baseline["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
    }
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)
    summary = cert.get("spectral", {}).get("summary", {})
    # modules_checked non-int is ignored; and deadband gets parsed
    assert "modules_checked" not in summary
    assert isinstance(summary.get("deadband"), float)
