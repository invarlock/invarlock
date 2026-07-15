from copy import deepcopy
from unittest.mock import patch

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)
from tests.reporting._support_primary_metric import independent_slice_summary


def test_paired_path_skips_ci_when_zero_replicates():
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
                "final": 10.2,
                "ratio_vs_baseline": 1.02,
                "display_ci": (0.98, 1.06),
            },
            "bootstrap": {
                "method": "percentile",
                "replicates": 0,
                "alpha": 0.1,
                "seed": 0,
            },
            "preview_final_slice_delta_summary": independent_slice_summary(
                0.0198,
                preview_windows=2,
                final_windows=2,
            ),
        },
        "data": {
            "dataset": "dummy",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 2,
            "final_n": 2,
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
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [0.1, 0.2]}},
    }
    baseline = deepcopy(report)
    baseline["run_id"] = "b"
    baseline["edit"]["name"] = "noop"
    baseline["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 9.8,
        "final": 9.8,
    }
    baseline["evaluation_windows"] = {
        "final": {"window_ids": [1, 2], "logloss": [0.09, 0.19]}
    }

    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)
    stats = cert.get("dataset", {}).get("windows", {}).get("stats", {})
    # Paired baseline windows remain counted independently from preview/final slice evidence.
    assert stats.get("paired_windows") == 2
    assert stats.get("pairing") == "independent_preview_final"
