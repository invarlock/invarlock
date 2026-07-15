from copy import deepcopy
from unittest.mock import patch

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_spectral_baseline_extracted_from_baseline_metrics_block():
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
            # Provide spectral guard metrics at run to build summary
            "spectral": {"max_spectral_norm": 2.0, "mean_spectral_norm": 1.0},
        },
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [{"name": "spectral", "passed": True, "metrics": {}}],
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
    baseline["metrics"]["spectral"] = {
        "max_spectral_norm_final": 1.5,
        "mean_spectral_norm_final": 0.9,
    }
    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(report, baseline)
    spectral = cert.get("spectral", {})
    summary = spectral.get("summary", {})
    assert summary.get("baseline_max_spectral_norm") == 1.5
    assert summary.get("baseline_mean_spectral_norm") == 0.9
