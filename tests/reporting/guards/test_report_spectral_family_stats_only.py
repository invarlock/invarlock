from copy import deepcopy
from unittest.mock import patch

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_spectral_families_from_family_stats_only():
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
        },
        "guards": [
            {
                "name": "spectral",
                "passed": True,
                "metrics": {
                    "family_stats": {
                        "ffn": {
                            "count": 3,
                            "mean": 1.2,
                            "std": 0.1,
                            "min": 1.0,
                            "max": 1.3,
                        },
                        "attn": {
                            "count": 2,
                            "mean": 1.1,
                            "std": 0.1,
                            "min": 1.0,
                            "max": 1.2,
                        },
                    }
                },
                "policy": {},
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
    spectral = cert.get("spectral", {})
    assert spectral.get("families", {}).get("ffn", {}).get("count") == 3
