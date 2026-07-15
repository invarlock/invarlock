from copy import deepcopy
from unittest.mock import patch

from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_extract_invariants_skips_non_dict_guard_violations():
    # Craft report with invariants guard that has a mixed violations list (string + dict)
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
            }
        },
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [
            {
                "name": "invariants",
                "passed": False,
                "metrics": {
                    "checks_performed": 1,
                    "violations_found": 1,
                    "fatal_violations": 1,
                    "warning_violations": 0,
                },
                "violations": [
                    "not-a-dict",  # should be skipped
                    {
                        "check": "weight_nan",
                        "type": "fatal",
                        "severity": "error",
                        "layer": 0,
                    },
                ],
            }
        ],
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
    failures = cert.get("invariants", {}).get("failures", [])
    # Only the dict entry should be present
    assert isinstance(failures, list) and len(failures) == 1
    assert failures[0].get("check") in {"weight_nan", "unknown"}
