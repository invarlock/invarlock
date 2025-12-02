from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_extract_invariants_skips_non_dict_guard_violations():
    # Craft report with invariants guard that has a mixed violations list (string + dict)
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {"ppl_preview": 10.0, "ppl_final": 10.0},
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
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "ppl_final": 10.0,
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)
    failures = cert.get("invariants", {}).get("failures", [])
    # Only the dict entry should be present
    assert isinstance(failures, list) and len(failures) == 1
    assert failures[0].get("check") in {"weight_nan", "unknown"}
