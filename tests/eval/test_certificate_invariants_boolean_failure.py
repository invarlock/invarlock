from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_invariants_boolean_failure_path_sets_failures_and_warn():
    # invariants_data contains a non-dict boolean False → records an error-severity failure
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 10.0,
            "invariants": {"bool_check": False},
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
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "ppl_final": 10.0,
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [0.1]}},
    }
    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)
    inv = cert.get("invariants", {})
    assert inv.get("status") in {"warn", "fail"}
    failures = inv.get("failures", [])
    assert any(f.get("check") == "bool_check" for f in failures)
