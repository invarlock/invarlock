from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def _base_report_with_windows():
    return {
        "meta": {
            "model_id": "m",
            "adapter": "a",
            "device": "cpu",
            "seed": 1,
            "tokenizer_hash": "tok-123",
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            },
            # Supply initial ratio_ci with non-numeric to hit observed-not-number branch
            "ppl_ratio_ci": ("x", "y"),
        },
        "data": {
            "dataset": "d",
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
                "sparsity": None,
            },
        },
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [1.0, 2.0]}},
    }


def test_meta_tokenizer_hash_propagates_and_ratio_ci_non_numeric_continues():
    report = _base_report_with_windows()
    # Normalization preserves tokenizer_hash from data/meta only when present in data
    report.setdefault("data", {})["tokenizer_hash"] = report["meta"]["tokenizer_hash"]
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [1.0, 2.0]}},
    }

    # Patch paired delta CI computation so ratio_ci_source == 'paired_baseline'
    with (
        patch("invarlock.reporting.certificate.validate_report", return_value=True),
        patch(
            "invarlock.reporting.certificate.compute_paired_delta_log_ci",
            return_value=(-0.1, 0.05),
        ),
    ):
        cert = make_certificate(report, baseline)

    # Tokenizer hash propagated under meta
    assert cert["meta"].get("tokenizer_hash") == "tok-123"

    # Ensure certificate construction succeeded despite non-numeric observed ratio_ci entries
    pm = cert.get("primary_metric", {})
    assert isinstance(pm.get("display_ci"), tuple | list)
