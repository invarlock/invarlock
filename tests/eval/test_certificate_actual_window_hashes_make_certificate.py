from unittest.mock import patch

from invarlock.reporting.certificate import make_certificate


def test_dataset_hash_uses_actual_evaluation_windows_ids():
    # Build report with explicit token IDs in evaluation windows so that
    # certificate computes actual hashes (sha256) rather than config fallback.
    report = {
        "meta": {"model_id": "m", "seed": 123},
        "metrics": {"ppl_preview": 10.0, "ppl_final": 10.0},
        "data": {
            "dataset": "d",
            "split": "val",
            "seq_len": 4,
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
        "evaluation_windows": {
            "preview": {"input_ids": [[1, 2, 3, 4]]},
            "final": {"input_ids": [[5, 6], [7, 8, 9]]},
        },
    }
    baseline = {"run_id": "b", "model_id": "m", "ppl_final": 10.0}

    with patch("invarlock.reporting.certificate.validate_report", return_value=True):
        cert = make_certificate(report, baseline)

    dataset_hash = cert["dataset"]["hash"]
    assert dataset_hash["preview"].startswith("sha256:")
    assert dataset_hash["final"].startswith("sha256:")
    # 4 tokens in preview, 5 in final
    assert dataset_hash["preview_tokens"] == 4
    assert dataset_hash["final_tokens"] == 5
    # No dataset-level hash when computed from token IDs
    assert dataset_hash["dataset"] is None
