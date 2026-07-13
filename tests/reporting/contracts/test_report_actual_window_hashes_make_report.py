from unittest.mock import patch

from invarlock.reporting.report_make import make_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def test_dataset_hash_uses_actual_evaluation_windows_ids():
    # Build report with explicit token IDs in evaluation windows so that
    # evaluation_report computes actual hashes (sha256) rather than config fallback.
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 123,
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev", "assurance": {"mode": "off"}},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 10.0}
        },
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
    baseline = {**report, "edit": {"name": "noop"}}

    with patch(
        "invarlock.reporting.report_normalization.validate_report", return_value=True
    ):
        cert = make_report(canonical_run_report(report), canonical_baseline(baseline))

    dataset_hash = cert["dataset"]["hash"]
    assert dataset_hash["preview"].startswith("sha256:")
    assert dataset_hash["final"].startswith("sha256:")
    # 4 tokens in preview, 5 in final
    assert dataset_hash["preview_tokens"] == 4
    assert dataset_hash["final_tokens"] == 5
    # No dataset-level hash when computed from token IDs
    assert dataset_hash["dataset"] is None
    assert dataset_hash["source"] == "explicit_token_ids"
