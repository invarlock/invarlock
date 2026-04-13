from unittest.mock import patch

import pytest

from invarlock.core.exceptions import ValidationError
from invarlock.reporting.report_make import make_report


def test_evaluation_report_ppl_both_invalid_rejects_strict_baseline():
    report = {
        "meta": {"model_id": "m", "seed": 1},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "preview": 0.0, "final": 0.0}
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
    # Baseline is invalid under the stricter contract, so make_report should fail.
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 0.0,
            "final": 0.0,
        },
        "ppl_final": 0.0,
    }
    with (
        patch(
            "invarlock.reporting.report_normalization.validate_report",
            return_value=True,
        ),
        pytest.raises(ValidationError, match="Baseline normalization failed"),
    ):
        make_report(report, baseline)
