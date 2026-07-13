from __future__ import annotations

from invarlock.reporting.report_make import make_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)
from tests.reporting._support_primary_metric import independent_slice_summary


def test_identity_deviation_in_dev_does_not_raise() -> None:
    # Construct a minimal report with drift ratio 1.0 but paired delta implying a different ratio
    report = canonical_run_report(
        {
            "meta": {
                "model_id": "bert-tiny",
                "adapter": "hf_mlm",
                "seed": 42,
                "auto": {"tier": "balanced"},
            },
            "context": {"profile": "dev", "assurance": {"mode": "off"}},
            "metrics": {
                "preview_final_slice_delta_summary": independent_slice_summary(
                    0.5, preview_windows=1, final_windows=1
                ),  # exp(0.5)=1.65 → mismatch
                "window_plan": {"profile": "dev"},
                "primary_metric": {"kind": "ppl_mlm", "preview": 100.0, "final": 100.0},
            },
            "evaluation_windows": {
                "final": {
                    "window_ids": [1, 2, 3],
                    "logloss": [1.0, 1.2, 0.9],
                    "token_counts": [10, 10, 10],
                }
            },
            "data": {
                "dataset": "wikitext2",
                "split": "validation",
                "seq_len": 128,
                "stride": 128,
                "preview_n": 1,
                "final_n": 1,
            },
            "edit": {"name": "structured"},
            "guards": [],
            "artifacts": {},
        }
    )
    baseline = canonical_baseline(
        {
            **report,
            "edit": {"name": "noop"},
            "metrics": {
                **report["metrics"],
                "primary_metric": {"kind": "ppl_mlm", "preview": 50.0, "final": 50.0},
            },
            "evaluation_windows": {
                "final": {
                    "window_ids": [10, 11, 12],
                    "logloss": [1.0, 1.2, 0.9],
                    "token_counts": [10, 10, 10],
                }
            },
        }
    )
    # Should not raise
    cert = make_report(report, baseline)
    assert isinstance(cert, dict)
