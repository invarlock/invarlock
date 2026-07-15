from __future__ import annotations

from invarlock.reporting.report_make import make_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def _mk_minimal_report_with_windows() -> dict:
    return canonical_run_report(
        {
            "meta": {
                "model_id": "m",
                "adapter": "hf_causal",
                "device": "cpu",
                "seed": 42,
                "auto": {"tier": "balanced"},
            },
            "context": {"profile": "dev", "assurance": {"mode": "off"}},
            "data": {
                "dataset": "unit",
                "split": "validation",
                "seq_len": 8,
                "stride": 8,
                "preview_n": 1,
                "final_n": 1,
            },
            "edit": {"name": "structured"},
            "metrics": {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": 10.0,
                    "final": 10.0,
                    "ratio_vs_baseline": 1.0,
                    "display_ci": (1.0, 1.0),
                }
            },
            "evaluation_windows": {
                "preview": {"window_ids": [1], "logloss": [1.0], "token_counts": [10]},
                "final": {"window_ids": [2], "logloss": [1.0], "token_counts": [10]},
            },
            "guards": [],
            "artifacts": {"events_path": "", "logs_path": ""},
        }
    )


def test_edit_digest_quantization():
    report = _mk_minimal_report_with_windows()
    # Quantization run
    report["edit"] = {"name": "quant_rtn", "config": {"bitwidth": 4, "scope": "ffn"}}
    report = canonical_run_report(report)
    baseline = canonical_baseline(
        {
            **report,
            "edit": {"name": "noop"},
        }
    )
    cert = make_report(report, baseline)
    ed = cert.get("provenance", {}).get("edit_digest", {})
    assert ed.get("family") == "quantization"
    ih = ed.get("impl_hash")
    assert isinstance(ih, str) and len(ih) >= 16
    assert ed.get("version") == 1


def test_edit_digest_report_only():
    report = _mk_minimal_report_with_windows()
    # Report-only (no in-run edit)
    report["edit"] = {"name": "noop"}
    report = canonical_run_report(report)
    baseline = canonical_baseline(
        {
            **report,
            "edit": {"name": "noop"},
        }
    )
    cert = make_report(report, baseline)
    ed = cert.get("provenance", {}).get("edit_digest", {})
    assert ed.get("family") == "report_only"
    ih = ed.get("impl_hash")
    assert isinstance(ih, str) and len(ih) >= 16
