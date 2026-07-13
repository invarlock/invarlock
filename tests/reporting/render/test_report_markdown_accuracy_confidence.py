from __future__ import annotations

from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_markdown_accuracy_low_baseline_note_and_confidence():
    # Build a valid evaluation_report via make_report, then tweak PM for the note
    report = canonical_run_report(
        {
            "meta": {
                "model_id": "m",
                "adapter": "hf",
                "device": "cpu",
                "seed": 1,
                "auto": {"tier": "balanced"},
            },
            "context": {"profile": "dev"},
            "data": {
                "dataset": "dummy",
                "split": "validation",
                "seq_len": 8,
                "stride": 8,
                "preview_n": 100,
                "final_n": 100,
            },
            "guards": [],
            "metrics": {
                "primary_metric": {"kind": "accuracy", "preview": 0.70, "final": 0.72},
                "classification": {
                    "preview": {"correct_total": 70, "total": 100},
                    "final": {"correct_total": 72, "total": 100},
                },
            },
            "edit": {"name": "noop"},
            "artifacts": {"events_path": "", "logs_path": ""},
        }
    )
    baseline = canonical_baseline(report)
    baseline["metrics"]["primary_metric"].update({"preview": 0.04, "final": 0.04})
    cert = make_report(report, baseline)
    # Force confidence label and baseline_point for rendering branches
    cert.setdefault("confidence", {})["label"] = "Medium"
    cert["primary_metric"]["baseline_point"] = 0.04
    cert["primary_metric"]["delta_vs_baseline_pp"] = 2.0
    md = render_report_markdown(cert)
    # Confidence label rendered
    assert "Confidence:" in md and "Medium" in md
    # Baseline < 5% note rendered for accuracy
    assert "baseline < 5%" in md
