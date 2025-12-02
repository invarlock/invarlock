from __future__ import annotations

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def test_markdown_accuracy_low_baseline_note_and_confidence():
    # Build a valid certificate via make_certificate, then tweak PM for the note
    report = {
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1},
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
    baseline = {"metrics": {"primary_metric": {"kind": "accuracy", "final": 0.04}}}
    cert = make_certificate(report, baseline)
    # Force confidence label and baseline_point for rendering branches
    cert.setdefault("confidence", {})["label"] = "Medium"
    cert["primary_metric"]["baseline_point"] = 0.04
    cert["primary_metric"]["ratio_vs_baseline"] = 0.02
    md = render_certificate_markdown(cert)
    # Confidence label rendered
    assert "Confidence:" in md and "Medium" in md
    # Baseline < 5% note rendered for accuracy
    assert "baseline < 5%" in md
