from __future__ import annotations

from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_guard_metric_impact_row_na_and_hysteresis_note():
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
                "dataset": "ds",
                "split": "val",
                "seq_len": 8,
                "stride": 8,
                "preview_n": 1,
                "final_n": 1,
            },
            "metrics": {
                "primary_metric": {
                    "kind": "ppl_causal",
                    "preview": 10.0,
                    "final": 10.0,
                    "ratio_vs_baseline": 1.0,
                    "display_ci": [1.0, 1.0],
                },
            },
            "evaluation_windows": {
                "preview": {
                    "window_ids": [1],
                    "logloss": [2.302585093],
                    "token_counts": [1],
                },
                "final": {
                    "window_ids": [2],
                    "logloss": [2.302585093],
                    "token_counts": [1],
                },
            },
            "edit": {"name": "noop"},
            "guards": [],
            "artifacts": {"events_path": "", "logs_path": ""},
        }
    )
    baseline = canonical_baseline(
        {
            **report,
            "edit": {
                "name": "noop",
                "plan_digest": "baseline_noop",
                "deltas": {"params_changed": 0},
            },
        }
    )
    cert = make_report(report, baseline)
    # Force guard metric impact evaluated without measured fields
    cert["guard_metric_impact"] = {"evaluated": True, "degradation_limit": 0.012}
    cert.setdefault("validation", {})["guard_metric_impact_acceptable"] = True
    cert["validation"]["hysteresis_applied"] = True
    md = render_report_markdown(cert)
    assert "Guard Metric Impact Acceptable" in md and "N/A" in md
    assert "hysteresis applied" in md
