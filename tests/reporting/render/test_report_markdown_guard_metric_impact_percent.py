from __future__ import annotations

import copy

from invarlock.reporting.rendering.markdown import render_report_markdown
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def test_markdown_guard_metric_display_value_format():
    report = {
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
            "final": {"window_ids": [2], "logloss": [2.302585093], "token_counts": [1]},
        },
        "edit": {"name": "noop"},
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }
    baseline = copy.deepcopy(report)
    cert = make_report(report, baseline)
    # Inject a complete canonical measured impact block.
    cert["guard_metric_impact"] = {
        "metric_kind": "ppl_causal",
        "direction": "lower",
        "degradation_basis": "relative_increase",
        "bare_value": 10.0,
        "guarded_value": 10.15,
        "bare_facts": {"weighted_logloss_sum": 2.302585093, "token_count": 1},
        "guarded_facts": {"weighted_logloss_sum": 2.317473, "token_count": 1},
        "evaluated": True,
        "passed": True,
        "degradation": 0.015,
        "degradation_limit": 0.02,
        "display_value": 1.5,
        "display_unit": "percent",
        "checks": {"guard_metric_impact": True},
        "diagnostics": [],
        "source": "unit_test",
        "schedule_digest": "a" * 32,
    }
    cert.setdefault("validation", {})["guard_metric_impact_acceptable"] = True
    md = render_report_markdown(cert)
    assert "Guard Metric Impact Acceptable" in md and "+1.50%" in md
