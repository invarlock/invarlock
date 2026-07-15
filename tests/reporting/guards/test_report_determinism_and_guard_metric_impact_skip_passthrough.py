from __future__ import annotations

from invarlock.reporting.report_types import RunReport, create_empty_report
from tests.reporting._support_auto_config import make_auto_config
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)
from tests.reporting.overhead._support import overhead_baseline_report


def _mk_minimal_report() -> RunReport:
    r = create_empty_report()
    r["meta"]["model_id"] = "m"
    r["meta"]["adapter"] = "hf"
    r["meta"]["device"] = "cpu"
    r["meta"]["auto"] = make_auto_config()
    r["context"] = {"profile": "dev"}
    r["data"]["dataset"] = "unit"
    r["data"]["split"] = "validation"
    r["data"]["seq_len"] = 8
    r["data"]["stride"] = 8
    r["data"]["preview_n"] = 2
    r["data"]["final_n"] = 2
    r["edit"]["name"] = "noop"
    r["metrics"]["primary_metric"] = {
        "kind": "ppl_causal",
        "preview": 10.0,
        "final": 10.0,
    }
    r["metrics"]["bootstrap"] = {
        "method": "percentile",
        "replicates": 50,
        "alpha": 0.05,
        "seed": 0,
        "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
    }
    r["metrics"]["preview_final_slice_delta_summary"] = {"mean": 0.0}
    r["metrics"]["preview_total_tokens"] = 50
    r["metrics"]["final_total_tokens"] = 50
    r["metrics"]["logloss_delta"] = 0.0
    r["metrics"]["logloss_delta_ci"] = (-0.01, 0.01)
    r["evaluation_windows"] = {
        "preview": {
            "window_ids": [3, 4],
            "logloss": [2.30, 2.30],
            "token_counts": [100, 100],
        },
        "final": {
            "window_ids": [1, 2],
            "logloss": [2.30, 2.30],
            "token_counts": [100, 100],
        },
    }
    return r


def _mk_minimal_baseline() -> dict:
    return overhead_baseline_report(
        metrics={
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
            },
        },
        tier="balanced",
        profile="dev",
    )


def test_evaluation_report_propagates_determinism_and_metric_impact_skip_fields() -> (
    None
):
    report = _mk_minimal_report()
    baseline = _mk_minimal_baseline()
    report["meta"]["determinism"] = {
        "requested": "strict",
        "level": "tolerance",
        "seed": 0,
    }
    report["guard_metric_impact"] = {
        "mode": "skipped",
        "skipped": True,
        "skip_reason": "context.run.skip_guard_metric_impact_check",
        "source": "config:context.run.skip_guard_metric_impact_check",
        "degradation_limit": 0.01,
    }

    cert = make_report(report, baseline)
    meta = cert.get("meta", {})
    assert isinstance(meta, dict)
    assert isinstance(meta.get("determinism"), dict)
    assert meta["determinism"]["level"] == "tolerance"

    impact = cert.get("guard_metric_impact", {})
    assert isinstance(impact, dict)
    assert impact.get("mode") == "skipped"
    assert impact.get("skipped") is True
    assert impact.get("skip_reason") == "context.run.skip_guard_metric_impact_check"
    assert impact.get("evaluated") is False
    assert impact.get("passed") is False
