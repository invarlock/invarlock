from __future__ import annotations

from copy import deepcopy

from invarlock.reporting.report_metric_impact import (
    compute_guard_metric_impact_from_guard,
)
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def _acc_report(correct: int, total: int) -> dict:
    # Minimal report embedding classification aggregates for accuracy
    return {
        "metrics": {
            "primary_metric": {
                "kind": "accuracy",
                "preview": correct / total,
                "final": correct / total,
            },
            "classification": {"final": {"correct_total": correct, "total": total}},
        },
        "evaluation_windows": {"final": {"example_ids": list(range(total))}},
    }


def test_guard_metric_impact_accuracy_delta_pp_basis() -> None:
    # Bare = 70%, Guarded = 68% → 2 percentage points of degradation.
    bare = _acc_report(70, 100)
    guarded = _acc_report(68, 100)
    out = compute_guard_metric_impact_from_guard(
        {"bare_report": bare, "guarded_report": guarded}, pm_kind_hint="accuracy"
    )
    assert isinstance(out, dict)
    assert out["metric_kind"] == "accuracy"
    assert out["direction"] == "higher"
    assert out["degradation_basis"] == "absolute_drop"
    assert abs(float(out["degradation"]) - 0.02) < 1e-6
    assert abs(float(out["display_value"]) - 2.0) < 1e-6
    assert out["display_unit"] == "percentage_points"


def test_make_evaluation_report_attaches_guard_metric_impact_for_accuracy() -> None:
    # Prepare a evaluation_report where primary_metric is accuracy and guard has bare/guarded
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf",
            "device": "cpu",
            "seed": 1,
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "metrics": {
            "primary_metric": {"kind": "accuracy", "preview": 0.70, "final": 0.68},
            "classification": {
                "final": {"correct_total": 68, "total": 100},
                "preview": {"correct_total": 70, "total": 100},
            },
        },
        "guard_metric_impact": {
            "degradation_limit": 0.03,
            "bare_report": _acc_report(70, 100),
            "guarded_report": _acc_report(68, 100),
        },
        "evaluation_windows": {},
        "edit": {"name": "noop"},
        "data": {
            "dataset": "accuracy-fixture",
            "split": "validation",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 1,
            "final_n": 1,
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    baseline = deepcopy(report)
    baseline["metrics"]["primary_metric"] = {
        "kind": "accuracy",
        "preview": 0.70,
        "final": 0.70,
    }
    cert = make_report(report, baseline)
    qo = cert.get("guard_metric_impact", {})
    assert isinstance(qo, dict)
    assert qo["evaluated"] is True
    assert qo["passed"] is True
    assert qo["metric_kind"] == "accuracy"
    assert qo["degradation_basis"] == "absolute_drop"
    assert abs(float(qo["degradation"]) - 0.02) < 1e-6
