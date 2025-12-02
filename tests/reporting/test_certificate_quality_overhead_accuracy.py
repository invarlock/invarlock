from __future__ import annotations

from invarlock.reporting.certificate import (
    _compute_quality_overhead_from_guard,
    make_certificate,
)


def _acc_report(correct: int, total: int) -> dict:
    # Minimal report embedding classification aggregates for accuracy
    return {
        "metrics": {
            "classification": {"final": {"correct_total": correct, "total": total}}
        }
    }


def test_quality_overhead_accuracy_delta_pp_basis() -> None:
    # Bare = 70%, Guarded = 68% → -2.0 pp delta
    bare = _acc_report(70, 100)
    guarded = _acc_report(68, 100)
    out = _compute_quality_overhead_from_guard(
        {"bare_report": bare, "guarded_report": guarded}, pm_kind_hint="accuracy"
    )
    assert isinstance(out, dict) and out.get("basis") == "delta_pp"
    assert abs(float(out.get("value", 0.0)) + 2.0) < 1e-6  # negative 2.0 pp


def test_make_certificate_attaches_quality_overhead_for_accuracy() -> None:
    # Prepare a certificate where primary_metric is accuracy and guard has bare/guarded
    report = {
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1},
        "metrics": {
            "primary_metric": {"kind": "accuracy", "preview": 0.70, "final": 0.68},
            "classification": {
                "final": {"correct_total": 68, "total": 100},
                "preview": {"correct_total": 70, "total": 100},
            },
        },
        "guard_overhead": {
            "bare_report": _acc_report(70, 100),
            "guarded_report": _acc_report(68, 100),
        },
        "evaluation_windows": {},
        "edit": {"name": "noop"},
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    baseline = {
        "metrics": {"primary_metric": {"kind": "accuracy", "final": 0.70}},
    }
    cert = make_certificate(report, baseline)
    qo = cert.get("quality_overhead", {})
    assert isinstance(qo, dict) and qo.get("basis") == "delta_pp"
    assert qo.get("kind") in {"accuracy", "vqa_accuracy"}
