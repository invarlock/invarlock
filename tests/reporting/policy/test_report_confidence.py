from __future__ import annotations

from invarlock.reporting.rendering.markdown import render_report_markdown
from invarlock.reporting.report_enrichment import compute_confidence_label
from invarlock.reporting.report_make import make_report as production_make_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
    refresh_runtime_policy_receipt,
)
from tests.reporting._support_canonical_reports import (
    make_canonical_report as make_report,
)


def _mk_report(ratio: float = 1.00, reps: int | None = None) -> dict:
    metrics = {
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 50.0,
            "final": 50.0 * ratio,
            "ratio_vs_baseline": ratio,
            "display_ci": (ratio, ratio),
        },
        "preview_total_tokens": 30000,
        "final_total_tokens": 30000,
    }
    if reps is not None:
        metrics["bootstrap"] = {"replicates": int(reps), "alpha": 0.05}
    return {
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "device": "cpu",
            "seed": 42,
            "ts": "now",
            "auto": {"tier": "balanced"},
        },
        "context": {"profile": "dev"},
        "data": {
            "dataset": "dummy",
            "split": "validation",
            "seq_len": 8,
            "stride": 4,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "noop",
            "deltas": {
                "params_changed": 0,
                "sparsity": None,
                "bitwidth_map": None,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "guards": [],
        "metrics": metrics,
        "artifacts": {"events_path": "", "logs_path": "", "checkpoint_path": None},
        "flags": {"guard_recovered": False, "rollback_reason": None},
    }


def test_confidence_label_high_when_stable_and_narrow_ci():
    report = _mk_report(ratio=1.02, reps=500)
    baseline = _mk_report(ratio=1.0, reps=500)
    cert = make_report(report, baseline)
    assert cert.get("confidence", {}).get("label") == "High"
    md = render_report_markdown(cert)
    assert "**Confidence:** High" in md


def test_confidence_label_medium_when_unstable():
    # Low replicates flags unstable
    report = _mk_report(ratio=1.02, reps=50)
    baseline = _mk_report(ratio=1.0, reps=50)
    cert = make_report(report, baseline)
    assert cert.get("confidence", {}).get("label") == "Medium"


def test_confidence_label_low_on_failure():
    report = _mk_report(ratio=1.30, reps=500)
    baseline = _mk_report(ratio=1.0, reps=500)
    cert = make_report(report, baseline)
    assert cert.get("confidence", {}).get("label") == "Low"


def test_confidence_thresholds_can_be_overridden_by_runtime_policy():
    report = _mk_report(ratio=1.02, reps=500)
    baseline = _mk_report(ratio=1.0, reps=500)
    canonical_report = canonical_run_report(report)
    canonical_report["resolved_policy"]["metrics"]["confidence"] = {
        "ppl_ratio_width_max": 0.02,
        "accuracy_delta_pp_width_max": 0.5,
    }
    canonical_report["resolved_policy"]["confidence"] = {
        "ppl_ratio_width_max": 0.02,
        "accuracy_delta_pp_width_max": 0.5,
    }
    cert = production_make_report(
        refresh_runtime_policy_receipt(canonical_report),
        canonical_baseline(baseline),
    )
    conf = cert.get("confidence", {})
    assert conf.get("basis") == "ppl_ratio"
    assert abs(float(conf.get("threshold")) - 0.02) < 1e-9


def test_compute_confidence_label_accuracy_basis():
    evaluation_report = {
        "primary_metric": {"kind": "accuracy", "display_ci": (75.0, 80.0)},
        "validation": {"primary_metric_acceptable": True},
        "resolved_policy": {
            "confidence": {
                "accuracy_delta_pp_width_max": 0.5,
                "ppl_ratio_width_max": 0.02,
            }
        },
    }
    label = compute_confidence_label(evaluation_report)
    assert label["basis"] == "accuracy"
    assert label["width"] == 5.0
    assert label["label"] == "Low"


def test_compute_confidence_label_handles_missing_ci():
    evaluation_report = {
        "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.1},
        "validation": {"primary_metric_acceptable": False},
        "resolved_policy": {},
    }
    label = compute_confidence_label(evaluation_report)
    assert label["basis"] == "primary_metric"
    assert label["label"] == "Low"


def test_compute_confidence_label_skips_non_interval_display_ci():
    evaluation_report = {
        "primary_metric": {"kind": "ppl_causal", "display_ci": "not-an-interval"},
        "validation": {"primary_metric_acceptable": True},
        "resolved_policy": {},
    }
    label = compute_confidence_label(evaluation_report)
    assert label["basis"] == "primary_metric"
