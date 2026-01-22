from __future__ import annotations

import invarlock.reporting.certificate as cert
from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


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
    cert = make_certificate(report, baseline)
    assert cert.get("confidence", {}).get("label") == "High"
    md = render_certificate_markdown(cert)
    assert "**Confidence:** High" in md


def test_confidence_label_medium_when_unstable():
    # Low replicates flags unstable
    report = _mk_report(ratio=1.02, reps=50)
    baseline = _mk_report(ratio=1.0, reps=50)
    cert = make_certificate(report, baseline)
    assert cert.get("confidence", {}).get("label") == "Medium"


def test_confidence_label_low_on_failure():
    report = _mk_report(ratio=1.30, reps=500)
    baseline = _mk_report(ratio=1.0, reps=500)
    cert = make_certificate(report, baseline)
    assert cert.get("confidence", {}).get("label") == "Low"


def test_confidence_thresholds_can_be_overridden_by_policy(monkeypatch):
    # Override tier policy confidence widths and expect threshold to reflect it
    from invarlock.core import auto_tuning as at

    base_policies = at.get_tier_policies()
    balanced = dict(base_policies.get("balanced", {}))
    metrics_obj = balanced.get("metrics", {})
    metrics = dict(metrics_obj) if isinstance(metrics_obj, dict) else {}
    metrics["confidence"] = {
        "ppl_ratio_width_max": 0.02,
        "accuracy_delta_pp_width_max": 0.5,
    }
    balanced["metrics"] = metrics
    patched = dict(base_policies)
    patched["balanced"] = balanced
    monkeypatch.setattr(at, "get_tier_policies", lambda *_a, **_k: patched)

    report = _mk_report(ratio=1.02, reps=500)
    baseline = _mk_report(ratio=1.0, reps=500)
    cert = make_certificate(report, baseline)
    conf = cert.get("confidence", {})
    assert conf.get("basis") == "ppl_ratio"
    assert abs(float(conf.get("threshold")) - 0.02) < 1e-9


def test_compute_confidence_label_accuracy_basis():
    certificate = {
        "primary_metric": {"kind": "accuracy", "display_ci": (0.75, 0.80)},
        "validation": {"primary_metric_acceptable": True},
        "resolved_policy": {
            "confidence": {
                "accuracy_delta_pp_width_max": 0.5,
                "ppl_ratio_width_max": 0.02,
            }
        },
    }
    label = cert._compute_confidence_label(certificate)
    assert label["basis"] == "accuracy"
    assert label["label"] == "High"


def test_compute_confidence_label_handles_missing_ci():
    certificate = {
        "primary_metric": {"kind": "ppl_causal", "ratio_vs_baseline": 1.1},
        "validation": {"primary_metric_acceptable": False},
        "resolved_policy": {},
    }
    label = cert._compute_confidence_label(certificate)
    assert label["basis"] == "primary_metric"
    assert label["label"] == "Low"


def test_compute_confidence_label_skips_non_interval_display_ci():
    certificate = {
        "primary_metric": {"kind": "ppl_causal", "display_ci": "not-an-interval"},
        "validation": {"primary_metric_acceptable": True},
        "resolved_policy": {},
    }
    label = cert._compute_confidence_label(certificate)
    assert label["basis"] == "primary_metric"
