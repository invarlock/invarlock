from __future__ import annotations

from invarlock.reporting.certificate import (
    CERTIFICATE_SCHEMA_VERSION,
    _compute_validation_flags,
)
from invarlock.reporting.render import render_certificate_markdown


def test_primary_metric_accuracy_gating_balanced_passes():
    ppl = {"preview_final_ratio": 1.0, "ratio_vs_baseline": 1.0}
    spectral = {"caps_applied": 0, "max_caps": 5}
    rmt = {"stable": True}
    invariants = {"status": "pass"}
    pm = {"kind": "accuracy", "final": 0.90, "ratio_vs_baseline": -0.5, "n_final": 400}

    flags = _compute_validation_flags(
        ppl,
        spectral,
        rmt,
        invariants,
        tier="balanced",
        _ppl_metrics=None,
        target_ratio=None,
        guard_overhead=None,
        primary_metric=pm,
    )
    assert flags["primary_metric_acceptable"] is True


def test_render_includes_primary_metric_section():
    cert = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "run_id": "abc123",
        "artifacts": {"generated_at": "now"},
        "plugins": {},
        "meta": {
            "model_id": "gpt2",
            "adapter": "hf_causal",
            "device": "cpu",
            "ts": "now",
            "seed": 42,
        },
        "dataset": {
            "provider": "wikitext2",
            "seq_len": 32,
            "windows": {"preview": 1, "final": 1, "seed": 42},
            "hash": {"preview_tokens": None, "final_tokens": None, "total_tokens": 0},
        },
        "invariants": {"status": "pass"},
        "spectral": {"caps_applied": 0, "max_caps": 5},
        "rmt": {"stable": True},
        "variance": {"enabled": False},
        "structure": {
            "layers_modified": 0,
            "params_changed": 0,
            "heads_pruned": 0,
            "neurons_pruned": 0,
        },
        "policies": {},
        "resolved_policy": {},
        "policy_provenance": {},
        "provenance": {},
        "edit_name": "noop",
        "validation": {
            "preview_final_drift_acceptable": True,
            "primary_metric_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "guard_overhead_acceptable": True,
        },
        "guard_overhead": {},
        "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        "primary_metric": {
            "kind": "accuracy",
            "preview": 0.8,
            "final": 0.9,
            "ratio_vs_baseline": +0.05,
        },
    }
    md = render_certificate_markdown(cert)
    assert "## Primary Metric" in md
    assert "Kind: accuracy" in md
    # With primary metric present, PPL/Drift rows should be omitted
    assert "PPL vs Baseline" not in md
    assert "Drift (final/preview)" not in md
