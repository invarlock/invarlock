from __future__ import annotations

from invarlock.reporting.certificate import CERTIFICATE_SCHEMA_VERSION
from invarlock.reporting.render import render_certificate_markdown


def test_autotune_labeling_is_informational() -> None:
    cert = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-auto",
        "artifacts": {"generated_at": "now"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "wikitext2",
            "seq_len": 16,
            "windows": {"preview": 1, "final": 1, "seed": 1},
            "hash": {"preview_tokens": 0, "final_tokens": 0, "total_tokens": 0},
        },
        "invariants": {"status": "pass"},
        "spectral": {"caps_applied": 0, "max_caps": 5},
        "rmt": {"stable": True},
        "structure": {"layers_modified": 0, "params_changed": 0},
        "variance": {"enabled": False},
        "validation": {
            "preview_final_drift_acceptable": True,
            "primary_metric_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "guard_overhead_acceptable": True,
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
            "gating_basis": "upper",
        },
        "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": 2.0},
        "policies": {},
        "resolved_policy": {},
        "policy_provenance": {},
        "guard_overhead": {},
    }
    md = render_certificate_markdown(cert)
    assert "Target Ratio vs Baseline:" not in md
    assert "Auto Policy Target Ratio (informational):" in md
