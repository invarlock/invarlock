from __future__ import annotations

from invarlock.reporting.certificate import CERTIFICATE_SCHEMA_VERSION
from invarlock.reporting.render import render_certificate_markdown


def _cert_with_pm(kind: str, basis: str) -> dict:
    return {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-basis",
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
            "kind": kind,
            "preview": 10.0,
            "final": 10.0,
            "ratio_vs_baseline": 1.0 if kind.startswith("ppl") else +0.0,
            "display_ci": [1.0, 1.0],
            "gating_basis": basis,
        },
        "auto": {"tier": "balanced", "probes_used": 0, "target_ppl_ratio": None},
        "policies": {},
        "resolved_policy": {},
        "policy_provenance": {},
        "guard_overhead": {},
    }


def test_pm_basis_coherence_ppl_upper() -> None:
    cert = _cert_with_pm("ppl_causal", "upper")
    md = render_certificate_markdown(cert)
    # Basis appears in the PM details and in the Quality Gates row
    assert "- Basis: upper" in md
    # In the Quality Gates table, the Basis column should reflect 'upper'
    assert "| Primary Metric Acceptable |" in md and "| upper |" in md


def test_pm_basis_coherence_accuracy_point() -> None:
    cert = _cert_with_pm("accuracy", "point")
    md = render_certificate_markdown(cert)
    assert "- Basis: point" in md
    assert "| Primary Metric Acceptable |" in md and "| point |" in md
