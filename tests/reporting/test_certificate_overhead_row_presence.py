from __future__ import annotations

from invarlock.reporting.certificate import CERTIFICATE_SCHEMA_VERSION
from invarlock.reporting.render import render_certificate_markdown


def _base_cert() -> dict:
    return {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-overhead",
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
        "auto": {"tier": "balanced", "probes_used": 0, "target_ppl_ratio": None},
        "policies": {},
        "resolved_policy": {},
        "policy_provenance": {},
    }


def _quality_gate_labels(md: str) -> list[str]:
    rows: list[str] = []
    for ln in md.splitlines():
        if ln.startswith("|") and "| Gate |" not in ln and "|------|" not in ln:
            parts = [p.strip() for p in ln.strip("|").split("|")]
            if parts:
                rows.append(parts[0])
    return rows


def test_overhead_row_omitted_when_not_evaluated() -> None:
    cert = _base_cert()
    cert["guard_overhead"] = {}
    md = render_certificate_markdown(cert)
    labels = _quality_gate_labels(md)
    assert "Guard Overhead Acceptable" not in labels


def test_overhead_row_present_when_evaluated() -> None:
    cert = _base_cert()
    cert["guard_overhead"] = {
        "evaluated": True,
        "overhead_ratio": 1.002,
        "overhead_threshold": 0.01,
    }
    md = render_certificate_markdown(cert)
    labels = _quality_gate_labels(md)
    assert "Guard Overhead Acceptable" in labels
