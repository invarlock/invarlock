from __future__ import annotations

from invarlock.reporting.certificate import CERTIFICATE_SCHEMA_VERSION
from invarlock.reporting.render import render_certificate_markdown


def _cert() -> dict:
    return {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-safety",
        "artifacts": {"generated_at": "now"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "wikitext2",
            "seq_len": 16,
            "windows": {"preview": 1, "final": 1, "seed": 1},
            "hash": {"preview_tokens": 0, "final_tokens": 0, "total_tokens": 0},
        },
        "invariants": {"status": "pass", "summary": {}},
        "spectral": {"caps_applied": 0, "max_caps": 5},
        "rmt": {"stable": True, "status": "ok"},
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
        "guard_overhead": {},
    }


def test_quality_gates_do_not_include_rmt_epsilon_rule() -> None:
    md = render_certificate_markdown(_cert())
    # Ensure 'RMT ε-rule' is not listed as a Quality Gate row
    assert "## Quality Gates" in md
    # Scan Quality Gates block
    qg_block = md.split("## Quality Gates", 1)[1].split("## ", 1)[0]
    assert "RMT ε-rule" not in qg_block


def test_safety_rows_follow_allowlist() -> None:
    md = render_certificate_markdown(_cert())
    # Guard check detail rows should include exactly these labels
    assert "## Guard Check Details" in md
    block = md.split("## Guard Check Details", 1)[1]
    assert "| Invariants |" in block
    assert "| Spectral Stability |" in block
    assert "| RMT Health |" in block
