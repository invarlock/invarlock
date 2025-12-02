from __future__ import annotations

from invarlock.reporting.certificate import CERTIFICATE_SCHEMA_VERSION
from invarlock.reporting.render import (
    compute_console_validation_block,
    render_certificate_markdown,
)


def _minimal_cert(pm_kind: str = "ppl_causal", guard_evaluated: bool = False) -> dict:
    # Minimal, structurally valid certificate focused on validation/MD rendering
    cert = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-xyz",
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
            "kind": pm_kind,
            "preview": 50.0,
            "final": 50.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
            "gating_basis": "upper" if pm_kind.startswith("ppl") else "point",
        },
        "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
        "policies": {},
        "resolved_policy": {},
        "policy_provenance": {},
    }
    if guard_evaluated:
        cert["guard_overhead"] = {
            "evaluated": True,
            "overhead_ratio": 1.005,
            "overhead_threshold": 0.01,
        }
    else:
        cert["guard_overhead"] = {}
    return cert


def _extract_quality_gate_rows(md: str) -> list[str]:
    # Parse the Quality Gates table first-column labels
    rows: list[str] = []
    lines = md.splitlines()
    in_table = False
    for ln in lines:
        if ln.strip() == "## Quality Gates":
            in_table = True
            continue
        if in_table and ln.startswith("## "):
            break
        if (
            in_table
            and ln.startswith("|")
            and "| Gate |" not in ln
            and "|------|" not in ln
        ):
            # take first cell text
            parts = [p.strip() for p in ln.strip("|").split("|")]
            if parts:
                rows.append(parts[0])
    return rows


def test_md_overall_status_matches_console_dev_no_overhead() -> None:
    cert = _minimal_cert(pm_kind="ppl_causal", guard_evaluated=False)
    block = compute_console_validation_block(cert)
    md = render_certificate_markdown(cert)
    expected = "PASS" if block["overall_pass"] else "FAIL"
    assert (
        f"**Overall Status:** ✅ {expected}" in md
        or f"**Overall Status:** ❌ {expected}" in md
    )


def test_md_quality_gates_match_console_presence_with_and_without_overhead() -> None:
    # No overhead evaluated: only Primary Metric and Preview Final Drift gates appear
    cert = _minimal_cert(pm_kind="ppl_causal", guard_evaluated=False)
    md = render_certificate_markdown(cert)
    rows = _extract_quality_gate_rows(md)
    assert "Primary Metric Acceptable" in rows
    assert "Preview Final Drift Acceptable" in rows
    assert all(r != "Guard Overhead Acceptable" for r in rows)

    # With overhead evaluated: Guard Overhead row appears
    cert2 = _minimal_cert(pm_kind="ppl_causal", guard_evaluated=True)
    md2 = render_certificate_markdown(cert2)
    rows2 = _extract_quality_gate_rows(md2)
    assert "Primary Metric Acceptable" in rows2
    assert "Preview Final Drift Acceptable" in rows2
    assert "Guard Overhead Acceptable" in rows2
