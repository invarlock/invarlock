from __future__ import annotations

from invarlock.reporting.certificate import CERTIFICATE_SCHEMA_VERSION
from invarlock.reporting.render import render_certificate_markdown


def _base_cert() -> dict:
    return {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-x",
        "artifacts": {"generated_at": "now"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "wikitext2",
            "seq_len": 16,
            "windows": {"preview": 1, "final": 1, "seed": 1},
            "hash": {"preview_tokens": 0, "final_tokens": 0, "total_tokens": 0},
        },
        "structure": {"layers_modified": 0, "params_changed": 0},
        "invariants": {"summary": {}, "status": "pass", "failures": []},
        "spectral": {"caps_applied": 0, "max_caps": 0},
        "rmt": {},
        "variance": {"enabled": False},
        "policies": {},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
        },
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": 10.0,
            "final": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
            "gating_basis": "point",
        },
        "auto": {"tier": "balanced", "probes_used": 0, "target_pm_ratio": None},
    }


def test_system_overhead_zero_renders_na() -> None:
    cert = _base_cert()
    cert["system_overhead"] = {
        "latency_ms_p50": {"baseline": 0.0, "edited": 0.0},
        "throughput_sps": {"baseline": 0.0, "edited": 0.0},
    }
    md = render_certificate_markdown(cert)
    # Expect System Overhead section to render N/A values instead of 0/0/0
    assert "## System Overhead" in md
    assert "N/A" in md
