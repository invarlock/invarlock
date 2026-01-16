from __future__ import annotations

from copy import deepcopy

from invarlock.reporting import render as render_mod


def _base_certificate() -> dict[str, object]:
    return {
        "schema_version": "v1",
        "run_id": "run-guard",
        "artifacts": {"generated_at": "2024-01-01T00:00:00Z"},
        "plugins": {"guards": []},
        "meta": {},
        "dataset": {
            "provider": "demo",
            "seq_len": 16,
            "windows": {"preview": 2, "final": 4},
        },
        "invariants": {"failures": []},
        "variance": {"enabled": True, "summary": {"stable": True}, "policy": {}},
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 1.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "guard_overhead_acceptable": True,
        },
        "policies": {"active": []},
        "auto": {"tier": "balanced", "probes_used": ["spectral"]},
        "guard_overhead": {"evaluated": True, "ok": True},
    }


def test_render_certificate_markdown_includes_guard_sections():
    cert = deepcopy(_base_certificate())
    cert["plugins"]["guards"] = [
        {"name": "spectral", "version": "1.0", "module": "invarlock.guards.spectral"}
    ]
    cert["spectral"] = {
        "multiple_testing": {"alpha": 0.05, "method": "holm"},
        "sigma_quantile": 2.5,
        "deadband": 0.25,
        "max_caps": 3,
        "caps_applied": 1,
        "summary": {"caps_exceeded": True},
        "caps_applied_by_family": {"attn": 2},
        "family_caps": {"attn": {"kappa": 0.85}},
        "family_z_quantiles": {
            "attn": {"q95": 1.2, "q99": 2.3, "max": 2.8, "count": 5}
        },
        "policy": {"family_caps": {"attn": {"kappa": 0.8}}},
        "top_z_scores": {"attn": [{"module": "attn.0", "z": 3.1}]},
    }
    cert["rmt"] = {
        "families": {
            "mlp": {"epsilon": 0.2, "bare": 10, "guarded": 6},
        }
    }

    report = render_mod.render_certificate_markdown(cert)

    assert "Spectral Guard" in report
    assert "Multiple Testing" in report
    assert "Caps Applied" in report
    assert "Top |z| per family" in report
    assert "RMT Guard" in report
    assert "| Family | ε_f" in report


def test_render_certificate_markdown_handles_sparse_spectral_sections():
    cert = deepcopy(_base_certificate())
    cert["plugins"]["guards"] = [
        {"name": "spectral", "version": "1.0", "module": "invarlock.guards.spectral"}
    ]
    cert["spectral"] = {
        "multiple_testing": {},
        "sigma_quantile": None,
        "deadband": None,
        "max_caps": None,
        "caps_applied": 0,
        "summary": {"caps_exceeded": False},
        "caps_applied_by_family": {},
        "family_z_quantiles": {},
        "policy": {},
        "top_z_scores": {"attn": [{"module": "attn.block", "z": None}]},
    }
    rendered = render_mod.render_certificate_markdown(cert)

    assert "Spectral Guard" in rendered
    assert "Spectral Summary" not in rendered  # dropped when no numeric knobs
    assert "attn.block (|z|=n/a)" in rendered


def test_render_certificate_markdown_rmt_handles_non_numeric_counts():
    cert = deepcopy(_base_certificate())
    cert["plugins"]["guards"] = [
        {"name": "rmt", "version": "1.0", "module": "invarlock.guards.rmt"}
    ]
    cert["spectral"] = {"caps_applied": 0, "max_caps": 0, "summary": {}}
    cert["rmt"] = {
        "families": {
            "mlp": {"epsilon": 0.2, "bare": "ten", "guarded": "five"},
        },
        "delta_total": "unknown",
        "stable": False,
    }

    rendered = render_mod.render_certificate_markdown(cert)

    assert "| mlp | 0.200 | - | - | - |" in rendered
    assert "- Status: \u274c FAIL" in rendered
    assert "- Families: 1" in rendered
