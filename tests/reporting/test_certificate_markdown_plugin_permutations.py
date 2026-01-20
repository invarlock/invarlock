from __future__ import annotations

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def _mk_minimal_report() -> dict:
    return {
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 1,
            "final_n": 1,
        },
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.0],
            }
        },
        "evaluation_windows": {
            "preview": {
                "window_ids": [1],
                "logloss": [2.302585093],
                "token_counts": [1],
            },
            "final": {"window_ids": [2], "logloss": [2.302585093], "token_counts": [1]},
        },
        "edit": {"name": "noop"},
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def test_plugin_provenance_adapter_edit_only():
    rep = _mk_minimal_report()
    base = {
        **_mk_minimal_report(),
        "edit": {
            "name": "baseline",
            "plan_digest": "baseline_noop",
            "deltas": {"params_changed": 0},
        },
    }
    cert = make_certificate(rep, base)
    cert["plugins"] = {
        "adapter": {
            "name": "HF_Causal",
            "version": "1.0",
            "module": "invarlock.adapters.hf_causal",
        },
        "edit": {
            "name": "quant_rtn",
            "version": "0.1",
            "module": "invarlock.edits.quant_rtn",
        },
        "guards": [],
    }
    md = render_certificate_markdown(cert)
    assert "## Plugin Provenance" in md
    assert "Adapter:" in md and "Edit:" in md
    # No guards list when empty
    assert "- Guards:" not in md


def test_plugin_provenance_guards_only():
    rep = _mk_minimal_report()
    base = {
        **_mk_minimal_report(),
        "edit": {
            "name": "baseline",
            "plan_digest": "baseline_noop",
            "deltas": {"params_changed": 0},
        },
    }
    cert = make_certificate(rep, base)
    cert["plugins"] = {
        "guards": [
            {
                "name": "SpectralGuard",
                "version": "2.0",
                "module": "invarlock.guards.spectral",
            },
            {
                "name": "RMTGuard",
                "version": "1.2",
                "module": "invarlock.guards.rmt",
                "entry_point": "RMT",
            },
        ]
    }
    md = render_certificate_markdown(cert)
    assert "## Plugin Provenance" in md
    assert "- Guards:" in md
    assert "SpectralGuard" in md and "RMTGuard" in md
