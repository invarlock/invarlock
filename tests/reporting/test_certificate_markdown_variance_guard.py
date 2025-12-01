from __future__ import annotations

from invarlock.reporting.certificate import make_certificate
from invarlock.reporting.render import render_certificate_markdown


def _mk_base() -> tuple[dict, dict]:
    report = {
        "meta": {"model_id": "m", "adapter": "hf", "device": "cpu", "seed": 1},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 1,
            "final_n": 1,
        },
        "edit": {
            "name": "noop",
            "plan_digest": "d",
            "deltas": {"params_changed": 0, "layers_modified": 0},
        },
        "guards": [],
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
        "artifacts": {"events_path": "", "logs_path": ""},
    }
    baseline = {
        "meta": {"auto": {"tier": "balanced"}},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    return report, baseline


def test_variance_guard_enabled_and_disabled_rendering() -> None:
    rep, base = _mk_base()
    cert = make_certificate(rep, base)
    # Enabled path
    cert["variance"] = {"enabled": True, "gain": 0.02}
    md = render_certificate_markdown(cert)
    assert "Variance Guard" in md and "Enabled:" in md and "Gain:" in md

    # Disabled with policy explanation
    cert2 = make_certificate(rep, base)
    cert2["variance"] = {"enabled": False}
    cert2.setdefault("policies", {}).setdefault("variance", {})
    cert2["policies"]["variance"]["min_effect_lognll"] = 0.001
    md2 = render_certificate_markdown(cert2)
    assert "Variance Guard" in md2 and "Disabled" in md2 and "Predictive gate" in md2
