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
            "name": "quant_rtn",
            "plan_digest": "d",
            "deltas": {"params_changed": 10, "layers_modified": 2},
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


def test_markdown_quant_rtn_linear_modules_quantized_row() -> None:
    rep, base = _mk_base()
    cert = make_certificate(rep, base)
    # Provide structure diagnostics to trigger quant_rtn detail line
    cert["structure"] = {
        "params_changed": 10,
        "layers_modified": 2,
        "bitwidths": [8, 8, 8, 8],
        "compression_diagnostics": {
            "target_analysis": {"modules_eligible": 6, "modules_modified": 4}
        },
    }
    md = render_certificate_markdown(cert)
    assert "Linear Modules Quantized" in md
