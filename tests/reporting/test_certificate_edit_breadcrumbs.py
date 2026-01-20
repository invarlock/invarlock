from __future__ import annotations

from invarlock.reporting.certificate import make_certificate


def _mk_minimal_report_with_windows() -> dict:
    return {
        "meta": {"model_id": "m", "adapter": "hf_causal", "device": "cpu", "seed": 42},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": (1.0, 1.0),
            }
        },
        "evaluation_windows": {
            "preview": {"window_ids": [1], "logloss": [1.0], "token_counts": [10]},
            "final": {"window_ids": [2], "logloss": [1.0], "token_counts": [10]},
        },
        "guards": [],
        "artifacts": {"events_path": "", "logs_path": ""},
    }


def test_edit_digest_quantization():
    report = _mk_minimal_report_with_windows()
    # Quantization run
    report["edit"] = {"name": "quant_rtn", "config": {"bitwidth": 4, "scope": "ffn"}}
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    cert = make_certificate(report, baseline)
    ed = cert.get("provenance", {}).get("edit_digest", {})
    assert ed.get("family") == "quantization"
    ih = ed.get("impl_hash")
    assert isinstance(ih, str) and len(ih) >= 16
    assert ed.get("version") == 1


def test_edit_digest_cert_only():
    report = _mk_minimal_report_with_windows()
    # Cert-only (no in-run edit)
    report["edit"] = {"name": "noop"}
    baseline = {
        "run_id": "b",
        "model_id": "m",
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
    }
    cert = make_certificate(report, baseline)
    ed = cert.get("provenance", {}).get("edit_digest", {})
    assert ed.get("family") == "cert_only"
    ih = ed.get("impl_hash")
    assert isinstance(ih, str) and len(ih) >= 16
