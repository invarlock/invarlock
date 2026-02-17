from invarlock.reporting.report_builder import make_report


def test_cert_provenance_records_tiny_relax():
    # Minimal report that passes relaxed path
    report = {
        "meta": {"model_id": "m", "adapter": "hf_causal", "device": "cpu", "seed": 1},
        "context": {"run": {"tiny_relax": True}},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [2.3]}},
    }
    baseline = {
        "meta": {"model_id": "m"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [2.3]}},
    }
    cert = make_report(report, baseline)
    assert cert.get("auto", {}).get("tiny_relax") is True
    assert "tiny_relax" in (cert.get("provenance", {}).get("flags", []))
