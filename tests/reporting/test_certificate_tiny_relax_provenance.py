from invarlock.reporting.certificate import make_certificate


def test_cert_provenance_records_tiny_relax(monkeypatch):
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    # Minimal report that passes relaxed path
    report = {
        "meta": {"model_id": "m", "adapter": "hf_gpt2", "device": "cpu", "seed": 1},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [2.3]}},
    }
    baseline = {
        "meta": {"model_id": "m"},
        "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 10.0}},
        "evaluation_windows": {"final": {"window_ids": [1], "logloss": [2.3]}},
    }
    cert = make_certificate(report, baseline)
    assert cert.get("auto", {}).get("tiny_relax") is True
    assert "tiny_relax" in (cert.get("provenance", {}).get("flags", []))
