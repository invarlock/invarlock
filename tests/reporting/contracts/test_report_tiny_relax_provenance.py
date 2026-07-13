from invarlock.reporting.report_make import make_report
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)


def test_cert_provenance_records_tiny_relax():
    # Minimal report that passes relaxed path
    report = canonical_run_report(
        {
            "meta": {
                "model_id": "m",
                "adapter": "hf_causal",
                "device": "cpu",
                "seed": 1,
                "auto": {"tier": "balanced"},
            },
            "context": {
                "profile": "dev",
                "assurance": {"mode": "off"},
                "run": {"tiny_relax": True},
            },
            "data": {
                "dataset": "unit",
                "split": "validation",
                "seq_len": 8,
                "stride": 8,
                "preview_n": 1,
                "final_n": 1,
            },
            "edit": {"name": "structured"},
            "guards": [],
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 10.0}
            },
            "evaluation_windows": {"final": {"window_ids": [1], "logloss": [2.3]}},
        }
    )
    baseline = canonical_baseline(
        {
            **report,
            "edit": {"name": "noop"},
        }
    )
    cert = make_report(report, baseline)
    assert cert.get("auto", {}).get("tiny_relax") is True
    assert "tiny_relax" in (cert.get("provenance", {}).get("flags", []))


def test_cert_provenance_ignores_env_tiny_relax(monkeypatch):
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    report = canonical_run_report(
        {
            "meta": {
                "model_id": "m",
                "adapter": "hf_causal",
                "device": "cpu",
                "seed": 1,
                "auto": {"tier": "balanced"},
            },
            "context": {"profile": "dev", "assurance": {"mode": "off"}},
            "data": {
                "dataset": "unit",
                "split": "validation",
                "seq_len": 8,
                "stride": 8,
                "preview_n": 1,
                "final_n": 1,
            },
            "edit": {"name": "structured"},
            "guards": [],
            "metrics": {
                "primary_metric": {"kind": "ppl_causal", "preview": 10.0, "final": 10.0}
            },
            "evaluation_windows": {"final": {"window_ids": [1], "logloss": [2.3]}},
        }
    )
    baseline = canonical_baseline(
        {
            **report,
            "edit": {"name": "noop"},
        }
    )
    cert = make_report(report, baseline)
    assert cert.get("auto", {}).get("tiny_relax") is not True
    assert "tiny_relax" not in (cert.get("provenance", {}).get("flags", []))
