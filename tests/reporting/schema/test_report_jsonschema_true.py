from __future__ import annotations

from types import SimpleNamespace

from invarlock.reporting import report_schema as C


def test_validate_evaluation_report_jsonschema_true(monkeypatch):
    # Provide a dummy jsonschema with validate() that returns success
    monkeypatch.setattr(
        C, "jsonschema", SimpleNamespace(validate=lambda instance, schema: None)
    )
    cert = {
        "schema_version": C.REPORT_SCHEMA_VERSION,
        "run_id": "rid",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {"preview": 0, "final": 0, "stats": {}},
        },
        "primary_metric": {"kind": "ppl_causal", "final": 1.0},
    }
    assert C.validate_report(cert) is True
