from __future__ import annotations

from types import SimpleNamespace

from invarlock.reporting import certificate as C


def test_validate_certificate_jsonschema_true(monkeypatch):
    # Provide a dummy jsonschema with validate() that returns success
    monkeypatch.setattr(
        C, "jsonschema", SimpleNamespace(validate=lambda instance, schema: None)
    )
    cert = {
        "schema_version": C.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "rid",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "p",
            "seq_len": 8,
            "windows": {"preview": 0, "final": 0},
        },
    }
    assert C.validate_certificate(cert) is True
