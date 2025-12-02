from __future__ import annotations

from invarlock.reporting import certificate_schema as schema_mod


def _base_certificate() -> dict:
    return {
        "schema_version": schema_mod.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-1234",
        "artifacts": {},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "synthetic",
            "seq_len": 16,
            "windows": {"preview": 2, "final": 2},
        },
        "primary_metric": {"kind": "accuracy", "final": 0.9},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }


def test_validate_certificate_accepts_valid_payload():
    cert = _base_certificate()
    assert schema_mod.validate_certificate(cert) is True


def test_validate_certificate_fallback_when_schema_fails(monkeypatch):
    monkeypatch.setattr(
        schema_mod, "_validate_with_jsonschema", lambda cert: False, raising=False
    )
    cert = {
        "schema_version": schema_mod.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "fallback",
        "primary_metric": {"kind": "ppl_mlm"},
    }
    assert schema_mod.validate_certificate(cert) is True


def test_validate_certificate_rejects_bad_validation_values():
    cert = _base_certificate()
    cert["validation"]["primary_metric_acceptable"] = "yes"
    assert schema_mod.validate_certificate(cert) is False


def test_validate_with_jsonschema_handles_missing_library(monkeypatch):
    monkeypatch.setattr(schema_mod, "jsonschema", None, raising=False)
    assert schema_mod._validate_with_jsonschema(_base_certificate()) is True
