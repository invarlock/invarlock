from __future__ import annotations

import copy
import json

import invarlock.reporting.certificate_schema as schema_mod


def test_load_validation_allowlist_default(monkeypatch):
    target = (
        schema_mod.Path(schema_mod.__file__).resolve().parents[3]
        / "contracts"
        / "validation_keys.json"
    )
    target_str = str(target.resolve())
    real_exists = schema_mod.Path.exists

    def fake_exists(self):
        try:
            current = str(self.resolve())
        except Exception:
            current = str(self)
        if current == target_str:
            return False
        return real_exists(self)

    monkeypatch.setattr(schema_mod.Path, "exists", fake_exists, raising=False)
    allowlist = schema_mod._load_validation_allowlist()
    assert allowlist == set(schema_mod._VALIDATION_ALLOWLIST_DEFAULT)


def test_load_validation_allowlist_reads_file(monkeypatch, tmp_path):
    target = (
        schema_mod.Path(schema_mod.__file__).resolve().parents[3]
        / "contracts"
        / "validation_keys.json"
    )
    target_str = str(target.resolve())
    fake_file = tmp_path / "validation_keys.json"
    fake_file.write_text(json.dumps(["primary_metric_acceptable", "custom_flag"]))

    real_exists = schema_mod.Path.exists
    real_read = schema_mod.Path.read_text

    def fake_exists(self):
        try:
            current = str(self.resolve())
        except Exception:
            current = str(self)
        if current == target_str:
            return True
        return real_exists(self)

    def fake_read(self, *args, **kwargs):
        try:
            current = str(self.resolve())
        except Exception:
            current = str(self)
        if current == target_str:
            return fake_file.read_text()
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(schema_mod.Path, "exists", fake_exists, raising=False)
    monkeypatch.setattr(schema_mod.Path, "read_text", fake_read, raising=False)
    allowlist = schema_mod._load_validation_allowlist()
    assert allowlist == {"primary_metric_acceptable", "custom_flag"}


def test_load_validation_allowlist_non_list_payload(monkeypatch, tmp_path):
    target = (
        schema_mod.Path(schema_mod.__file__).resolve().parents[3]
        / "contracts"
        / "validation_keys.json"
    )
    target_str = str(target.resolve())
    fake_file = tmp_path / "validation_keys.json"
    fake_file.write_text(json.dumps({"oops": True}))

    real_exists = schema_mod.Path.exists
    real_read = schema_mod.Path.read_text

    def fake_exists(self):
        try:
            current = str(self.resolve())
        except Exception:
            current = str(self)
        if current == target_str:
            return True
        return real_exists(self)

    def fake_read(self, *args, **kwargs):
        try:
            current = str(self.resolve())
        except Exception:
            current = str(self)
        if current == target_str:
            return fake_file.read_text()
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(schema_mod.Path, "exists", fake_exists, raising=False)
    monkeypatch.setattr(schema_mod.Path, "read_text", fake_read, raising=False)
    allowlist = schema_mod._load_validation_allowlist()
    assert allowlist == set(schema_mod._VALIDATION_ALLOWLIST_DEFAULT)


def test_validate_with_jsonschema_handles_missing_library(monkeypatch):
    monkeypatch.setattr(schema_mod, "jsonschema", None, raising=False)
    assert schema_mod._validate_with_jsonschema({"schema_version": "v1"})


def test_validate_with_jsonschema_failure(monkeypatch):
    class BrokenSchema:
        @staticmethod
        def validate(*_args, **_kwargs):
            raise ValueError("bad")

    monkeypatch.setattr(schema_mod, "jsonschema", BrokenSchema, raising=False)
    assert schema_mod._validate_with_jsonschema({"schema_version": "v1"}) is False


def test_validate_certificate_schema_version_mismatch():
    assert schema_mod.validate_certificate({"schema_version": "v0"}) is False


def test_validate_certificate_fallback_and_allowlist(monkeypatch):
    cert = {
        "schema_version": schema_mod.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-123",
        "primary_metric": {"final": 1.0},
        "validation": {"custom_flag": True},
    }

    orig_schema = copy.deepcopy(
        schema_mod.CERTIFICATE_JSON_SCHEMA["properties"]["validation"]
    )

    monkeypatch.setattr(
        schema_mod, "_load_validation_allowlist", lambda: {"custom_flag"}
    )
    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda _: False)

    try:
        assert schema_mod.validate_certificate(cert) is True
        vspec = schema_mod.CERTIFICATE_JSON_SCHEMA["properties"]["validation"]
        assert vspec["properties"] == {"custom_flag": {"type": "boolean"}}
        assert vspec["additionalProperties"] is False
    finally:
        schema_mod.CERTIFICATE_JSON_SCHEMA["properties"]["validation"] = orig_schema


def test_validate_certificate_rejects_non_boolean_flags(monkeypatch):
    cert = {
        "schema_version": schema_mod.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "run-123",
        "primary_metric": {"final": 1.0},
        "validation": {"primary_metric_acceptable": "yes"},
    }
    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda _: True)
    assert schema_mod.validate_certificate(cert) is False


def test_load_validation_allowlist_handles_exception(monkeypatch):
    from pathlib import Path

    def boom(self):
        raise RuntimeError("fail")

    monkeypatch.setattr(Path, "resolve", boom, raising=False)
    allowlist = schema_mod._load_validation_allowlist()
    assert allowlist == set(schema_mod._VALIDATION_ALLOWLIST_DEFAULT)


def test_validate_certificate_allowlist_error(monkeypatch):
    cert = {
        "schema_version": schema_mod.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "r1",
        "primary_metric": {"kind": "ppl_causal", "final": 1.0},
    }
    monkeypatch.setattr(
        schema_mod,
        "_load_validation_allowlist",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda _: True)
    assert schema_mod.validate_certificate(cert) is True


def test_validate_certificate_handles_missing_validation_schema(monkeypatch):
    cert = {
        "schema_version": schema_mod.CERTIFICATE_SCHEMA_VERSION,
        "run_id": "r2",
        "primary_metric": {"kind": "ppl_causal", "final": 1.0},
    }
    monkeypatch.setitem(schema_mod.CERTIFICATE_JSON_SCHEMA, "properties", None)
    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda _: True)
    assert schema_mod.validate_certificate(cert) is True
