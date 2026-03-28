from __future__ import annotations

import copy

import invarlock.reporting.report_schema as schema_mod
import invarlock.reporting.report_validation_allowlist as allowlist_mod


def test_load_validation_allowlist_default(monkeypatch):
    def _missing(_filename: str):
        raise FileNotFoundError

    monkeypatch.setattr(allowlist_mod, "load_json_contract", _missing)
    allowlist = allowlist_mod.load_validation_allowlist()
    assert allowlist == set(schema_mod._VALIDATION_ALLOWLIST_DEFAULT)


def test_load_validation_allowlist_reads_file(monkeypatch):
    monkeypatch.setattr(
        allowlist_mod,
        "load_json_contract",
        lambda _filename: ["primary_metric_acceptable", "custom_flag"],
    )
    allowlist = allowlist_mod.load_validation_allowlist()
    assert allowlist == {"primary_metric_acceptable", "custom_flag"}


def test_load_validation_allowlist_non_list_payload(monkeypatch):
    monkeypatch.setattr(
        allowlist_mod, "load_json_contract", lambda _filename: {"oops": True}
    )
    allowlist = allowlist_mod.load_validation_allowlist()
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


def test_validate_with_jsonschema_success(monkeypatch):
    class ValidSchema:
        @staticmethod
        def validate(*_args, **_kwargs):
            return None

    monkeypatch.setattr(schema_mod, "jsonschema", ValidSchema, raising=False)
    assert schema_mod._validate_with_jsonschema({"schema_version": "v1"}) is True


def test_validate_report_schema_version_mismatch():
    assert schema_mod.validate_report({"schema_version": "v0"}) is False


def test_validate_report_fallback_and_allowlist(monkeypatch):
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-123",
        "primary_metric": {"final": 1.0},
        "validation": {"custom_flag": True},
    }

    orig_schema = copy.deepcopy(
        schema_mod.REPORT_JSON_SCHEMA["properties"]["validation"]
    )

    monkeypatch.setattr(
        allowlist_mod, "load_validation_allowlist", lambda: {"custom_flag"}
    )
    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda _: False)

    try:
        assert schema_mod.validate_report(cert) is True
        vspec = schema_mod.REPORT_JSON_SCHEMA["properties"]["validation"]
        assert vspec["properties"] == {"custom_flag": {"type": "boolean"}}
        assert vspec["additionalProperties"] is False
    finally:
        schema_mod.REPORT_JSON_SCHEMA["properties"]["validation"] = orig_schema


def test_validate_report_rejects_non_boolean_flags(monkeypatch):
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-123",
        "primary_metric": {"final": 1.0},
        "validation": {"primary_metric_acceptable": "yes"},
    }
    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda _: True)
    assert schema_mod.validate_report(cert) is False


def test_load_validation_allowlist_handles_exception(monkeypatch):
    def boom(_filename: str):
        raise RuntimeError("fail")

    monkeypatch.setattr(allowlist_mod, "load_json_contract", boom)
    allowlist = allowlist_mod.load_validation_allowlist()
    assert allowlist == set(schema_mod._VALIDATION_ALLOWLIST_DEFAULT)


def test_validate_report_allowlist_error(monkeypatch):
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "r1",
        "primary_metric": {"kind": "ppl_causal", "final": 1.0},
    }
    monkeypatch.setattr(
        allowlist_mod,
        "load_validation_allowlist",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda _: True)
    assert schema_mod.validate_report(cert) is True


def test_validate_report_handles_missing_validation_schema(monkeypatch):
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "r2",
        "primary_metric": {"kind": "ppl_causal", "final": 1.0},
    }
    monkeypatch.setitem(schema_mod.REPORT_JSON_SCHEMA, "properties", None)
    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda _: True)
    assert schema_mod.validate_report(cert) is True


def test_validate_report_handles_non_mapping_validation_spec(monkeypatch):
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "r3",
        "primary_metric": {"kind": "ppl_causal", "final": 1.0},
    }
    original = copy.deepcopy(schema_mod.REPORT_JSON_SCHEMA["properties"])
    try:
        schema_mod.REPORT_JSON_SCHEMA["properties"] = {"validation": []}
        monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda _: True)
        assert schema_mod.validate_report(cert) is True
    finally:
        schema_mod.REPORT_JSON_SCHEMA["properties"] = original


def test_validate_report_handles_type_error_from_validation_block(monkeypatch):
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "r4",
        "primary_metric": {"kind": "ppl_causal", "final": 1.0},
        "validation": None,
    }
    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda _: True)
    assert schema_mod.validate_report(cert) is False
