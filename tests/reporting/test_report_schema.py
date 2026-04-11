from __future__ import annotations

import copy
import importlib

import pytest

import invarlock.reporting.report_schema as schema_mod
import invarlock.reporting.report_validation_allowlist as allowlist_mod
from invarlock.core import metric_kind_contract as metric_kind_mod


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
    assert schema_mod._validate_with_jsonschema({"schema_version": "v1"}) is False


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


def test_validate_report_rejects_payload_when_schema_validation_fails(monkeypatch):
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-123",
        "primary_metric": {"kind": "accuracy", "final": 1.0},
        "validation": {"custom_flag": True},
    }

    orig_schema = copy.deepcopy(
        schema_mod.REPORT_JSON_SCHEMA["properties"]["validation"]
    )

    monkeypatch.setattr(
        allowlist_mod, "load_validation_allowlist", lambda: {"custom_flag"}
    )
    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda *_args: False)

    try:
        assert schema_mod.validate_report(cert) is False
        vspec = schema_mod.REPORT_JSON_SCHEMA["properties"]["validation"]
        assert vspec == orig_schema
    finally:
        schema_mod.REPORT_JSON_SCHEMA["properties"]["validation"] = orig_schema


def test_validate_report_does_not_leak_allowlist_mutations_between_calls(
    monkeypatch,
):
    custom = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-123",
        "primary_metric": {"kind": "accuracy", "final": 1.0},
        "validation": {"custom_flag": True},
    }
    standard = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-456",
        "primary_metric": {"kind": "accuracy", "final": 1.0},
        "validation": {"primary_metric_acceptable": True},
    }

    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda *_args: True)
    monkeypatch.setattr(
        allowlist_mod, "load_validation_allowlist", lambda: {"custom_flag"}
    )
    assert schema_mod.validate_report(custom) is True

    monkeypatch.setattr(
        allowlist_mod,
        "load_validation_allowlist",
        lambda: {"primary_metric_acceptable"},
    )
    assert schema_mod.validate_report(standard) is True


def test_validate_report_rejects_non_boolean_flags(monkeypatch):
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-123",
        "primary_metric": {"kind": "accuracy", "final": 1.0},
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


def test_validate_report_rejects_unknown_primary_metric_kind(monkeypatch):
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-unknown-kind",
        "primary_metric": {"kind": "vqa_accuracy", "final": 1.0},
    }

    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda *_args: True)
    assert schema_mod.validate_report(cert) is False


def test_validate_report_rejects_non_concrete_primary_metric_kind(monkeypatch):
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-auto-kind",
        "primary_metric": {"kind": "auto", "final": 1.0},
    }

    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda *_args: True)
    assert schema_mod.validate_report(cert) is False


def test_validate_report_rejects_non_mapping_primary_metric(monkeypatch):
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-bad-primary-metric",
        "primary_metric": ["accuracy"],
    }

    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda *_args: True)
    assert schema_mod.validate_report(cert) is False


def test_validate_report_rejects_unknown_validation_key(monkeypatch):
    cert = {
        "schema_version": schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": "run-bad-validation-key",
        "primary_metric": {"kind": "accuracy", "final": 1.0},
        "validation": {"unexpected_flag": True},
    }

    monkeypatch.setattr(schema_mod, "_validate_with_jsonschema", lambda *_args: True)
    monkeypatch.setattr(
        allowlist_mod,
        "load_validation_allowlist",
        lambda: {"primary_metric_acceptable"},
    )
    assert schema_mod.validate_report(cert) is False


def test_report_schema_import_tolerates_allowlist_bootstrap_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(
            allowlist_mod,
            "apply_validation_allowlist_schema",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
        )
        reloaded = importlib.reload(schema_mod)
        assert "validation" in reloaded.REPORT_JSON_SCHEMA["properties"]

    importlib.reload(schema_mod)


def test_report_schema_import_tolerates_metric_kind_bootstrap_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(
            metric_kind_mod,
            "load_metric_kind_catalog",
            lambda: (_ for _ in ()).throw(
                metric_kind_mod.MetricKindContractError("boom")
            ),
        )
        reloaded = importlib.reload(schema_mod)
        kind_schema = reloaded.REPORT_JSON_SCHEMA["properties"]["primary_metric"][
            "properties"
        ]["kind"]
        assert isinstance(kind_schema, dict)

    importlib.reload(schema_mod)
