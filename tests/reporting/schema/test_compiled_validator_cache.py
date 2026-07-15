from __future__ import annotations

from types import SimpleNamespace

from invarlock import evidence_pack_integrity as manifest_mod
from invarlock import public_contracts
from invarlock.reporting import report_schema
from invarlock.reporting import verify_check_helpers_metrics as verify_metrics


class _CountingValidator:
    compilations = 0
    validations = 0

    @classmethod
    def check_schema(cls, _schema: object) -> None:
        return None

    def __init__(self, _schema: object) -> None:
        type(self).compilations += 1

    def validate(self, _instance: object) -> None:
        type(self).validations += 1


def _counting_jsonschema() -> SimpleNamespace:
    return SimpleNamespace(
        validators=SimpleNamespace(validator_for=lambda _schema: _CountingValidator),
        validate=lambda **_kwargs: None,
    )


def test_report_validator_compiles_once_for_repeated_validations(monkeypatch) -> None:
    _CountingValidator.compilations = 0
    _CountingValidator.validations = 0
    report_schema._compiled_report_validator.cache_clear()
    monkeypatch.setattr(report_schema, "jsonschema", _counting_jsonschema())
    monkeypatch.setattr(
        report_schema,
        "load_validation_allowlist_strict",
        lambda: {"primary_metric_acceptable"},
    )
    payload = {
        "schema_version": report_schema.REPORT_SCHEMA_VERSION,
        "run_id": "cache-test",
        "primary_metric": {"kind": "accuracy"},
        "validation": {"primary_metric_acceptable": True},
    }

    decisions = [report_schema.validate_report(payload) for _ in range(20)]

    assert decisions == [True] * 20
    assert _CountingValidator.compilations == 1
    assert _CountingValidator.validations == 20
    assert 1 - (_CountingValidator.compilations / len(decisions)) >= 0.9


def test_report_validator_fallback_still_applies_validation_allowlist(
    monkeypatch,
) -> None:
    captured_schemas: list[dict[str, object]] = []

    def validate(*, instance: object, schema: dict[str, object]) -> None:
        del instance
        captured_schemas.append(schema)

    monkeypatch.setattr(report_schema, "jsonschema", SimpleNamespace(validate=validate))
    monkeypatch.setattr(
        report_schema,
        "load_validation_allowlist_strict",
        lambda: {"primary_metric_acceptable"},
    )
    payload = {
        "schema_version": report_schema.REPORT_SCHEMA_VERSION,
        "run_id": "fallback-test",
        "primary_metric": {"kind": "accuracy"},
        "validation": {"primary_metric_acceptable": True},
    }

    assert report_schema.validate_report(payload) is True
    validation_schema = captured_schemas[0]["properties"]["validation"]
    assert validation_schema["additionalProperties"] is False
    assert set(validation_schema["properties"]) == {"primary_metric_acceptable"}


def test_manifest_validator_compiles_and_loads_once(monkeypatch) -> None:
    _CountingValidator.compilations = 0
    _CountingValidator.validations = 0
    manifest_mod._compiled_manifest_validator.cache_clear()
    schema_loads = 0

    def load_schema() -> dict[str, object]:
        nonlocal schema_loads
        schema_loads += 1
        return {"type": "object"}

    schema_runtime = _counting_jsonschema()
    monkeypatch.setattr(manifest_mod, "jsonschema", schema_runtime)
    monkeypatch.setattr(manifest_mod, "_DEFAULT_MANIFEST_SCHEMA_LOADER", load_schema)
    monkeypatch.setattr(manifest_mod, "load_evidence_pack_manifest_schema", load_schema)
    payload = {
        "format": manifest_mod.EVIDENCE_PACK_FORMAT,
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": "a" * 64,
    }

    decisions = [manifest_mod.validate_manifest_payload(payload) for _ in range(20)]

    assert decisions == [[]] * 20
    assert schema_loads == 1
    assert _CountingValidator.compilations == 1
    assert _CountingValidator.validations == 20


def test_strict_verify_validator_compiles_once(monkeypatch) -> None:
    _CountingValidator.compilations = 0
    _CountingValidator.validations = 0
    verify_metrics._compiled_canonical_report_validator.cache_clear()
    schema_runtime = _counting_jsonschema()
    monkeypatch.setattr(report_schema, "jsonschema", schema_runtime)
    payload = {"schema_version": report_schema.REPORT_SCHEMA_VERSION}

    decisions = [
        verify_metrics._validate_report_schema_strict(payload) for _ in range(20)
    ]

    assert decisions == [True] * 20
    assert _CountingValidator.compilations == 1
    assert _CountingValidator.validations == 20


def test_public_schema_loader_returns_fresh_mutable_payloads() -> None:
    first = public_contracts.load_evidence_pack_manifest_schema()
    first["properties"]["format"]["const"] = "mutated-by-caller"

    second = public_contracts.load_evidence_pack_manifest_schema()

    assert second["properties"]["format"]["const"] == "evidence-pack-v1"


def test_compiled_manifest_preserves_schema_error_text() -> None:
    manifest_mod._compiled_manifest_validator.cache_clear()
    _schema, validator = manifest_mod._compiled_manifest_validator(
        id(manifest_mod.jsonschema)
    )
    assert validator is not None
    payload = {
        "format": "wrong-format",
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": "a" * 64,
    }
    expected_error = None
    try:
        validator.validate(payload)
    except manifest_mod.jsonschema_validation_error_types() as exc:
        expected_error = f"manifest schema validation failed: {exc}"

    assert expected_error is not None
    assert manifest_mod.validate_manifest_payload(payload) == [expected_error]
