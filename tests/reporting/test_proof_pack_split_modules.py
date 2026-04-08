from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import invarlock.proof_pack_integrity as proof_pack_integrity_mod
import invarlock.proof_pack_manifest as proof_pack_manifest_mod


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_proof_pack_manifest_jsonschema_helper_uses_exceptions_and_fallback_attrs(
    monkeypatch,
) -> None:
    class _ValidationError(Exception):
        pass

    class _SchemaError(Exception):
        pass

    monkeypatch.setattr(proof_pack_manifest_mod, "jsonschema", None, raising=False)
    assert proof_pack_manifest_mod._jsonschema_validation_error_types() == ()

    jsonschema_stub = SimpleNamespace(
        exceptions=SimpleNamespace(ValidationError=_ValidationError),
        SchemaError=_SchemaError,
    )
    monkeypatch.setattr(
        proof_pack_manifest_mod, "jsonschema", jsonschema_stub, raising=False
    )

    assert proof_pack_manifest_mod._jsonschema_validation_error_types() == (
        _ValidationError,
        _SchemaError,
    )


def test_proof_pack_manifest_validate_manifest_covers_schema_failure_and_direct_validate(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{invalid", encoding="utf-8")
    errors = proof_pack_manifest_mod.validate_manifest(manifest_path)
    assert "manifest is not valid JSON" in errors[0]

    _write_json(
        manifest_path,
        {
            "format": proof_pack_manifest_mod.PROOF_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )
    monkeypatch.setattr(
        proof_pack_manifest_mod,
        "load_proof_pack_manifest_schema",
        lambda: {"type": "object"},
        raising=True,
    )

    class _ValidationError(Exception):
        pass

    def _fail_validate(*_args, **_kwargs):
        raise _ValidationError("schema boom")

    monkeypatch.setattr(
        proof_pack_manifest_mod,
        "jsonschema",
        SimpleNamespace(
            exceptions=SimpleNamespace(ValidationError=_ValidationError),
            validate=_fail_validate,
        ),
        raising=False,
    )
    errors = proof_pack_manifest_mod.validate_manifest(manifest_path)
    assert errors == ["manifest schema validation failed: schema boom"]

    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        proof_pack_manifest_mod,
        "jsonschema",
        SimpleNamespace(
            ValidationError=_ValidationError,
            validate=lambda instance, schema: calls.append((instance, schema)),
        ),
        raising=False,
    )
    assert proof_pack_manifest_mod.validate_manifest(manifest_path) == []
    assert calls == [
        (
            json.loads(manifest_path.read_text(encoding="utf-8")),
            {"type": "object"},
        )
    ]

    direct_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        proof_pack_manifest_mod,
        "jsonschema",
        SimpleNamespace(
            validate=lambda instance, schema: direct_calls.append((instance, schema)),
        ),
        raising=False,
    )
    assert proof_pack_manifest_mod.validate_manifest(manifest_path) == []
    assert direct_calls == [
        (
            json.loads(manifest_path.read_text(encoding="utf-8")),
            {"type": "object"},
        )
    ]


def test_proof_pack_integrity_jsonschema_helper_uses_exceptions_and_fallback_attrs(
    monkeypatch,
) -> None:
    class _ValidationError(Exception):
        pass

    class _SchemaError(Exception):
        pass

    monkeypatch.setattr(proof_pack_integrity_mod, "jsonschema", None, raising=False)
    assert proof_pack_integrity_mod.jsonschema_validation_error_types() == ()

    jsonschema_stub = SimpleNamespace(
        exceptions=SimpleNamespace(ValidationError=_ValidationError),
        SchemaError=_SchemaError,
    )
    monkeypatch.setattr(
        proof_pack_integrity_mod, "jsonschema", jsonschema_stub, raising=False
    )

    assert proof_pack_integrity_mod.jsonschema_validation_error_types() == (
        _ValidationError,
        _SchemaError,
    )


def test_proof_pack_integrity_validate_manifest_covers_schema_failure_and_direct_validate(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{invalid", encoding="utf-8")
    errors = proof_pack_integrity_mod.validate_manifest(manifest_path)
    assert "manifest is not valid JSON" in errors[0]

    _write_json(
        manifest_path,
        {
            "format": proof_pack_manifest_mod.PROOF_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )
    monkeypatch.setattr(
        proof_pack_integrity_mod,
        "load_proof_pack_manifest_schema",
        lambda: {"type": "object"},
        raising=True,
    )

    class _ValidationError(Exception):
        pass

    def _fail_validate(*_args, **_kwargs):
        raise _ValidationError("schema boom")

    monkeypatch.setattr(
        proof_pack_integrity_mod,
        "jsonschema",
        SimpleNamespace(
            exceptions=SimpleNamespace(ValidationError=_ValidationError),
            validate=_fail_validate,
        ),
        raising=False,
    )
    errors = proof_pack_integrity_mod.validate_manifest(manifest_path)
    assert errors == ["manifest schema validation failed: schema boom"]

    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        proof_pack_integrity_mod,
        "jsonschema",
        SimpleNamespace(
            ValidationError=_ValidationError,
            validate=lambda instance, schema: calls.append((instance, schema)),
        ),
        raising=False,
    )
    assert proof_pack_integrity_mod.validate_manifest(manifest_path) == []
    assert calls == [
        (
            json.loads(manifest_path.read_text(encoding="utf-8")),
            {"type": "object"},
        )
    ]

    direct_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        proof_pack_integrity_mod,
        "jsonschema",
        SimpleNamespace(
            validate=lambda instance, schema: direct_calls.append((instance, schema)),
        ),
        raising=False,
    )
    assert proof_pack_integrity_mod.validate_manifest(manifest_path) == []
    assert direct_calls == [
        (
            json.loads(manifest_path.read_text(encoding="utf-8")),
            {"type": "object"},
        )
    ]


def test_proof_pack_validate_manifest_falls_back_to_manual_checks_without_schema(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {
            "format": proof_pack_manifest_mod.PROOF_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
            "evidence_level": "invalid",
        },
    )

    monkeypatch.setattr(
        proof_pack_manifest_mod,
        "load_proof_pack_manifest_schema",
        lambda: None,
        raising=True,
    )
    monkeypatch.setattr(
        proof_pack_integrity_mod,
        "load_proof_pack_manifest_schema",
        lambda: None,
        raising=True,
    )

    manifest_errors = proof_pack_manifest_mod.validate_manifest(manifest_path)
    integrity_errors = proof_pack_integrity_mod.validate_manifest(manifest_path)

    assert (
        "manifest evidence_level must be 'low', 'medium', or 'high'" in manifest_errors
    )
    assert integrity_errors == manifest_errors
