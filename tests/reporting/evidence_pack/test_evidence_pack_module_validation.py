from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import invarlock.evidence_pack_integrity as evidence_pack_integrity_mod


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_evidence_pack_hashing_streams_files_without_read_bytes(
    monkeypatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "artifact.bin"
    payload = b"invarlock-streaming-hash" * 500_000
    artifact.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    def reject_full_file_copy(_path: Path) -> bytes:
        raise AssertionError("hashing must not materialize a full-file bytes copy")

    monkeypatch.setattr(Path, "read_bytes", reject_full_file_copy)

    assert evidence_pack_integrity_mod._sha256_path_hex(artifact) == expected
    assert evidence_pack_integrity_mod._sha256_file(artifact) == f"sha256:{expected}"

    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    packed_artifact = pack_dir / "artifact.bin"
    packed_artifact.write_bytes(payload)
    (pack_dir / "checksums.sha256").write_text(
        f"{expected}  artifact.bin\n", encoding="utf-8"
    )
    checksum_errors, covered_paths = evidence_pack_integrity_mod.verify_checksums(
        pack_dir
    )
    assert checksum_errors == []
    assert covered_paths == {"artifact.bin"}


def test_evidence_pack_integrity_jsonschema_helper_uses_exceptions_and_fallback_attrs(
    monkeypatch,
) -> None:
    class _ValidationError(Exception):
        pass

    class _SchemaError(Exception):
        pass

    monkeypatch.setattr(evidence_pack_integrity_mod, "jsonschema", None, raising=False)
    assert evidence_pack_integrity_mod.jsonschema_validation_error_types() == ()

    jsonschema_stub = SimpleNamespace(
        exceptions=SimpleNamespace(ValidationError=_ValidationError),
        SchemaError=_SchemaError,
    )
    monkeypatch.setattr(
        evidence_pack_integrity_mod, "jsonschema", jsonschema_stub, raising=False
    )

    assert evidence_pack_integrity_mod.jsonschema_validation_error_types() == (
        _ValidationError,
        _SchemaError,
    )


def test_evidence_pack_integrity_validate_manifest_covers_schema_failure_and_direct_validate(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{invalid", encoding="utf-8")
    errors = evidence_pack_integrity_mod.validate_manifest(manifest_path)
    assert "manifest is not valid JSON" in errors[0]

    _write_json(
        manifest_path,
        {
            "format": evidence_pack_integrity_mod.EVIDENCE_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )
    monkeypatch.setattr(
        evidence_pack_integrity_mod,
        "load_evidence_pack_manifest_schema",
        lambda: {"type": "object"},
        raising=True,
    )

    class _ValidationError(Exception):
        pass

    def _fail_validate(*_args, **_kwargs):
        raise _ValidationError("schema boom")

    monkeypatch.setattr(
        evidence_pack_integrity_mod,
        "jsonschema",
        SimpleNamespace(
            exceptions=SimpleNamespace(ValidationError=_ValidationError),
            validate=_fail_validate,
        ),
        raising=False,
    )
    errors = evidence_pack_integrity_mod.validate_manifest(manifest_path)
    assert errors == ["manifest schema validation failed: schema boom"]

    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        evidence_pack_integrity_mod,
        "jsonschema",
        SimpleNamespace(
            ValidationError=_ValidationError,
            validate=lambda instance, schema: calls.append((instance, schema)),
        ),
        raising=False,
    )
    assert evidence_pack_integrity_mod.validate_manifest(manifest_path) == []
    assert calls == [
        (
            json.loads(manifest_path.read_text(encoding="utf-8")),
            {"type": "object"},
        )
    ]

    direct_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        evidence_pack_integrity_mod,
        "jsonschema",
        SimpleNamespace(
            validate=lambda instance, schema: direct_calls.append((instance, schema)),
        ),
        raising=False,
    )
    assert evidence_pack_integrity_mod.validate_manifest(manifest_path) == []
    assert direct_calls == [
        (
            json.loads(manifest_path.read_text(encoding="utf-8")),
            {"type": "object"},
        )
    ]


def test_evidence_pack_validate_manifest_falls_back_to_manual_checks_without_schema(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {
            "format": evidence_pack_integrity_mod.EVIDENCE_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
            "evidence_level": "invalid",
        },
    )

    monkeypatch.setattr(
        evidence_pack_integrity_mod,
        "load_evidence_pack_manifest_schema",
        lambda: None,
        raising=True,
    )

    integrity_errors = evidence_pack_integrity_mod.validate_manifest(manifest_path)

    assert (
        "manifest evidence_level must be 'low', 'medium', or 'high'" in integrity_errors
    )
