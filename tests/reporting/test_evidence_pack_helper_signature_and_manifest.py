from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import invarlock.evidence_pack as evidence_pack_mod
import invarlock.evidence_pack_integrity as evidence_pack_integrity_mod
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome
from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME

__all__ = [
    "RUNTIME_MANIFEST_FILENAME",
    "VerifyExecutionResult",
    "VerifyOutcome",
    "_sign_pack",
    "_write_json",
    "_write_manifest_and_checksums",
    "_write_pack_scaffold",
    "evidence_pack_mod",
]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return evidence_pack_mod._sha256_bytes(data)


def _digest(path: Path) -> str:
    return evidence_pack_mod._sha256_file(path)


def _write_pack_scaffold(pack_dir: Path) -> tuple[Path, Path, Path]:
    report_path = (
        pack_dir / "reports" / "model" / "clean" / "noop" / "evaluation.report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}", encoding="utf-8")
    _write_json(report_path.parent / RUNTIME_MANIFEST_FILENAME, {"ok": True})

    final_verdict = pack_dir / "results" / "final_verdict.json"
    environment = pack_dir / "metadata" / "environment.json"
    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(environment, {"platform": "test"})
    return report_path, final_verdict, environment


def _write_manifest_and_checksums(
    pack_dir: Path,
    *,
    report_path: Path,
    final_verdict: Path,
    environment: Path,
    manifest_overrides: dict[str, object] | None = None,
    checksum_lines: list[str] | None = None,
) -> None:
    rel_report = str(report_path.relative_to(pack_dir)).replace("\\", "/")
    rel_runtime = str(
        (report_path.parent / RUNTIME_MANIFEST_FILENAME).relative_to(pack_dir)
    ).replace("\\", "/")
    rel_verdict = str(final_verdict.relative_to(pack_dir)).replace("\\", "/")
    rel_environment = str(environment.relative_to(pack_dir)).replace("\\", "/")
    if checksum_lines is None:
        checksum_lines = [
            f"{_sha256_bytes(final_verdict.read_bytes())}  {rel_verdict}",
            f"{_sha256_bytes(environment.read_bytes())}  {rel_environment}",
            f"{_sha256_bytes(report_path.read_bytes())}  {rel_report}",
            f"{_sha256_bytes((report_path.parent / RUNTIME_MANIFEST_FILENAME).read_bytes())}  {rel_runtime}",
        ]
    checksums_path = pack_dir / "checksums.sha256"
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    manifest = {
        "format": evidence_pack_mod.EVIDENCE_PACK_FORMAT,
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": _sha256_bytes(checksums_path.read_bytes()),
        "subject": {
            "name": "final_verdict",
            "path": rel_verdict,
            "digest": _digest(final_verdict),
        },
        "environment": {
            "path": rel_environment,
            "digest": _digest(environment),
        },
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    _write_json(pack_dir / "manifest.json", manifest)


def _sign_pack(
    pack_dir: Path,
    tmp_path: Path,
    *,
    record_manifest_fingerprint: bool = True,
    manifest_fingerprint_override: str | None = None,
) -> str:
    key_root = (
        tmp_path
        / f"evidence-pack-signing-key-{len(list(tmp_path.glob('evidence-pack-signing-key-*.pem'))):02d}.pem"
    )
    private_key = key_root
    public_key = key_root.with_name(f"{key_root.stem}.pub.pem")
    fingerprint = evidence_pack_mod._generate_signing_keypair(
        private_key,
        public_key_path=public_key,
    )
    if record_manifest_fingerprint or manifest_fingerprint_override is not None:
        manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["signing_key_fingerprint"] = (
            fingerprint
            if manifest_fingerprint_override is None
            else manifest_fingerprint_override
        )
        _write_json(pack_dir / "manifest.json", manifest)
    evidence_pack_mod._sign_manifest(
        pack_dir / "manifest.json", signing_key_path=private_key
    )
    return fingerprint


def test_signature_warnings_to_errors_converts_signature_paths() -> None:
    assert evidence_pack_mod._signature_warnings_to_errors(
        [
            "manifest.signature.json missing; pack is unsigned.",
            "other warning",
        ]
    ) == [
        "manifest.signature.json missing; signed manifest required by default.",
        "other warning",
    ]


def test_signing_key_validation_and_generation_error_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing-key.pem"
    assert evidence_pack_integrity_mod.validate_signing_key(missing) == [
        f"signing key file not found: {missing}"
    ]

    invalid_key = tmp_path / "invalid-key.pem"
    invalid_key.write_text("not-a-pem", encoding="utf-8")
    assert (
        "signing key is invalid:"
        in evidence_pack_integrity_mod.validate_signing_key(invalid_key)[0]
    )

    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rsa_key_path = tmp_path / "rsa-key.pem"
    rsa_key_path.write_bytes(
        rsa_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    assert (
        "Ed25519" in evidence_pack_integrity_mod.validate_signing_key(rsa_key_path)[0]
    )

    private_key = tmp_path / "existing-private.pem"
    public_key = tmp_path / "existing-public.pem"
    private_key.write_text("exists", encoding="utf-8")
    with pytest.raises(FileExistsError):
        evidence_pack_integrity_mod.generate_signing_keypair(
            private_key,
            public_key_path=public_key,
        )

    private_key.unlink()
    public_key.write_text("exists", encoding="utf-8")
    with pytest.raises(FileExistsError):
        evidence_pack_integrity_mod.generate_signing_keypair(
            private_key,
            public_key_path=public_key,
        )


def test_manual_validate_manifest_reports_structural_errors() -> None:
    assert evidence_pack_mod._manual_validate_manifest("bad") == [
        "manifest must decode to a JSON object"
    ]

    errors = evidence_pack_mod._manual_validate_manifest(
        {
            "format": "wrong",
            "checksums_sha256": "other.txt",
            "checksums_sha256_digest": "short",
            "network_mode": "wifi",
            "artifacts": {},
            "builder": {"id": "", "name": ""},
            "subject": {"path": "", "digest": "bad"},
            "invocation": {"config_source": "bad", "parameters": "bad"},
            "environment": {"path": "", "digest": "bad"},
            "materials": [{"name": "", "path": "", "digest": "bad"}],
        }
    )

    assert any("manifest format must be" in error for error in errors)
    assert any("checksums_sha256 must point" in error for error in errors)
    assert any("64-char sha256 hex" in error for error in errors)
    assert any("network_mode must be" in error for error in errors)
    assert any("artifacts must be a list" in error for error in errors)
    assert any("builder.id must be a non-empty string" in error for error in errors)
    assert any("builder.name must be a non-empty string" in error for error in errors)
    assert any("subject.path must be a non-empty string" in error for error in errors)
    assert any(
        "subject.digest must be a sha256:... string" in error for error in errors
    )
    assert any(
        "invocation.config_source must be an object" in error for error in errors
    )
    assert any("invocation.parameters must be an object" in error for error in errors)
    assert any(
        "environment.path must be a non-empty string" in error for error in errors
    )
    assert any(
        "materials[0].name must be a non-empty string" in error for error in errors
    )

    errors = evidence_pack_mod._manual_validate_manifest(
        {
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
            "builder": "bad",
            "invocation": "bad",
            "materials": "bad",
        }
    )
    assert any("manifest missing required field: format" in error for error in errors)
    assert any("manifest builder must be an object" in error for error in errors)
    assert any("manifest invocation must be an object" in error for error in errors)
    assert any("manifest materials must be a list" in error for error in errors)


def test_manual_validate_manifest_covers_empty_digest_refs_and_non_dict_materials() -> (
    None
):
    errors = evidence_pack_mod._manual_validate_manifest(
        {
            "format": evidence_pack_mod.EVIDENCE_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
            "subject": {},
            "environment": {},
            "materials": ["not-a-dict"],
        }
    )

    assert errors == ["manifest materials[0] must be an object"]


def test_validate_manifest_and_load_json_object_cover_error_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{invalid", encoding="utf-8")
    errors = evidence_pack_mod.validate_manifest(manifest_path)
    assert "manifest is not valid JSON" in errors[0]

    manifest_path.write_bytes(b"\xff")
    errors = evidence_pack_mod.validate_manifest(manifest_path)
    assert "manifest is not valid JSON" in errors[0]

    _write_json(
        manifest_path,
        {
            "format": evidence_pack_mod.EVIDENCE_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )
    monkeypatch.setattr(
        evidence_pack_mod,
        "load_evidence_pack_manifest_schema",
        lambda: {"type": "object"},
        raising=True,
    )
    monkeypatch.setattr(
        evidence_pack_mod.jsonschema,
        "validate",
        lambda instance, schema: (_ for _ in ()).throw(
            evidence_pack_mod.jsonschema.exceptions.ValidationError("schema boom")
        ),
        raising=True,
    )
    errors = evidence_pack_mod.validate_manifest(manifest_path)
    assert errors == ["manifest schema validation failed: schema boom"]

    payload, errors = evidence_pack_mod._load_json_object(
        tmp_path / "missing.json", label="demo"
    )
    assert payload is None
    assert errors == [f"demo file not found: {tmp_path / 'missing.json'}"]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{invalid", encoding="utf-8")
    payload, errors = evidence_pack_mod._load_json_object(bad_json, label="demo")
    assert payload is None
    assert "demo is not valid JSON" in errors[0]

    bad_utf8 = tmp_path / "bad-utf8.json"
    bad_utf8.write_bytes(b"\xff")
    payload, errors = evidence_pack_mod._load_json_object(bad_utf8, label="demo")
    assert payload is None
    assert "demo is not valid JSON" in errors[0]

    array_json = tmp_path / "array.json"
    array_json.write_text("[1, 2]", encoding="utf-8")
    payload, errors = evidence_pack_mod._load_json_object(array_json, label="demo")
    assert payload is None
    assert errors == [f"demo must decode to a JSON object: {array_json}"]


def test_jsonschema_helper_and_direct_validate_fallback_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _ValidationError(Exception):
        pass

    monkeypatch.setattr(evidence_pack_mod, "jsonschema", None, raising=False)
    assert evidence_pack_mod._jsonschema_validation_error_types() == ()

    jsonschema_stub = SimpleNamespace(
        ValidationError=_ValidationError,
        validate=lambda instance, schema: None,
    )
    monkeypatch.setattr(evidence_pack_mod, "jsonschema", jsonschema_stub, raising=False)
    assert evidence_pack_mod._jsonschema_validation_error_types() == (_ValidationError,)

    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {
            "format": evidence_pack_mod.EVIDENCE_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
            "subject": {
                "name": "final_verdict",
                "path": "results/final_verdict.json",
                "digest": "sha256:" + ("b" * 64),
            },
            "environment": {
                "path": "metadata/environment.json",
                "digest": "sha256:" + ("c" * 64),
            },
        },
    )
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        evidence_pack_mod,
        "load_evidence_pack_manifest_schema",
        lambda: {"type": "object"},
        raising=True,
    )
    monkeypatch.setattr(
        evidence_pack_mod,
        "jsonschema",
        SimpleNamespace(
            validate=lambda instance, schema: calls.append((instance, schema))
        ),
        raising=False,
    )
    assert evidence_pack_mod.validate_manifest(manifest_path) == []
    assert calls == [
        (
            json.loads(manifest_path.read_text(encoding="utf-8")),
            {"type": "object"},
        )
    ]


def test_material_and_reference_helpers_cover_invalid_paths(tmp_path: Path) -> None:
    assert evidence_pack_mod._material_spec("missing-separator") is None
    assert evidence_pack_mod._material_spec(" =demo.json") is None
    assert evidence_pack_mod._material_spec("demo= ") is None
    assert evidence_pack_mod._material_spec("demo=payload.json") == (
        "demo",
        Path("payload.json"),
    )

    assert evidence_pack_mod._validate_material_name("good.name-1") is None
    assert "material names must match" in str(
        evidence_pack_mod._validate_material_name("../bad")
    )

    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    target = pack_dir / "file.json"
    target.write_text("{}", encoding="utf-8")

    assert (
        evidence_pack_mod._validate_reference(
            pack_dir=pack_dir, label="demo", payload="bad"
        )
        == []
    )
    assert evidence_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={"digest": _digest(target)},
    ) == ["demo must include a non-empty path when digest verification is enabled"]
    assert evidence_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={"path": "file.json", "digest": "bad"},
    ) == ["demo digest must be a sha256:... string"]
    assert evidence_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={"path": "../escape.json", "digest": _digest(target)},
    ) == ["demo path escapes the pack root: ../escape.json"]
    assert evidence_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={
            "path": "missing.json",
            "digest": "sha256:" + ("a" * 64),
        },
    ) == ["demo path is missing: missing.json"]
    mismatch = evidence_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={"path": "file.json", "digest": "sha256:" + ("b" * 64)},
    )
    assert "demo digest mismatch for file.json" in mismatch[0]
    assert (
        evidence_pack_mod._validate_reference(
            pack_dir=pack_dir,
            label="demo",
            payload={"path": "file.json", "digest": _digest(target)},
        )
        == []
    )


def test_checksum_and_extra_file_helpers_cover_error_paths(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )

    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("checksums_sha256_digest")
    _write_json(pack_dir / "manifest.json", manifest)
    assert evidence_pack_mod._verify_manifest_binds_checksums(pack_dir) == [
        "manifest.json missing checksums_sha256_digest (pack is not tamper-evident)."
    ]

    manifest["checksums_sha256_digest"] = ""
    _write_json(pack_dir / "manifest.json", manifest)
    assert evidence_pack_mod._verify_manifest_binds_checksums(pack_dir) == [
        "manifest.json checksums_sha256_digest is empty."
    ]

    manifest["checksums_sha256_digest"] = "a" * 64
    _write_json(pack_dir / "manifest.json", manifest)
    assert (
        "checksums.sha256 digest mismatch"
        in evidence_pack_mod._verify_manifest_binds_checksums(pack_dir)[0]
    )

    (pack_dir / "checksums.sha256").write_text(
        "\n".join(
            [
                "not a checksum line",
                f"{'a' * 64}  ../escape.txt",
                f"{'b' * 64}  missing.txt",
                f"{'c' * 64}  {final_verdict.relative_to(pack_dir).as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    entries, errors = evidence_pack_mod._parse_checksums(pack_dir)
    assert entries[0][1] == "../escape.txt"
    assert "line 1 is not a valid sha256 entry" in errors[0]
    checksum_errors, covered = evidence_pack_mod._verify_checksums(pack_dir)
    assert "../escape.txt" in covered
    assert any("escapes the pack root" in error for error in checksum_errors)
    assert any("missing from pack: missing.txt" in error for error in checksum_errors)
    assert any(
        "checksum mismatch for results/final_verdict.json" in error
        for error in checksum_errors
    )

    (pack_dir / "extra.txt").write_text("extra", encoding="utf-8")
    extra_errors, extra_warnings = evidence_pack_mod._verify_no_extra_files(
        pack_dir, covered_paths=covered, strict=False
    )
    assert extra_errors == []
    assert "extra files not covered" in extra_warnings[0]
    extra_errors, extra_warnings = evidence_pack_mod._verify_no_extra_files(
        pack_dir, covered_paths=covered, strict=True
    )
    assert "extra files not covered" in extra_errors[0]
    assert extra_warnings == []


def test_verify_signature_covers_missing_signature_failure_and_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )

    errors, warnings, fingerprint = evidence_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert errors == []
    assert warnings == ["manifest.signature.json missing; pack is unsigned."]
    assert fingerprint is None

    errors, warnings, fingerprint = evidence_pack_mod._verify_signature(
        pack_dir, strict=True
    )
    assert errors == [
        "manifest.signature.json missing (strict mode requires a signed manifest)."
    ]
    assert warnings == []
    assert fingerprint is None

    (pack_dir / "manifest.signature.json").write_text("{invalid", encoding="utf-8")
    errors, warnings, fingerprint = evidence_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert "manifest.signature.json is not valid JSON" in errors[0]
    assert warnings == []
    assert fingerprint is None

    fingerprint = _sign_pack(pack_dir, tmp_path)
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["format"] = "tampered"
    _write_json(pack_dir / "manifest.json", manifest)
    errors, warnings, fingerprint_out = evidence_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert "manifest signature verification failed." in errors[0]
    assert warnings == []
    assert fingerprint_out is None

    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )
    mismatch_fingerprint = _sign_pack(
        pack_dir,
        tmp_path,
        manifest_fingerprint_override="EXPECTED",
    )
    errors, warnings, fingerprint_out = evidence_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert "signing_key_fingerprint (EXPECTED) does not match" in errors[0]
    assert warnings == []
    assert fingerprint_out == mismatch_fingerprint


def test_verify_signature_uses_default_failure_text_and_rejects_malformed_bundle(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )
    _write_json(
        pack_dir / "manifest.signature.json",
        {
            "format": "evidence-pack-signature-v1",
            "algorithm": "ed25519",
            "signing_key_fingerprint": "sha256:" + ("a" * 64),
            "public_key": {"encoding": "pem", "value": "bad-key"},
            "signature": {"encoding": "base64", "value": ""},
        },
    )
    errors, warnings, fingerprint = evidence_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert (
        "manifest.signature.json signature.value must be a non-empty base64 string."
        in errors[0]
    )
    assert warnings == []
    assert fingerprint is None

    _write_json(
        pack_dir / "manifest.signature.json",
        {
            "format": "wrong",
            "algorithm": "ed25519",
            "signing_key_fingerprint": "sha256:" + ("a" * 64),
            "public_key": {"encoding": "pem", "value": "bad-key"},
            "signature": {"encoding": "base64", "value": "abc"},
        },
    )
    errors, warnings, fingerprint = evidence_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert "manifest.signature.json format must be" in errors[0]
    assert warnings == []
    assert fingerprint is None


def test_signature_bundle_structure_and_decode_error_paths(tmp_path: Path) -> None:
    signature_path = tmp_path / "manifest.signature.json"

    _write_json(signature_path, ["not", "an", "object"])
    bundle, errors = evidence_pack_integrity_mod._load_signature_bundle(signature_path)
    assert bundle is None
    assert errors == ["manifest.signature.json must decode to a JSON object."]

    _write_json(
        signature_path,
        {
            "format": "wrong",
            "algorithm": "rsa",
            "public_key": "bad",
            "signature": "bad",
            "signing_key_fingerprint": "bad",
        },
    )
    bundle, errors = evidence_pack_integrity_mod._load_signature_bundle(signature_path)
    assert bundle is None
    assert "manifest.signature.json algorithm must be 'ed25519'." in errors
    assert "manifest.signature.json public_key must be an object." in errors
    assert "manifest.signature.json signature must be an object." in errors
    assert (
        "manifest.signature.json signing_key_fingerprint must be a sha256:... string."
        in errors
    )

    _write_json(
        signature_path,
        {
            "format": "evidence-pack-signature-v1",
            "algorithm": "ed25519",
            "public_key": {"encoding": "der", "value": ""},
            "signature": {"encoding": "hex", "value": ""},
            "signing_key_fingerprint": "sha256:" + ("a" * 64),
        },
    )
    bundle, errors = evidence_pack_integrity_mod._load_signature_bundle(signature_path)
    assert bundle is None
    assert "manifest.signature.json public_key.encoding must be 'pem'." in errors
    assert (
        "manifest.signature.json public_key.value must be a non-empty PEM string."
        in errors
    )
    assert "manifest.signature.json signature.encoding must be 'base64'." in errors
    assert (
        "manifest.signature.json signature.value must be a non-empty base64 string."
        in errors
    )


def test_verify_signature_bundle_fingerprint_and_decode_error_paths(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )
    expected_fingerprint = _sign_pack(pack_dir, tmp_path)
    signature_path = pack_dir / "manifest.signature.json"

    bundle = json.loads(signature_path.read_text(encoding="utf-8"))
    bundle["signing_key_fingerprint"] = "sha256:" + ("0" * 64)
    _write_json(signature_path, bundle)
    errors, warnings, fingerprint = evidence_pack_integrity_mod.verify_signature(
        pack_dir, strict=False
    )
    assert "does not match bundled public key" in errors[0]
    assert warnings == []
    assert fingerprint == expected_fingerprint

    bundle = json.loads(signature_path.read_text(encoding="utf-8"))
    bundle["signing_key_fingerprint"] = expected_fingerprint
    bundle["signature"]["value"] = "%%%not-base64%%%"
    _write_json(signature_path, bundle)
    errors, warnings, fingerprint = evidence_pack_integrity_mod.verify_signature(
        pack_dir, strict=False
    )
    assert "manifest signature verification failed." in errors[0]
    assert warnings == []
    assert fingerprint is None
