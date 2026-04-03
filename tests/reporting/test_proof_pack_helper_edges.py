from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import invarlock.proof_pack as proof_pack_mod
import invarlock.proof_pack_integrity as proof_pack_integrity_mod
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome
from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return proof_pack_mod._sha256_bytes(data)


def _digest(path: Path) -> str:
    return proof_pack_mod._sha256_file(path)


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
        "format": proof_pack_mod.PROOF_PACK_FORMAT,
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
        / f"proof-pack-signing-key-{len(list(tmp_path.glob('proof-pack-signing-key-*.pem'))):02d}.pem"
    )
    private_key = key_root
    public_key = key_root.with_name(f"{key_root.stem}.pub.pem")
    fingerprint = proof_pack_mod._generate_signing_keypair(
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
    proof_pack_mod._sign_manifest(
        pack_dir / "manifest.json", signing_key_path=private_key
    )
    return fingerprint


def test_signature_warnings_to_errors_converts_signature_paths() -> None:
    assert proof_pack_mod._signature_warnings_to_errors(
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
    assert proof_pack_integrity_mod.validate_signing_key(missing) == [
        f"signing key file not found: {missing}"
    ]

    invalid_key = tmp_path / "invalid-key.pem"
    invalid_key.write_text("not-a-pem", encoding="utf-8")
    assert (
        "signing key is invalid:"
        in proof_pack_integrity_mod.validate_signing_key(invalid_key)[0]
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
    assert "Ed25519" in proof_pack_integrity_mod.validate_signing_key(rsa_key_path)[0]

    private_key = tmp_path / "existing-private.pem"
    public_key = tmp_path / "existing-public.pem"
    private_key.write_text("exists", encoding="utf-8")
    with pytest.raises(FileExistsError):
        proof_pack_integrity_mod.generate_signing_keypair(
            private_key,
            public_key_path=public_key,
        )

    private_key.unlink()
    public_key.write_text("exists", encoding="utf-8")
    with pytest.raises(FileExistsError):
        proof_pack_integrity_mod.generate_signing_keypair(
            private_key,
            public_key_path=public_key,
        )


def test_manual_validate_manifest_reports_structural_errors() -> None:
    assert proof_pack_mod._manual_validate_manifest("bad") == [
        "manifest must decode to a JSON object"
    ]

    errors = proof_pack_mod._manual_validate_manifest(
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

    errors = proof_pack_mod._manual_validate_manifest(
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
    errors = proof_pack_mod._manual_validate_manifest(
        {
            "format": proof_pack_mod.PROOF_PACK_FORMAT,
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
    errors = proof_pack_mod.validate_manifest(manifest_path)
    assert "manifest is not valid JSON" in errors[0]

    manifest_path.write_bytes(b"\xff")
    errors = proof_pack_mod.validate_manifest(manifest_path)
    assert "manifest is not valid JSON" in errors[0]

    _write_json(
        manifest_path,
        {
            "format": proof_pack_mod.PROOF_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )
    monkeypatch.setattr(
        proof_pack_mod,
        "load_proof_pack_manifest_schema",
        lambda: {"type": "object"},
        raising=True,
    )
    monkeypatch.setattr(
        proof_pack_mod.jsonschema,
        "validate",
        lambda instance, schema: (_ for _ in ()).throw(
            proof_pack_mod.jsonschema.exceptions.ValidationError("schema boom")
        ),
        raising=True,
    )
    errors = proof_pack_mod.validate_manifest(manifest_path)
    assert errors == ["manifest schema validation failed: schema boom"]

    payload, errors = proof_pack_mod._load_json_object(
        tmp_path / "missing.json", label="demo"
    )
    assert payload is None
    assert errors == [f"demo file not found: {tmp_path / 'missing.json'}"]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{invalid", encoding="utf-8")
    payload, errors = proof_pack_mod._load_json_object(bad_json, label="demo")
    assert payload is None
    assert "demo is not valid JSON" in errors[0]

    bad_utf8 = tmp_path / "bad-utf8.json"
    bad_utf8.write_bytes(b"\xff")
    payload, errors = proof_pack_mod._load_json_object(bad_utf8, label="demo")
    assert payload is None
    assert "demo is not valid JSON" in errors[0]

    array_json = tmp_path / "array.json"
    array_json.write_text("[1, 2]", encoding="utf-8")
    payload, errors = proof_pack_mod._load_json_object(array_json, label="demo")
    assert payload is None
    assert errors == [f"demo must decode to a JSON object: {array_json}"]


def test_jsonschema_helper_and_direct_validate_fallback_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _ValidationError(Exception):
        pass

    monkeypatch.setattr(proof_pack_mod, "jsonschema", None, raising=False)
    assert proof_pack_mod._jsonschema_validation_error_types() == ()

    jsonschema_stub = SimpleNamespace(
        ValidationError=_ValidationError,
        validate=lambda instance, schema: None,
    )
    monkeypatch.setattr(proof_pack_mod, "jsonschema", jsonschema_stub, raising=False)
    assert proof_pack_mod._jsonschema_validation_error_types() == (_ValidationError,)

    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {
            "format": proof_pack_mod.PROOF_PACK_FORMAT,
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
        proof_pack_mod,
        "load_proof_pack_manifest_schema",
        lambda: {"type": "object"},
        raising=True,
    )
    monkeypatch.setattr(
        proof_pack_mod,
        "jsonschema",
        SimpleNamespace(
            validate=lambda instance, schema: calls.append((instance, schema))
        ),
        raising=False,
    )
    assert proof_pack_mod.validate_manifest(manifest_path) == []
    assert calls == [
        (
            json.loads(manifest_path.read_text(encoding="utf-8")),
            {"type": "object"},
        )
    ]


def test_material_and_reference_helpers_cover_invalid_paths(tmp_path: Path) -> None:
    assert proof_pack_mod._material_spec("missing-separator") is None
    assert proof_pack_mod._material_spec(" =demo.json") is None
    assert proof_pack_mod._material_spec("demo= ") is None
    assert proof_pack_mod._material_spec("demo=payload.json") == (
        "demo",
        Path("payload.json"),
    )

    assert proof_pack_mod._validate_material_name("good.name-1") is None
    assert "material names must match" in str(
        proof_pack_mod._validate_material_name("../bad")
    )

    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    target = pack_dir / "file.json"
    target.write_text("{}", encoding="utf-8")

    assert (
        proof_pack_mod._validate_reference(
            pack_dir=pack_dir, label="demo", payload="bad"
        )
        == []
    )
    assert proof_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={"digest": _digest(target)},
    ) == ["demo must include a non-empty path when digest verification is enabled"]
    assert proof_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={"path": "file.json", "digest": "bad"},
    ) == ["demo digest must be a sha256:... string"]
    assert proof_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={"path": "../escape.json", "digest": _digest(target)},
    ) == ["demo path escapes the pack root: ../escape.json"]
    assert proof_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={
            "path": "missing.json",
            "digest": "sha256:" + ("a" * 64),
        },
    ) == ["demo path is missing: missing.json"]
    mismatch = proof_pack_mod._validate_reference(
        pack_dir=pack_dir,
        label="demo",
        payload={"path": "file.json", "digest": "sha256:" + ("b" * 64)},
    )
    assert "demo digest mismatch for file.json" in mismatch[0]
    assert (
        proof_pack_mod._validate_reference(
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
    assert proof_pack_mod._verify_manifest_binds_checksums(pack_dir) == [
        "manifest.json missing checksums_sha256_digest (pack is not tamper-evident)."
    ]

    manifest["checksums_sha256_digest"] = ""
    _write_json(pack_dir / "manifest.json", manifest)
    assert proof_pack_mod._verify_manifest_binds_checksums(pack_dir) == [
        "manifest.json checksums_sha256_digest is empty."
    ]

    manifest["checksums_sha256_digest"] = "a" * 64
    _write_json(pack_dir / "manifest.json", manifest)
    assert (
        "checksums.sha256 digest mismatch"
        in proof_pack_mod._verify_manifest_binds_checksums(pack_dir)[0]
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
    entries, errors = proof_pack_mod._parse_checksums(pack_dir)
    assert entries[0][1] == "../escape.txt"
    assert "line 1 is not a valid sha256 entry" in errors[0]
    checksum_errors, covered = proof_pack_mod._verify_checksums(pack_dir)
    assert "../escape.txt" in covered
    assert any("escapes the pack root" in error for error in checksum_errors)
    assert any("missing from pack: missing.txt" in error for error in checksum_errors)
    assert any(
        "checksum mismatch for results/final_verdict.json" in error
        for error in checksum_errors
    )

    (pack_dir / "extra.txt").write_text("extra", encoding="utf-8")
    extra_errors, extra_warnings = proof_pack_mod._verify_no_extra_files(
        pack_dir, covered_paths=covered, strict=False
    )
    assert extra_errors == []
    assert "extra files not covered" in extra_warnings[0]
    extra_errors, extra_warnings = proof_pack_mod._verify_no_extra_files(
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

    errors, warnings, fingerprint = proof_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert errors == []
    assert warnings == ["manifest.signature.json missing; pack is unsigned."]
    assert fingerprint is None

    errors, warnings, fingerprint = proof_pack_mod._verify_signature(
        pack_dir, strict=True
    )
    assert errors == [
        "manifest.signature.json missing (strict mode requires a signed manifest)."
    ]
    assert warnings == []
    assert fingerprint is None

    (pack_dir / "manifest.signature.json").write_text("{invalid", encoding="utf-8")
    errors, warnings, fingerprint = proof_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert "manifest.signature.json is not valid JSON" in errors[0]
    assert warnings == []
    assert fingerprint is None

    fingerprint = _sign_pack(pack_dir, tmp_path)
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["format"] = "tampered"
    _write_json(pack_dir / "manifest.json", manifest)
    errors, warnings, fingerprint_out = proof_pack_mod._verify_signature(
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
    errors, warnings, fingerprint_out = proof_pack_mod._verify_signature(
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
            "format": "proof-pack-signature-v1",
            "algorithm": "ed25519",
            "signing_key_fingerprint": "sha256:" + ("a" * 64),
            "public_key": {"encoding": "pem", "value": "bad-key"},
            "signature": {"encoding": "base64", "value": ""},
        },
    )
    errors, warnings, fingerprint = proof_pack_mod._verify_signature(
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
    errors, warnings, fingerprint = proof_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert "manifest.signature.json format must be" in errors[0]
    assert warnings == []
    assert fingerprint is None


def test_signature_bundle_structure_and_decode_error_paths(tmp_path: Path) -> None:
    signature_path = tmp_path / "manifest.signature.json"

    _write_json(signature_path, ["not", "an", "object"])
    bundle, errors = proof_pack_integrity_mod._load_signature_bundle(signature_path)
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
    bundle, errors = proof_pack_integrity_mod._load_signature_bundle(signature_path)
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
            "format": "proof-pack-signature-v1",
            "algorithm": "ed25519",
            "public_key": {"encoding": "der", "value": ""},
            "signature": {"encoding": "hex", "value": ""},
            "signing_key_fingerprint": "sha256:" + ("a" * 64),
        },
    )
    bundle, errors = proof_pack_integrity_mod._load_signature_bundle(signature_path)
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
    errors, warnings, fingerprint = proof_pack_integrity_mod.verify_signature(
        pack_dir, strict=False
    )
    assert "does not match bundled public key" in errors[0]
    assert warnings == []
    assert fingerprint == expected_fingerprint

    bundle = json.loads(signature_path.read_text(encoding="utf-8"))
    bundle["signing_key_fingerprint"] = expected_fingerprint
    bundle["signature"]["value"] = "%%%not-base64%%%"
    _write_json(signature_path, bundle)
    errors, warnings, fingerprint = proof_pack_integrity_mod.verify_signature(
        pack_dir, strict=False
    )
    assert "manifest signature verification failed." in errors[0]
    assert warnings == []
    assert fingerprint is None


def test_verify_signature_rejects_non_ed25519_public_key(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )

    rsa_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        rsa_private.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )
    _write_json(
        pack_dir / "manifest.signature.json",
        {
            "format": "proof-pack-signature-v1",
            "algorithm": "ed25519",
            "signing_key_fingerprint": "sha256:" + ("a" * 64),
            "public_key": {"encoding": "pem", "value": public_pem},
            "signature": {"encoding": "base64", "value": "YWJj"},
        },
    )

    errors, warnings, fingerprint = proof_pack_integrity_mod.verify_signature(
        pack_dir, strict=False
    )
    assert errors == [
        "manifest signature verification failed. public key must be Ed25519."
    ]
    assert warnings == []
    assert fingerprint is None


def test_verify_reports_and_inspect_cover_error_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_pack = tmp_path / "empty"
    empty_pack.mkdir()
    errors, payload = proof_pack_mod._verify_reports(
        empty_pack, json_out_path=None, profile="dev"
    )
    assert errors == ["No reports found in pack."]
    assert payload is None

    error_only_pack = tmp_path / "error-only"
    error_report = (
        error_only_pack
        / "reports"
        / "model"
        / "errors"
        / "noop"
        / "evaluation.report.json"
    )
    error_report.parent.mkdir(parents=True, exist_ok=True)
    error_report.write_text("{}", encoding="utf-8")
    errors, payload = proof_pack_mod._verify_reports(
        error_only_pack, json_out_path=None, profile="dev"
    )
    assert errors == [
        "No clean reports found in pack (only error-injection reports present)."
    ]
    assert payload is None

    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )
    (pack_dir / "reports" / "model" / "errors" / "noop").mkdir(
        parents=True, exist_ok=True
    )
    (
        pack_dir / "reports" / "model" / "errors" / "noop" / "evaluation.report.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )
    json_out = tmp_path / "nested.json"
    verify_calls: list[list[str]] = []

    def _fake_run_verify(reports: list[Path], *, profile: str):
        verify_calls.append([str(path) for path in reports])
        if len(verify_calls) == 1:
            return VerifyExecutionResult(
                outcome=VerifyOutcome.OK,
                payload={"ok": False},
                diagnostics=(),
            )
        raise RuntimeError("ignore nested error reports")

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        _fake_run_verify,
        raising=True,
    )
    errors, payload = proof_pack_mod._verify_reports(
        pack_dir, json_out_path=json_out, profile="release"
    )
    assert errors == [
        "error-injection report verification failed: ignore nested error reports"
    ]
    assert payload == {"ok": False}
    assert len(verify_calls) == 2

    missing_result = proof_pack_mod.inspect_proof_pack(tmp_path / "missing")
    missing_payload = missing_result.payload
    exit_code = missing_result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.MISSING
    assert missing_payload["ok"] is False

    invalid_pack = tmp_path / "invalid"
    invalid_pack.mkdir()
    (invalid_pack / "manifest.json").write_text("{invalid", encoding="utf-8")
    (invalid_pack / "checksums.sha256").write_text("", encoding="utf-8")
    invalid_result = proof_pack_mod.inspect_proof_pack(invalid_pack)
    invalid_payload = invalid_result.payload
    exit_code = invalid_result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.FORMAT
    assert invalid_payload["ok"] is False


def test_verify_reports_covers_remaining_payload_contract_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _build_pack(name: str, *, with_errors: bool) -> Path:
        pack_dir = tmp_path / name
        report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
        _write_manifest_and_checksums(
            pack_dir,
            report_path=report_path,
            final_verdict=final_verdict,
            environment=environment,
        )
        if with_errors:
            error_dir = pack_dir / "reports" / "model" / "errors" / "noop"
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
        return pack_dir

    pack_with_errors = _build_pack("with-errors", with_errors=True)

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        lambda reports, *, profile: (
            VerifyExecutionResult(
                outcome=VerifyOutcome.OK,
                payload={"ok": True},
                diagnostics=(),
            )
            if "clean" in str(reports[0])
            else VerifyExecutionResult(
                outcome=VerifyOutcome.OK,
                payload=["bad-payload"],
                diagnostics=(),
            )
        ),
        raising=True,
    )
    errors, payload = proof_pack_mod._verify_reports(
        pack_with_errors, json_out_path=None, profile="dev"
    )
    assert errors == [
        "error-injection report verification did not return a JSON object."
    ]
    assert payload == {"ok": True}

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        lambda reports, *, profile: (
            VerifyExecutionResult(
                outcome=VerifyOutcome.OK,
                payload=None,
                diagnostics=(),
            )
            if "clean" in str(reports[0])
            else VerifyExecutionResult(
                outcome=VerifyOutcome.OK,
                payload={"ok": True},
                diagnostics=(),
            )
        ),
        raising=True,
    )
    errors, payload = proof_pack_mod._verify_reports(
        pack_with_errors, json_out_path=None, profile="dev"
    )
    assert errors == ["clean report verification did not return a JSON object."]
    assert payload is None

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        lambda reports, *, profile: (
            VerifyExecutionResult(
                outcome=VerifyOutcome.OK,
                payload=["clean-bad"],
                diagnostics=(),
            )
            if "clean" in str(reports[0])
            else VerifyExecutionResult(
                outcome=VerifyOutcome.OK,
                payload={"ok": True},
                diagnostics=(),
            )
        ),
        raising=True,
    )
    errors, payload = proof_pack_mod._verify_reports(
        pack_with_errors, json_out_path=None, profile="dev"
    )
    assert errors == ["clean report verification did not return a JSON object."]
    assert payload is None

    pack_clean_only = _build_pack("clean-only", with_errors=False)
    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        lambda reports, *, profile: VerifyExecutionResult(
            outcome=VerifyOutcome.POLICY_FAIL,
            payload={"ok": False},
            diagnostics=(),
        ),
        raising=True,
    )
    errors, payload = proof_pack_mod._verify_reports(
        pack_clean_only, json_out_path=None, profile="release"
    )
    assert errors == ["invarlock verify reported report verification failures."]
    assert payload == {"ok": False}


def test_build_and_verify_proof_pack_cover_usage_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final_verdict = tmp_path / "final.json"
    _write_json(final_verdict, {"verdict": "PASS"})
    report_path = tmp_path / "report.json"
    _write_json(report_path, {"ok": True})
    runtime_manifest = report_path.parent / RUNTIME_MANIFEST_FILENAME
    _write_json(runtime_manifest, {"ok": True})
    source_repo = tmp_path / "source_repo.json"
    _write_json(source_repo, {"commit": "abc123"})
    environment = tmp_path / "environment.json"
    _write_json(environment, {"platform": "test"})
    material = tmp_path / "material.json"
    _write_json(material, {"name": "demo"})

    result = proof_pack_mod.build_proof_pack(
        tmp_path / "out-none",
        final_verdict_path=final_verdict,
        report_paths=[],
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.USAGE
    assert "at least one --report input" in payload["errors"][0]

    existing_out = tmp_path / "existing"
    existing_out.mkdir()
    result = proof_pack_mod.build_proof_pack(
        existing_out,
        final_verdict_path=final_verdict,
        report_paths=[report_path],
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.USAGE
    assert "already exists" in payload["errors"][0]

    result = proof_pack_mod.build_proof_pack(
        tmp_path / "out-invalid-material",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        material_specs=[("../bad", material), ("../bad", material)],
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.FORMAT
    assert any("Invalid material name" in error for error in payload["errors"])
    assert any("Duplicate material name" in error for error in payload["errors"])

    runtime_manifest.unlink()
    result = proof_pack_mod.build_proof_pack(
        tmp_path / "out-missing-sidecar",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.FORMAT
    assert any("report sidecar file not found" in error for error in payload["errors"])
    _write_json(runtime_manifest, {"ok": True})

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        lambda reports, profile: VerifyExecutionResult(
            outcome=VerifyOutcome.POLICY_FAIL,
            payload={"ok": False},
            diagnostics=(),
        ),
        raising=True,
    )
    result = proof_pack_mod.build_proof_pack(
        tmp_path / "out-verify-fail",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.REPORTS
    assert payload["verify"] == {"ok": False}

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        lambda reports, profile: VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True},
            diagnostics=(),
        ),
        raising=True,
    )
    result = proof_pack_mod.build_proof_pack(
        tmp_path / "out-ok",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        source_repo_path=source_repo,
        environment_path=environment,
        material_specs=[("demo", material)],
        readme_path=tmp_path / "missing-readme.md",
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.OK
    assert payload["ok"] is True
    assert any("README file not found" in warning for warning in payload["warnings"])

    result = proof_pack_mod.verify_proof_pack(
        tmp_path / "missing-pack", skip_verify=True
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.MISSING
    assert payload["ok"] is False

    result = proof_pack_mod.verify_proof_pack(
        tmp_path / "out-ok",
        json_out_path=(tmp_path / "out-ok" / "verify.json"),
        skip_verify=True,
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.USAGE
    assert "--json-out must point outside the pack directory." in payload["errors"]


def test_run_verify_command_delegates_to_verify_reports_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = tmp_path / "evaluation.report.json"
    report.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_verify_contract(
        reports: list[Path],
        *,
        baseline=None,
        tolerance=1e-9,
        profile=None,
        allow_unattested_artifacts=False,
        json_mode=False,
    ):
        captured["reports"] = reports
        captured["profile"] = profile
        captured["json_mode"] = json_mode
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True},
            diagnostics=(),
        )

    monkeypatch.setattr(
        proof_pack_mod,
        "run_verify_reports",
        _fake_verify_contract,
        raising=False,
    )

    result = proof_pack_mod._run_verify_command([report], profile="release")

    assert result.outcome == VerifyOutcome.OK
    assert result.payload == {"ok": True}
    assert captured["reports"] == [report]
    assert captured["profile"] == "release"
    assert captured["json_mode"] is True


def test_manual_validate_manifest_accepts_valid_optional_sections() -> None:
    payload = {
        "format": proof_pack_mod.PROOF_PACK_FORMAT,
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": "a" * 64,
        "network_mode": "offline",
        "artifacts": [],
        "builder": {"id": "builder-1", "name": "Builder"},
        "subject": {
            "path": "results/final_verdict.json",
            "digest": "sha256:" + ("b" * 64),
        },
        "invocation": {
            "config_source": {
                "path": "metadata/source_repo.json",
                "digest": "sha256:" + ("c" * 64),
            },
            "parameters": {"profile": "ci"},
        },
        "environment": {
            "path": "metadata/environment.json",
            "digest": "sha256:" + ("d" * 64),
        },
        "materials": [
            {
                "name": "evidence",
                "path": "metadata/evidence.json",
                "digest": "sha256:" + ("e" * 64),
            }
        ],
    }

    assert proof_pack_mod._manual_validate_manifest(payload) == []


def test_validate_manifest_uses_manual_validation_when_schema_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {
            "format": "wrong",
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )
    monkeypatch.setattr(
        proof_pack_mod,
        "load_proof_pack_manifest_schema",
        lambda: None,
        raising=True,
    )

    errors = proof_pack_mod.validate_manifest(manifest_path)
    assert any("manifest format must be" in error for error in errors)


def test_validate_reference_allows_empty_path_and_digest_pair(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()

    assert (
        proof_pack_mod._validate_reference(
            pack_dir=pack_dir,
            label="demo",
            payload={"path": None, "digest": None},
        )
        == []
    )


def test_verify_manifest_attestation_rejects_non_object_manifest(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")

    assert proof_pack_mod.verify_manifest_attestation(pack_dir) == [
        "manifest must decode to a JSON object"
    ]


def test_verify_manifest_attestation_skips_non_dict_invocation_and_materials(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
        manifest_overrides={
            "subject": None,
            "invocation": "not-a-dict",
            "materials": "not-a-list",
        },
    )

    assert proof_pack_mod.verify_manifest_attestation(pack_dir) == []


def test_parse_checksums_ignores_blank_lines(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "checksums.sha256").write_text(
        f"\n{'a' * 64}  results/final_verdict.json\n\n",
        encoding="utf-8",
    )

    entries, errors = proof_pack_mod._parse_checksums(pack_dir)
    assert errors == []
    assert entries == [("a" * 64, "results/final_verdict.json")]


def test_verify_signature_success_without_manifest_fingerprint_returns_signer(
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
    expected_fingerprint = _sign_pack(
        pack_dir,
        tmp_path,
        record_manifest_fingerprint=False,
    )
    errors, warnings, fingerprint = proof_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert errors == []
    assert warnings == []
    assert fingerprint == expected_fingerprint


def test_verify_signature_success_with_matching_fingerprint_returns_signer(
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
    errors, warnings, fingerprint = proof_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert errors == []
    assert warnings == []
    assert fingerprint == expected_fingerprint


def test_inspect_proof_pack_reports_missing_manifest_and_checksums(
    tmp_path: Path,
) -> None:
    missing_manifest = tmp_path / "missing-manifest"
    missing_manifest.mkdir()
    _write_json(missing_manifest / "checksums.sha256", {})

    result = proof_pack_mod.inspect_proof_pack(missing_manifest)
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.MISSING
    assert payload["issues"] == ["manifest.json missing in pack."]

    missing_checksums = tmp_path / "missing-checksums"
    missing_checksums.mkdir()
    _write_json(
        missing_checksums / "manifest.json",
        {
            "format": proof_pack_mod.PROOF_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )

    result = proof_pack_mod.inspect_proof_pack(missing_checksums)
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.MISSING
    assert payload["issues"] == ["checksums.sha256 missing in pack."]


def test_inspect_proof_pack_signed_pack_omits_unsigned_warning_and_reports_extras(
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
    _sign_pack(pack_dir, tmp_path)
    (pack_dir / "extra.bin").write_text("extra", encoding="utf-8")

    result = proof_pack_mod.inspect_proof_pack(pack_dir)
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.INTEGRITY
    assert payload["ok"] is False
    assert not any("pack is unsigned" in issue for issue in payload["issues"])
    assert any("extra files not covered" in issue for issue in payload["issues"])


def test_inspect_proof_pack_unsigned_clean_pack_reports_warning_without_extras(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "unsigned-pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )

    result = proof_pack_mod.inspect_proof_pack(pack_dir)
    payload = result.payload

    assert result.status == proof_pack_mod.ProofPackStatus.OK
    assert payload["ok"] is True
    assert payload["integrity"]["extra_files"] == []
    assert (
        "manifest.signature.json missing; strict verification would fail."
        in payload["issues"]
    )


def test_build_proof_pack_copies_readme_and_environment_without_optional_refs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final_verdict = tmp_path / "final.json"
    report_path = tmp_path / "report.json"
    runtime_manifest = report_path.parent / RUNTIME_MANIFEST_FILENAME
    environment = tmp_path / "environment.json"
    readme = tmp_path / "README.md"
    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(report_path, {"ok": True})
    _write_json(runtime_manifest, {"ok": True})
    _write_json(environment, {"platform": "test"})
    readme.write_text("# Proof Pack\n", encoding="utf-8")

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        lambda reports, profile: VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True},
            diagnostics=(),
        ),
        raising=True,
    )

    result = proof_pack_mod.build_proof_pack(
        tmp_path / "out-readme",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        environment_path=environment,
        readme_path=readme,
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.OK
    assert payload["ok"] is True
    manifest = json.loads(
        (tmp_path / "out-readme" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["evidence_level"] == "medium"
    assert "invocation" not in manifest
    assert "materials" not in manifest
    assert manifest["environment"]["path"] == "metadata/environment.json"
    assert (tmp_path / "out-readme" / "README.md").is_file()


def test_build_proof_pack_copies_source_repo_without_environment_or_materials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final_verdict = tmp_path / "final.json"
    report_path = tmp_path / "report.json"
    runtime_manifest = report_path.parent / RUNTIME_MANIFEST_FILENAME
    source_repo = tmp_path / "source_repo.json"
    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(report_path, {"ok": True})
    _write_json(runtime_manifest, {"ok": True})
    _write_json(source_repo, {"commit": "abc123"})

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        lambda reports, profile: VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True},
            diagnostics=(),
        ),
        raising=True,
    )

    result = proof_pack_mod.build_proof_pack(
        tmp_path / "out-source-only",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        source_repo_path=source_repo,
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.OK
    assert payload["ok"] is True
    manifest = json.loads(
        (tmp_path / "out-source-only" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["evidence_level"] == "medium"
    assert manifest["verification"]["clean_reports"] == 1
    assert (
        manifest["invocation"]["config_source"]["path"] == "metadata/source_repo.json"
    )
    assert "environment" not in manifest
    assert "materials" not in manifest
    readme_text = (tmp_path / "out-source-only" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "Evidence level: medium" in readme_text


def test_verify_proof_pack_reports_missing_manifest_and_checksums(
    tmp_path: Path,
) -> None:
    missing_manifest = tmp_path / "missing-manifest"
    missing_manifest.mkdir()
    (missing_manifest / "checksums.sha256").write_text("", encoding="utf-8")

    result = proof_pack_mod.verify_proof_pack(missing_manifest, skip_verify=True)
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.MISSING
    assert payload["errors"] == ["manifest.json missing in pack."]

    missing_checksums = tmp_path / "missing-checksums"
    missing_checksums.mkdir()
    _write_json(
        missing_checksums / "manifest.json",
        {
            "format": proof_pack_mod.PROOF_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )

    result = proof_pack_mod.verify_proof_pack(missing_checksums, skip_verify=True)
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.MISSING
    assert payload["errors"] == ["checksums.sha256 missing in pack."]


def test_verify_proof_pack_returns_format_for_invalid_manifest(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    _write_json(
        pack_dir / "manifest.json",
        {
            "format": "wrong",
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )
    (pack_dir / "checksums.sha256").write_text("", encoding="utf-8")

    result = proof_pack_mod.verify_proof_pack(pack_dir, skip_verify=True)
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.FORMAT
    assert payload["errors"]
    assert any(
        "schema validation failed" in error or "manifest format must be" in error
        for error in payload["errors"]
    )


def test_verify_proof_pack_returns_signature_failure_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )

    monkeypatch.setattr(
        proof_pack_mod,
        "_verify_signature",
        lambda pack_dir, strict: (["bad signature"], [], "FPR123"),
        raising=True,
    )

    result = proof_pack_mod.verify_proof_pack(pack_dir, skip_verify=True)
    payload = result.payload
    exit_code = result.status
    assert exit_code == proof_pack_mod.ProofPackStatus.SIGNATURE
    assert payload["errors"] == ["bad signature"]
    assert payload["signer_fingerprint"] == "FPR123"


def test_verify_proof_pack_skip_verify_fails_closed_without_signature_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )

    monkeypatch.setattr(
        proof_pack_mod,
        "_verify_reports",
        lambda *args, **kwargs: pytest.fail(
            "_verify_reports should not run when skip_verify=True"
        ),
        raising=True,
    )
    monkeypatch.setattr(
        proof_pack_mod,
        "unattested_artifacts_allowed",
        lambda: False,
        raising=True,
    )

    result = proof_pack_mod.verify_proof_pack(pack_dir, skip_verify=True)

    assert result.status == proof_pack_mod.ProofPackStatus.SIGNATURE
    assert result.payload["ok"] is False
    assert "verify" not in result.payload
    assert result.payload["warnings"] == []
    assert result.payload["errors"] == [
        "manifest.signature.json missing; signed manifest required by default."
    ]


def test_verify_proof_pack_skip_verify_allows_explicit_unattested_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    report_path, final_verdict, environment = _write_pack_scaffold(pack_dir)
    _write_manifest_and_checksums(
        pack_dir,
        report_path=report_path,
        final_verdict=final_verdict,
        environment=environment,
    )

    monkeypatch.setattr(
        proof_pack_mod,
        "_verify_reports",
        lambda *args, **kwargs: pytest.fail(
            "_verify_reports should not run when skip_verify=True"
        ),
        raising=True,
    )
    monkeypatch.setattr(
        proof_pack_mod,
        "unattested_artifacts_allowed",
        lambda: True,
        raising=True,
    )

    result = proof_pack_mod.verify_proof_pack(pack_dir, skip_verify=True)

    assert result.status == proof_pack_mod.ProofPackStatus.OK
    assert result.payload["ok"] is True
    assert "verify" not in result.payload
    assert result.payload["warnings"] == [
        "manifest.signature.json missing; pack is unsigned."
    ]


def test_build_verify_result_includes_signer_and_verify_payload(tmp_path: Path) -> None:
    payload = proof_pack_mod._build_verify_result(
        pack_dir=tmp_path / "pack",
        ok=False,
        strict=True,
        skip_verify=False,
        warnings=["warn"],
        errors=["err"],
        signer_fingerprint="ABC123",
        verify_payload={"ok": False},
        status=proof_pack_mod.ProofPackStatus.OK,
    )

    assert payload.payload["signer_fingerprint"] == "ABC123"
    assert payload.payload["verify"] == {"ok": False}
