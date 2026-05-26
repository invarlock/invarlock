from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import invarlock.evidence_pack as evidence_pack_mod
import invarlock.evidence_pack_integrity as evidence_pack_integrity_mod
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome
from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME


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
            "format": "evidence-pack-signature-v1",
            "algorithm": "ed25519",
            "signing_key_fingerprint": "sha256:" + ("a" * 64),
            "public_key": {"encoding": "pem", "value": public_pem},
            "signature": {"encoding": "base64", "value": "YWJj"},
        },
    )

    errors, warnings, fingerprint = evidence_pack_integrity_mod.verify_signature(
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
    errors, payload = evidence_pack_mod._verify_reports(
        empty_pack,
        json_out_path=None,
        profile="dev",
        report_assurance="report",
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
    errors, payload = evidence_pack_mod._verify_reports(
        error_only_pack,
        json_out_path=None,
        profile="dev",
        report_assurance="report",
    )
    assert errors == [
        "No clean reports found in pack (only error-injection reports present)."
    ]
    assert payload is None

    json_out_off = tmp_path / "report-assurance-off.json"
    errors, payload = evidence_pack_mod._verify_reports(
        error_only_pack,
        json_out_path=json_out_off,
        profile="dev",
        report_assurance="off",
    )
    assert errors == []
    assert payload == {
        "ok": True,
        "skipped": True,
        "reason": "report_assurance_off",
        "reports": 1,
    }
    assert json.loads(json_out_off.read_text(encoding="utf-8")) == payload

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

    def _fake_run_verify(
        reports: list[Path], *, profile: str, report_assurance: str = "report"
    ):
        verify_calls.append([str(path) for path in reports])
        if len(verify_calls) == 1:
            return VerifyExecutionResult(
                outcome=VerifyOutcome.OK,
                payload={"ok": False},
                diagnostics=(),
            )
        raise RuntimeError("ignore nested error reports")

    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        _fake_run_verify,
        raising=True,
    )
    errors, payload = evidence_pack_mod._verify_reports(
        pack_dir,
        json_out_path=json_out,
        profile="release",
        report_assurance="report",
    )
    assert errors == [
        "error-injection report verification failed: ignore nested error reports"
    ]
    assert payload == {"ok": False}
    assert len(verify_calls) == 2

    missing_result = evidence_pack_mod.inspect_evidence_pack(tmp_path / "missing")
    missing_payload = missing_result.payload
    exit_code = missing_result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.MISSING
    assert missing_payload["ok"] is False

    invalid_pack = tmp_path / "invalid"
    invalid_pack.mkdir()
    (invalid_pack / "manifest.json").write_text("{invalid", encoding="utf-8")
    (invalid_pack / "checksums.sha256").write_text("", encoding="utf-8")
    invalid_result = evidence_pack_mod.inspect_evidence_pack(invalid_pack)
    invalid_payload = invalid_result.payload
    exit_code = invalid_result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.FORMAT
    assert invalid_payload["ok"] is False
