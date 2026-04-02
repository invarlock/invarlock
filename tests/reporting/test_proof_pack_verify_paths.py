from __future__ import annotations

import json
from pathlib import Path

import pytest

import invarlock.proof_pack as proof_pack_mod
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome
from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


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

    checksum_lines = [
        f"{proof_pack_mod._sha256_bytes(final_verdict.read_bytes())}  results/final_verdict.json",
        f"{proof_pack_mod._sha256_bytes(environment.read_bytes())}  metadata/environment.json",
        f"{proof_pack_mod._sha256_bytes(report_path.read_bytes())}  reports/model/clean/noop/evaluation.report.json",
        f"{proof_pack_mod._sha256_bytes((report_path.parent / RUNTIME_MANIFEST_FILENAME).read_bytes())}  reports/model/clean/noop/{RUNTIME_MANIFEST_FILENAME}",
    ]
    checksums_path = pack_dir / "checksums.sha256"
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    _write_json(
        pack_dir / "manifest.json",
        {
            "format": proof_pack_mod.PROOF_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": proof_pack_mod._sha256_bytes(
                checksums_path.read_bytes()
            ),
            "subject": {
                "name": "final_verdict",
                "path": "results/final_verdict.json",
                "digest": proof_pack_mod._sha256_file(final_verdict),
            },
            "environment": {
                "path": "metadata/environment.json",
                "digest": proof_pack_mod._sha256_file(environment),
            },
        },
    )
    return report_path, final_verdict, environment


def test_attestation_helpers_cover_reference_and_no_extra_paths(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    report_path, _final_verdict, _environment = _write_pack_scaffold(pack_dir)
    source_repo = pack_dir / "metadata" / "source_repo.json"
    material = pack_dir / "metadata" / "evidence.json"
    _write_json(source_repo, {"commit": "abc123"})
    _write_json(material, {"ok": True})

    checksums_path = pack_dir / "checksums.sha256"
    checksums_path.write_text(
        checksums_path.read_text(encoding="utf-8")
        + f"{proof_pack_mod._sha256_bytes(source_repo.read_bytes())}  metadata/source_repo.json\n"
        + f"{proof_pack_mod._sha256_bytes(material.read_bytes())}  metadata/evidence.json\n",
        encoding="utf-8",
    )
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["checksums_sha256_digest"] = proof_pack_mod._sha256_bytes(
        checksums_path.read_bytes()
    )
    manifest["invocation"] = {
        "config_source": {
            "path": "metadata/source_repo.json",
            "digest": proof_pack_mod._sha256_file(source_repo),
        }
    }
    manifest["materials"] = [
        {
            "name": "evidence",
            "path": "metadata/evidence.json",
            "digest": proof_pack_mod._sha256_file(material),
        }
    ]
    _write_json(pack_dir / "manifest.json", manifest)

    assert (
        proof_pack_mod._path_within_dir(pack_dir, pack_dir.parent / "outside.json")
        is False
    )
    assert proof_pack_mod.verify_manifest_attestation(pack_dir) == []
    covered_paths = set(proof_pack_mod._relative_file_paths(pack_dir))
    assert proof_pack_mod._verify_no_extra_files(
        pack_dir, covered_paths=covered_paths, strict=True
    ) == ([], [])
    assert report_path.is_file()


def test_verify_reports_success_writes_json_and_records_error_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    _write_pack_scaffold(pack_dir)
    error_dir = pack_dir / "reports" / "model" / "errors" / "noop"
    error_dir.mkdir(parents=True, exist_ok=True)
    (error_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
    json_out = tmp_path / "verify.json"

    def _fake_run_verify(reports: list[Path], *, profile: str) -> VerifyExecutionResult:
        if "errors" in reports[0].as_posix():
            return VerifyExecutionResult(
                outcome=VerifyOutcome.OK,
                payload={"ok": False},
                diagnostics=(),
            )
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True},
            diagnostics=(),
        )

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        _fake_run_verify,
        raising=True,
    )

    errors, payload = proof_pack_mod._verify_reports(
        pack_dir, json_out_path=json_out, profile="release"
    )

    assert errors == []
    assert payload is not None
    assert payload["error_injection"]["verify"] == {"ok": False}
    assert json.loads(json_out.read_text(encoding="utf-8")) == payload


def test_verify_signature_bundle_error_paths(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    _write_pack_scaffold(pack_dir)
    (pack_dir / "manifest.signature.json").write_text("{invalid", encoding="utf-8")

    errors, warnings, fingerprint = proof_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert "manifest.signature.json is not valid JSON" in errors[0]
    assert warnings == []
    assert fingerprint is None

    _write_json(
        pack_dir / "manifest.signature.json",
        {
            "format": "proof-pack-signature-v1",
            "algorithm": "ed25519",
            "signing_key_fingerprint": "sha256:" + ("a" * 64),
            "public_key": {"encoding": "pem", "value": "bad-key"},
            "signature": {"encoding": "base64", "value": "bad"},
        },
    )
    errors, warnings, fingerprint = proof_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert "manifest signature verification failed." in errors[0]
    assert warnings == []
    assert fingerprint is None


def test_verify_proof_pack_covers_success_integrity_and_report_failure_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _success_signature(_pack_dir: Path, *, strict: bool):
        return [], [], "ABC123"

    pack_success = tmp_path / "success"
    _write_pack_scaffold(pack_success)
    seen: dict[str, object] = {}

    def _success_verify_reports(
        pack_dir: Path, *, json_out_path: Path | None, profile: str
    ) -> tuple[list[str], dict[str, object]]:
        seen["pack_dir"] = pack_dir
        seen["json_out_path"] = json_out_path
        seen["profile"] = profile
        return [], {"ok": True}

    monkeypatch.setattr(
        proof_pack_mod,
        "_verify_signature",
        _success_signature,
        raising=True,
    )
    monkeypatch.setattr(
        proof_pack_mod,
        "_verify_reports",
        _success_verify_reports,
        raising=True,
    )

    json_out = tmp_path / "proof-pack-verify.json"
    result = proof_pack_mod.verify_proof_pack(
        pack_success,
        json_out_path=json_out,
        skip_verify=False,
        strict=True,
        profile="release",
    )
    assert result.status == proof_pack_mod.ProofPackStatus.OK
    assert result.payload["ok"] is True
    assert result.payload["strict"] is True
    assert result.payload["signer_fingerprint"] == "ABC123"
    assert result.payload["verify"] == {"ok": True}
    assert seen == {
        "pack_dir": pack_success,
        "json_out_path": json_out,
        "profile": "release",
    }

    pack_integrity = tmp_path / "integrity"
    _write_pack_scaffold(pack_integrity)
    (pack_integrity / "extra.bin").write_text("extra", encoding="utf-8")
    integrity_result = proof_pack_mod.verify_proof_pack(
        pack_integrity,
        skip_verify=True,
        strict=True,
    )
    assert integrity_result.status == proof_pack_mod.ProofPackStatus.INTEGRITY
    assert any(
        "extra files not covered" in error
        for error in integrity_result.payload["errors"]
    )

    pack_reports = tmp_path / "reports"
    _write_pack_scaffold(pack_reports)
    monkeypatch.setattr(
        proof_pack_mod,
        "_verify_reports",
        lambda pack_dir, *, json_out_path, profile: (["verify failed"], {"ok": False}),
        raising=True,
    )
    reports_result = proof_pack_mod.verify_proof_pack(
        pack_reports,
        skip_verify=False,
        strict=False,
    )
    assert reports_result.status == proof_pack_mod.ProofPackStatus.REPORTS
    assert reports_result.payload["errors"] == ["verify failed"]
    assert reports_result.payload["verify"] == {"ok": False}
