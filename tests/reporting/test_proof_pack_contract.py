from __future__ import annotations

import hashlib
import json
from pathlib import Path

import invarlock.proof_pack as proof_pack_mod
from invarlock.proof_pack import (
    ProofPackStatus,
    validate_manifest,
    verify_manifest_attestation,
    verify_proof_pack,
)
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_ref(path: Path, rel_path: str) -> dict[str, str]:
    return {
        "path": rel_path,
        "digest": f"sha256:{_sha256_file(path)}",
    }


def _write_runtime_manifest(report_path: Path) -> None:
    _write_json(
        report_path.parent / RUNTIME_MANIFEST_FILENAME,
        {
            "manifest_version": 1,
            "generated_at_utc": "2026-03-21T00:00:00+00:00",
            "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
            "execution_mode": "container",
            "report": {
                "filename": report_path.name,
                "path": report_path.as_posix(),
                "sha256": _sha256_file(report_path),
            },
            "config": {
                "path": None,
                "sha256": None,
                "source": "missing",
            },
            "runtime": {
                "container_execution": True,
                "image_digest": _VALID_TEST_IMAGE_DIGEST,
                "image_ref": "invarlock-runtime:local",
                "allow_network": False,
                "allow_remote_code": False,
                "allow_third_party_plugins": False,
            },
        },
    )


def _write_checksums(pack_dir: Path, rel_paths: list[str]) -> None:
    lines = []
    for rel_path in rel_paths:
        digest = _sha256_file(pack_dir / rel_path)
        lines.append(f"{digest}  {rel_path}")
    (pack_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _build_pack(pack_dir: Path, *, report_rel_path: str) -> Path:
    final_verdict = pack_dir / "results/final_verdict.json"
    source_repo = pack_dir / "metadata/source_repo.json"
    environment = pack_dir / "metadata/environment.json"
    materials = pack_dir / "metadata/model_revisions.json"
    report = pack_dir / report_rel_path

    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(source_repo, {"commit": "abc123"})
    _write_json(environment, {"platform": "test"})
    _write_json(materials, {"models": {"org/model": {"revision": "rev1"}}})
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}", encoding="utf-8")
    _write_runtime_manifest(report)

    covered = [
        "results/final_verdict.json",
        "metadata/source_repo.json",
        "metadata/environment.json",
        "metadata/model_revisions.json",
        report_rel_path,
        str((Path(report_rel_path).parent / RUNTIME_MANIFEST_FILENAME).as_posix()),
    ]
    _write_checksums(pack_dir, covered)

    manifest = {
        "format": "proof-pack-v1",
        "checksums_sha256": "checksums.sha256",
        "checksums_sha256_digest": _sha256_file(pack_dir / "checksums.sha256"),
        "subject": {
            "name": "final_verdict",
            **_digest_ref(final_verdict, "results/final_verdict.json"),
        },
        "invocation": {
            "config_source": _digest_ref(source_repo, "metadata/source_repo.json")
        },
        "environment": _digest_ref(environment, "metadata/environment.json"),
        "materials": [
            {
                "name": "model_revisions",
                **_digest_ref(materials, "metadata/model_revisions.json"),
            }
        ],
    }
    _write_json(pack_dir / "manifest.json", manifest)
    return pack_dir


def test_proof_pack_manifest_and_attestation_round_trip(tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )

    assert validate_manifest(pack_dir / "manifest.json") == []
    assert verify_manifest_attestation(pack_dir) == []

    result = verify_proof_pack(pack_dir, skip_verify=True)
    payload = result.payload
    exit_code = result.status
    assert exit_code == ProofPackStatus.SIGNATURE
    assert payload["ok"] is False
    assert payload["errors"] == [
        "manifest.json.asc missing; signed manifest required by default."
    ]


def test_proof_pack_verify_rejects_json_out_inside_pack(tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )

    result = verify_proof_pack(
        pack_dir, json_out_path=pack_dir / "verify.json", skip_verify=True
    )
    payload = result.payload
    exit_code = result.status

    assert exit_code == ProofPackStatus.USAGE
    assert payload["ok"] is False
    assert "--json-out must point outside the pack directory." in payload["errors"]


def test_proof_pack_verify_strict_rejects_extra_files(tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    (pack_dir / "extra.txt").write_text("extra", encoding="utf-8")
    original_verify_gpg = proof_pack_mod._verify_gpg
    proof_pack_mod._verify_gpg = lambda pack_dir, strict: ([], [], None)

    try:
        result = verify_proof_pack(pack_dir, skip_verify=True, strict=True)
    finally:
        proof_pack_mod._verify_gpg = original_verify_gpg

    payload = result.payload
    exit_code = result.status
    assert exit_code == ProofPackStatus.INTEGRITY
    assert payload["ok"] is False
    assert any("extra files not covered" in error for error in payload["errors"])


def test_proof_pack_verify_requires_clean_reports(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/errors/noop/evaluation.report.json",
    )
    monkeypatch.setattr(
        proof_pack_mod,
        "unattested_artifacts_allowed",
        lambda: True,
        raising=True,
    )

    result = verify_proof_pack(pack_dir)
    payload = result.payload
    exit_code = result.status

    assert exit_code == ProofPackStatus.REPORTS
    assert payload["ok"] is False
    assert any("No clean reports found" in error for error in payload["errors"])


def test_proof_pack_verify_writes_nested_verify_json(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    json_out = tmp_path / "verify.json"

    monkeypatch.setattr(
        proof_pack_mod,
        "_run_verify_command",
        lambda reports, profile: VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={
                "format_version": "verify-v1",
                "ok": True,
                "reports": [str(path) for path in reports],
            },
            diagnostics=(),
        ),
    )
    monkeypatch.setattr(
        proof_pack_mod,
        "unattested_artifacts_allowed",
        lambda: True,
        raising=True,
    )

    result = verify_proof_pack(pack_dir, json_out_path=json_out)
    payload = result.payload
    exit_code = result.status

    assert exit_code == ProofPackStatus.OK
    assert payload["verify"]["format_version"] == "verify-v1"
    assert json.loads(json_out.read_text(encoding="utf-8"))["ok"] is True
