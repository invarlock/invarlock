from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import invarlock.evidence_pack as evidence_pack_mod
from invarlock.cli.app import app
from invarlock.reporting import verify_contract as verify_mod
from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)
_ALLOW_UNVERIFIED_PROVENANCE_ENV = {"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "1"}


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
    payload = {
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
    }
    _write_json(report_path.parent / RUNTIME_MANIFEST_FILENAME, payload)


def _successful_verify_payload(reports: list[Path]) -> dict[str, object]:
    return {
        "format_version": "verify-v1",
        "ok": True,
        "reports": [str(path) for path in reports],
    }


def _successful_verify_result(
    reports: list[Path],
) -> verify_mod.VerifyExecutionResult:
    return verify_mod.VerifyExecutionResult(
        outcome=verify_mod.VerifyOutcome.OK,
        payload=_successful_verify_payload(reports),
        diagnostics=(),
    )


def _write_checksums(pack_dir: Path, rel_paths: list[str]) -> None:
    lines = []
    for rel_path in rel_paths:
        digest = _sha256_file(pack_dir / rel_path)
        lines.append(f"{digest}  {rel_path}")
    (pack_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _build_report_payload() -> dict[str, object]:
    spectral_contract = {
        "estimator": {"type": "power_iter", "iters": 4, "init": "ones"}
    }
    rmt_contract = {
        "estimator": {"type": "power_iter", "iters": 3, "init": "ones"},
        "activation_sampling": {
            "windows": {"count": 8, "indices_policy": "evenly_spaced"}
        },
    }
    return {
        "schema_version": "v1",
        "run_id": "evidence-pack-cli-test",
        "artifacts": {"generated_at": "2024-01-01T00:00:00"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "unit",
            "seq_len": 8,
            "windows": {
                "preview": 2,
                "final": 2,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 2}, "final": {"used": 2}},
                    "paired_windows": 2,
                },
            },
        },
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
        "baseline_ref": {
            "run_id": "baseline-run",
            "model_id": "model",
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        },
        "artifacts_extra": {},
        "primary_metric": {
            "kind": "ppl_causal",
            "final": 10.0,
            "preview": 10.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "spectral": {
            "evaluated": True,
            "measurement_contract": spectral_contract,
            "measurement_contract_hash": verify_mod._measurement_contract_digest(
                spectral_contract
            ),
            "measurement_contract_match": True,
        },
        "rmt": {
            "evaluated": True,
            "measurement_contract": rmt_contract,
            "measurement_contract_hash": verify_mod._measurement_contract_digest(
                rmt_contract
            ),
            "measurement_contract_match": True,
        },
        "resolved_policy": {
            "spectral": {"measurement_contract": spectral_contract},
            "rmt": {"measurement_contract": rmt_contract},
        },
        "evaluation_windows": {
            "final": {
                "logloss": [math.log(10.0)],
                "token_counts": [1],
            }
        },
    }


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
    _write_json(report, _build_report_payload())
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
        "format": "evidence-pack-v1",
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


def test_evidence_pack_help_lists_verify() -> None:
    result = CliRunner().invoke(app, ["advanced", "evidence-pack", "--help"])
    assert result.exit_code == 0
    assert "verify" in result.output
    assert "inspect" in result.output
    assert "build" in result.output
    assert "keygen" in result.output


def test_evidence_pack_verify_json_round_trip(monkeypatch, tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    json_out = tmp_path / "verify.json"

    monkeypatch.setattr(
        "invarlock.evidence_pack._run_verify_command",
        lambda reports, profile: _successful_verify_result(reports),
        raising=False,
    )

    result = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "verify",
            str(pack_dir),
            "--json",
            "--json-out",
            str(json_out),
        ],
        env=_ALLOW_UNVERIFIED_PROVENANCE_ENV,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["format_version"] == "evidence-pack-verify-v1"
    assert payload["ok"] is True
    assert payload["verify"]["format_version"] == "verify-v1"
    assert json.loads(json_out.read_text(encoding="utf-8"))["ok"] is True


def test_evidence_pack_verify_human_success(monkeypatch, tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    monkeypatch.setattr(
        "invarlock.evidence_pack._run_verify_command",
        lambda reports, profile: _successful_verify_result(reports),
        raising=False,
    )

    result = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "verify", str(pack_dir)],
        env=_ALLOW_UNVERIFIED_PROVENANCE_ENV,
    )

    assert result.exit_code == 0, result.output
    assert "warning(s)" in result.output
    assert "Evidence pack verified" in result.output


def test_evidence_pack_verify_human_failure_renders_errors(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.evidence_pack.verify_evidence_pack",
        lambda *args, **kwargs: SimpleNamespace(
            payload={
                "pack": str(pack_dir),
                "ok": False,
                "warnings": [],
                "errors": ["bad pack"],
            },
            status=evidence_pack_mod.EvidencePackStatus.INTEGRITY,
        ),
        raising=False,
    )

    result = CliRunner().invoke(
        app, ["advanced", "evidence-pack", "verify", str(pack_dir)]
    )

    assert result.exit_code == 6, result.output
    assert "Evidence pack verification failed" in result.output
    assert "bad pack" in result.output


def test_evidence_pack_verify_json_round_trip_with_verify_payload(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    monkeypatch.setattr(
        "invarlock.evidence_pack._run_verify_command",
        lambda reports, profile: _successful_verify_result(reports),
        raising=False,
    )

    result = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "verify", str(pack_dir), "--json"],
        env=_ALLOW_UNVERIFIED_PROVENANCE_ENV,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["format_version"] == "evidence-pack-verify-v1"
    assert payload["ok"] is True
    assert payload["verify"]["format_version"] == "verify-v1"
    assert payload["verify"]["ok"] is True


def test_evidence_pack_verify_json_round_trip_with_real_nested_verify(
    tmp_path: Path,
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )

    result = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "verify", str(pack_dir), "--json"],
        env=_ALLOW_UNVERIFIED_PROVENANCE_ENV,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["format_version"] == "evidence-pack-verify-v1"
    assert payload["ok"] is True
    assert payload["verify"]["format_version"] == "verify-v1"
    assert payload["verify"]["summary"]["reason"] == "ok"
    assert payload["verify"]["summary"]["ok"] is True


def test_evidence_pack_verify_rejects_missing_pack(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "verify", str(tmp_path / "missing"), "--json"],
    )

    assert result.exit_code == 3
    payload = json.loads(result.stdout.strip())
    assert payload["ok"] is False
    assert "resolution" not in payload


def test_evidence_pack_verify_human_failure(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["advanced", "evidence-pack", "verify", str(tmp_path / "missing")]
    )

    assert result.exit_code == 3
    assert "Evidence pack verification failed" in result.output
    assert "Pack directory not found" in result.output


def test_evidence_pack_inspect_json_summary(tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )

    result = CliRunner().invoke(
        app, ["advanced", "evidence-pack", "inspect", str(pack_dir), "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["format_version"] == "evidence-pack-inspect-v1"
    assert payload["ok"] is True
    assert payload["reports"]["total"] == 1
    assert payload["reports"]["clean"] == 1
    assert payload["reports"]["errors"] == 0
    assert payload["signature"]["present"] is False
    assert payload["integrity"]["checksums_bound"] is True
    assert payload["integrity"]["manifest_provenance_ok"] is True
    assert payload["strict_ready"] is False
    assert payload["evidence_level"] == "medium"
    assert any(
        "manifest.signature.json missing" in issue for issue in payload["issues"]
    )


def test_evidence_pack_inspect_human_success_and_failure(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()

    monkeypatch.setattr(
        "invarlock.cli.commands.evidence_pack.inspect_evidence_pack",
        lambda path: SimpleNamespace(
            payload={
                "pack": str(path),
                "ok": True,
                "issues": ["unsigned"],
            },
            status=evidence_pack_mod.EvidencePackStatus.OK,
        ),
        raising=False,
    )
    result = CliRunner().invoke(
        app, ["advanced", "evidence-pack", "inspect", str(pack_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "Evidence pack inspected" in result.output
    assert "unsigned" in result.output

    monkeypatch.setattr(
        "invarlock.cli.commands.evidence_pack.inspect_evidence_pack",
        lambda path: SimpleNamespace(
            payload={
                "pack": str(path),
                "ok": False,
                "issues": ["missing manifest"],
            },
            status=evidence_pack_mod.EvidencePackStatus.FORMAT,
        ),
        raising=False,
    )
    result = CliRunner().invoke(
        app, ["advanced", "evidence-pack", "inspect", str(pack_dir)]
    )
    assert result.exit_code == 4, result.output
    assert "Evidence pack inspection failed" in result.output
    assert "missing manifest" in result.output


def test_evidence_pack_build_json_round_trip(monkeypatch, tmp_path: Path) -> None:
    final_verdict = tmp_path / "final_verdict.json"
    source_repo = tmp_path / "source_repo.json"
    environment = tmp_path / "environment.json"
    model_revisions = tmp_path / "model_revisions.json"
    report = tmp_path / "evaluation.report.json"
    out_dir = tmp_path / "evidence_pack"

    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(source_repo, {"commit": "abc123"})
    _write_json(environment, {"platform": "test"})
    _write_json(model_revisions, {"models": {"org/model": {"revision": "rev1"}}})
    _write_json(report, _build_report_payload())
    _write_runtime_manifest(report)
    monkeypatch.setattr(
        "invarlock.evidence_pack._run_verify_command",
        lambda reports, profile: _successful_verify_result(reports),
        raising=False,
    )

    build = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "build",
            str(out_dir),
            "--final-verdict",
            str(final_verdict),
            "--source-repo",
            str(source_repo),
            "--environment",
            str(environment),
            "--material",
            f"model_revisions={model_revisions}",
            "--report",
            str(report),
            "--json",
        ],
    )

    assert build.exit_code == 0, build.output
    payload = json.loads(build.stdout.strip())
    assert payload["format_version"] == "evidence-pack-build-v1"
    assert payload["ok"] is True
    assert payload["reports"]["total"] == 1
    assert payload["pack"] == str(out_dir)
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "checksums.sha256").is_file()
    assert (out_dir / "results" / "final_verdict.json").is_file()
    assert (out_dir / "metadata" / "source_repo.json").is_file()
    assert (out_dir / "metadata" / "environment.json").is_file()
    assert (out_dir / "metadata" / "model_revisions.json").is_file()
    assert len(list(out_dir.glob("reports/**/evaluation.report.json"))) == 1

    verify = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "verify", str(out_dir), "--json"],
        env=_ALLOW_UNVERIFIED_PROVENANCE_ENV,
    )
    assert verify.exit_code == 0, verify.output
    verify_payload = json.loads(verify.stdout.strip())
    assert verify_payload["ok"] is True
    assert verify_payload["verify"]["ok"] is True


def test_evidence_pack_keygen_and_signed_build_round_trip(
    monkeypatch, tmp_path: Path
) -> None:
    final_verdict = tmp_path / "final_verdict.json"
    report = tmp_path / "evaluation.report.json"
    out_dir = tmp_path / "evidence_pack_signed"
    private_key = tmp_path / "evidence-pack-signing-key.pem"
    public_key = tmp_path / "evidence-pack-signing-key.pub.pem"

    _write_json(final_verdict, {"verdict": "PASS"})
    _write_json(report, _build_report_payload())
    _write_runtime_manifest(report)
    monkeypatch.setattr(
        "invarlock.evidence_pack._run_verify_command",
        lambda reports, profile: _successful_verify_result(reports),
        raising=False,
    )

    keygen = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "keygen",
            str(private_key),
            "--public-key-out",
            str(public_key),
            "--json",
        ],
    )
    assert keygen.exit_code == 0, keygen.output
    keygen_payload = json.loads(keygen.stdout.strip())
    assert keygen_payload["ok"] is True
    assert private_key.is_file()
    assert public_key.is_file()

    build = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "build",
            str(out_dir),
            "--final-verdict",
            str(final_verdict),
            "--report",
            str(report),
            "--signing-key",
            str(private_key),
            "--json",
        ],
    )
    assert build.exit_code == 0, build.output
    build_payload = json.loads(build.stdout.strip())
    assert build_payload["ok"] is True
    assert build_payload["signature"]["present"] is True
    assert (out_dir / "manifest.signature.json").is_file()

    verify = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "verify", str(out_dir), "--json"],
    )
    assert verify.exit_code == 0, verify.output
    verify_payload = json.loads(verify.stdout.strip())
    assert verify_payload["ok"] is True
    assert (
        verify_payload["signer_fingerprint"]
        == keygen_payload["signing_key_fingerprint"]
    )


def test_evidence_pack_keygen_console_paths_cover_success_and_existing_key(
    tmp_path: Path,
) -> None:
    private_key = tmp_path / "console-signing-key.pem"

    result = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "keygen", str(private_key)],
    )
    assert result.exit_code == 0, result.output
    assert "Evidence pack signing keypair created" in result.output
    assert private_key.name in result.output
    assert f"{private_key.stem}.pub.pem" in result.output
    assert "Private key:" in result.output
    assert "Public key:" in result.output
    assert "Fingerprint:" in result.output

    existing = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "keygen", str(private_key)],
    )
    assert existing.exit_code == 2, existing.output
    assert "Evidence pack key generation failed" in existing.output
    assert "private key output already exists" in existing.output


def test_evidence_pack_build_requires_reports(tmp_path: Path) -> None:
    final_verdict = tmp_path / "final_verdict.json"
    _write_json(final_verdict, {"verdict": "PASS"})

    result = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "build",
            str(tmp_path / "pack"),
            "--final-verdict",
            str(final_verdict),
            "--json",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout.strip())
    assert payload["format_version"] == "evidence-pack-build-v1"
    assert payload["ok"] is False
    assert any("at least one --report" in error for error in payload["errors"])


def test_evidence_pack_build_invalid_material_human_and_json(tmp_path: Path) -> None:
    final_verdict = tmp_path / "final_verdict.json"
    _write_json(final_verdict, {"verdict": "PASS"})

    result = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "build",
            str(tmp_path / "pack"),
            "--final-verdict",
            str(final_verdict),
            "--material",
            "bad-material",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "Invalid --material value" in result.output

    result = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "build",
            str(tmp_path / "pack-json"),
            "--final-verdict",
            str(final_verdict),
            "--material",
            "bad-material",
            "--json",
        ],
    )
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["ok"] is False
    assert "Invalid --material value" in payload["errors"][0]


def test_evidence_pack_build_human_success_and_failure(
    monkeypatch, tmp_path: Path
) -> None:
    final_verdict = tmp_path / "final_verdict.json"
    report = tmp_path / "evaluation.report.json"
    final_verdict.write_text('{"verdict":"PASS"}', encoding="utf-8")
    report.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "invarlock.cli.commands.evidence_pack.build_evidence_pack",
        lambda *args, **kwargs: SimpleNamespace(
            payload={
                "pack": str(tmp_path / "pack"),
                "ok": True,
                "warnings": ["unsigned"],
                "errors": [],
                "reports": {"total": 1},
                "verify": {"ok": True},
                "files": {"hashed": 2},
            },
            status=evidence_pack_mod.EvidencePackStatus.OK,
        ),
        raising=False,
    )
    result = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "build",
            str(tmp_path / "pack"),
            "--final-verdict",
            str(final_verdict),
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "warning(s)" in result.output
    assert "Evidence pack built" in result.output

    monkeypatch.setattr(
        "invarlock.cli.commands.evidence_pack.build_evidence_pack",
        lambda *args, **kwargs: SimpleNamespace(
            payload={
                "pack": str(tmp_path / "pack"),
                "ok": False,
                "warnings": [],
                "errors": ["build failed"],
                "reports": {"total": 1},
                "verify": None,
                "files": None,
            },
            status=evidence_pack_mod.EvidencePackStatus.REPORTS,
        ),
        raising=False,
    )
    result = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "build",
            str(tmp_path / "pack"),
            "--final-verdict",
            str(final_verdict),
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 7, result.output
    assert "Evidence pack build failed" in result.output
    assert "build failed" in result.output
