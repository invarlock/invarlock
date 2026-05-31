from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.reporting.evidence_pack.test_evidence_pack_helper_signature_and_manifest import (
    RUNTIME_MANIFEST_FILENAME,
    VerifyExecutionResult,
    VerifyOutcome,
    _sign_pack,
    _write_json,
    _write_manifest_and_checksums,
    _write_pack_scaffold,
    evidence_pack_integrity_mod,
    evidence_pack_mod,
)


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
        evidence_pack_mod,
        "_run_verify_command",
        lambda reports, *, profile, report_assurance="report": (
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
    errors, payload = evidence_pack_mod._verify_reports(
        pack_with_errors,
        json_out_path=None,
        profile="dev",
        report_assurance="report",
    )
    assert errors == [
        "expected-failure report verification did not return a JSON object."
    ]
    assert payload == {"ok": True}

    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        lambda reports, *, profile, report_assurance="report": (
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
    errors, payload = evidence_pack_mod._verify_reports(
        pack_with_errors,
        json_out_path=None,
        profile="dev",
        report_assurance="report",
    )
    assert errors == ["expected-pass report verification did not return a JSON object."]
    assert payload is None

    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        lambda reports, *, profile, report_assurance="report": (
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
    errors, payload = evidence_pack_mod._verify_reports(
        pack_with_errors,
        json_out_path=None,
        profile="dev",
        report_assurance="report",
    )
    assert errors == ["expected-pass report verification did not return a JSON object."]
    assert payload is None

    pack_clean_only = _build_pack("clean-only", with_errors=False)
    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        lambda reports, *, profile, report_assurance="report": VerifyExecutionResult(
            outcome=VerifyOutcome.POLICY_FAIL,
            payload={"ok": False},
            diagnostics=(),
        ),
        raising=True,
    )
    errors, payload = evidence_pack_mod._verify_reports(
        pack_clean_only,
        json_out_path=None,
        profile="release",
        report_assurance="report",
    )
    assert errors == ["invarlock verify reported report verification failures."]
    assert payload == {"ok": False}


def test_build_and_verify_evidence_pack_cover_usage_and_failure_paths(
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

    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out-none",
        final_verdict_path=final_verdict,
        report_paths=[],
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.USAGE
    assert "at least one --report input" in payload["errors"][0]

    existing_out = tmp_path / "existing"
    existing_out.mkdir()
    result = evidence_pack_mod.build_evidence_pack(
        existing_out,
        final_verdict_path=final_verdict,
        report_paths=[report_path],
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.USAGE
    assert "already exists" in payload["errors"][0]

    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out-release-review-weak",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        release_review=True,
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.USAGE
    assert any("--report-assurance strict" in error for error in payload["errors"])
    assert any("--signing-key" in error for error in payload["errors"])

    signing_key = tmp_path / "signing.key"
    evidence_pack_mod._generate_signing_keypair(
        signing_key,
        public_key_path=signing_key.with_suffix(".pub"),
    )
    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out-release-review-dev-profile",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        profile="dev",
        report_assurance="strict",
        signing_key_path=signing_key,
        release_review=True,
    )
    assert result.status == evidence_pack_mod.EvidencePackStatus.USAGE
    assert any("profile=dev" in error for error in result.payload["errors"])

    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out-release-review-invalid-profile",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        profile="staging",
        report_assurance="strict",
        signing_key_path=signing_key,
        release_review=True,
    )
    assert result.status == evidence_pack_mod.EvidencePackStatus.USAGE
    assert any(
        "--profile ci or --profile release" in error
        for error in result.payload["errors"]
    )

    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out-invalid-material",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        material_specs=[("../bad", material), ("../bad", material)],
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.FORMAT
    assert any("Invalid material name" in error for error in payload["errors"])
    assert any("Duplicate material name" in error for error in payload["errors"])

    runtime_manifest.unlink()
    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out-missing-sidecar",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.FORMAT
    assert any("report sidecar file not found" in error for error in payload["errors"])
    _write_json(runtime_manifest, {"ok": True})

    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        lambda reports, profile, report_assurance="report": VerifyExecutionResult(
            outcome=VerifyOutcome.POLICY_FAIL,
            payload={"ok": False},
            diagnostics=(),
        ),
        raising=True,
    )
    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out-verify-fail",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.REPORTS
    assert payload["verify"] == {"ok": False}

    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        lambda reports, profile, report_assurance="report": VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True},
            diagnostics=(),
        ),
        raising=True,
    )
    result = evidence_pack_mod.build_evidence_pack(
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
    assert exit_code == evidence_pack_mod.EvidencePackStatus.OK
    assert payload["ok"] is True
    assert payload["report_assurance"] == "report"
    assert any("README file not found" in warning for warning in payload["warnings"])

    result = evidence_pack_mod.verify_evidence_pack(
        tmp_path / "missing-pack", skip_verify=True
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.MISSING
    assert payload["ok"] is False

    result = evidence_pack_mod.verify_evidence_pack(
        tmp_path / "out-ok",
        json_out_path=(tmp_path / "out-ok" / "verify.json"),
        skip_verify=True,
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.USAGE
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
        allow_unverified_provenance=False,
        json_mode=False,
        assurance_mode="report",
    ):
        captured["reports"] = reports
        captured["profile"] = profile
        captured["json_mode"] = json_mode
        captured["assurance_mode"] = assurance_mode
        return VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True},
            diagnostics=(),
        )

    monkeypatch.setattr(
        evidence_pack_mod,
        "run_verify_reports",
        _fake_verify_contract,
        raising=False,
    )

    result = evidence_pack_mod._run_verify_command([report], profile="release")

    assert result.outcome == VerifyOutcome.OK
    assert result.payload == {"ok": True}
    assert captured["reports"] == [report]
    assert captured["profile"] == "release"
    assert captured["json_mode"] is True


def test_manual_validate_manifest_accepts_valid_optional_sections() -> None:
    payload = {
        "format": evidence_pack_mod.EVIDENCE_PACK_FORMAT,
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

    assert evidence_pack_mod._manual_validate_manifest(payload) == []


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
        evidence_pack_integrity_mod,
        "load_evidence_pack_manifest_schema",
        lambda: None,
        raising=True,
    )

    errors = evidence_pack_mod.validate_manifest(manifest_path)
    assert any("manifest format must be" in error for error in errors)


def test_validate_reference_allows_empty_path_and_digest_pair(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()

    assert (
        evidence_pack_mod._validate_reference(
            pack_dir=pack_dir,
            label="demo",
            payload={"path": None, "digest": None},
        )
        == []
    )


def test_verify_manifest_provenance_rejects_non_object_manifest(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")

    assert evidence_pack_mod.verify_manifest_provenance(pack_dir) == [
        "manifest must decode to a JSON object"
    ]


def test_verify_manifest_provenance_skips_non_dict_invocation_and_materials(
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

    assert evidence_pack_mod.verify_manifest_provenance(pack_dir) == []


def test_parse_checksums_ignores_blank_lines(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "checksums.sha256").write_text(
        f"\n{'a' * 64}  results/final_verdict.json\n\n",
        encoding="utf-8",
    )

    entries, errors = evidence_pack_mod._parse_checksums(pack_dir)
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
    errors, warnings, fingerprint = evidence_pack_mod._verify_signature(
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
    errors, warnings, fingerprint = evidence_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert errors == []
    assert warnings == []
    assert fingerprint == expected_fingerprint


def test_inspect_evidence_pack_reports_missing_manifest_and_checksums(
    tmp_path: Path,
) -> None:
    missing_manifest = tmp_path / "missing-manifest"
    missing_manifest.mkdir()
    _write_json(missing_manifest / "checksums.sha256", {})

    result = evidence_pack_mod.inspect_evidence_pack(missing_manifest)
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.MISSING
    assert payload["issues"] == ["manifest.json missing in pack."]

    missing_checksums = tmp_path / "missing-checksums"
    missing_checksums.mkdir()
    _write_json(
        missing_checksums / "manifest.json",
        {
            "format": evidence_pack_mod.EVIDENCE_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )

    result = evidence_pack_mod.inspect_evidence_pack(missing_checksums)
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.MISSING
    assert payload["issues"] == ["checksums.sha256 missing in pack."]


def test_inspect_evidence_pack_signed_pack_omits_unsigned_warning_and_reports_extras(
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

    result = evidence_pack_mod.inspect_evidence_pack(pack_dir)
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.INTEGRITY
    assert payload["ok"] is False
    assert not any("pack is unsigned" in issue for issue in payload["issues"])
    assert any("extra files not covered" in issue for issue in payload["issues"])


def test_inspect_evidence_pack_unsigned_clean_pack_reports_warning_without_extras(
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

    result = evidence_pack_mod.inspect_evidence_pack(pack_dir)
    payload = result.payload

    assert result.status == evidence_pack_mod.EvidencePackStatus.OK
    assert payload["ok"] is True
    assert payload["integrity"]["extra_files"] == []
    assert (
        "manifest.signature.json missing; strict verification would fail."
        in payload["issues"]
    )


def test_build_evidence_pack_copies_readme_and_environment_without_optional_refs(
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
    readme.write_text("# Evidence Pack\n", encoding="utf-8")

    monkeypatch.setattr(
        evidence_pack_mod,
        "_run_verify_command",
        lambda reports, profile, report_assurance="report": VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True},
            diagnostics=(),
        ),
        raising=True,
    )

    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out-readme",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        environment_path=environment,
        readme_path=readme,
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.OK
    assert payload["ok"] is True
    manifest = json.loads(
        (tmp_path / "out-readme" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["evidence_level"] == "medium"
    assert "invocation" not in manifest
    assert "materials" not in manifest
    assert manifest["environment"]["path"] == "metadata/environment.json"
    assert (tmp_path / "out-readme" / "README.md").is_file()


def test_build_evidence_pack_copies_source_repo_without_environment_or_materials(
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
        evidence_pack_mod,
        "_run_verify_command",
        lambda reports, profile, report_assurance="report": VerifyExecutionResult(
            outcome=VerifyOutcome.OK,
            payload={"ok": True},
            diagnostics=(),
        ),
        raising=True,
    )

    result = evidence_pack_mod.build_evidence_pack(
        tmp_path / "out-source-only",
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        source_repo_path=source_repo,
    )
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.OK
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


def test_verify_evidence_pack_reports_missing_manifest_and_checksums(
    tmp_path: Path,
) -> None:
    missing_manifest = tmp_path / "missing-manifest"
    missing_manifest.mkdir()
    (missing_manifest / "checksums.sha256").write_text("", encoding="utf-8")

    result = evidence_pack_mod.verify_evidence_pack(missing_manifest, skip_verify=True)
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.MISSING
    assert payload["errors"] == ["manifest.json missing in pack."]

    missing_checksums = tmp_path / "missing-checksums"
    missing_checksums.mkdir()
    _write_json(
        missing_checksums / "manifest.json",
        {
            "format": evidence_pack_mod.EVIDENCE_PACK_FORMAT,
            "checksums_sha256": "checksums.sha256",
            "checksums_sha256_digest": "a" * 64,
        },
    )

    result = evidence_pack_mod.verify_evidence_pack(missing_checksums, skip_verify=True)
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.MISSING
    assert payload["errors"] == ["checksums.sha256 missing in pack."]


def test_verify_evidence_pack_returns_format_for_invalid_manifest(
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

    result = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.FORMAT
    assert payload["errors"]
    assert any(
        "schema validation failed" in error or "manifest format must be" in error
        for error in payload["errors"]
    )


def test_verify_evidence_pack_returns_signature_failure_payload(
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
        evidence_pack_mod,
        "_verify_signature",
        lambda pack_dir, strict: (["bad signature"], [], "FPR123"),
        raising=True,
    )

    result = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)
    payload = result.payload
    exit_code = result.status
    assert exit_code == evidence_pack_mod.EvidencePackStatus.SIGNATURE
    assert payload["errors"] == ["bad signature"]
    assert payload["signer_fingerprint"] == "FPR123"
