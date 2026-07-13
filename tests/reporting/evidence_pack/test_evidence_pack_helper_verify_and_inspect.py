from __future__ import annotations

from pathlib import Path

import pytest

import invarlock.evidence_pack_support as evidence_pack_support_mod
from tests.reporting._support_evidence_pack_paths import (
    VerifyExecutionResult,
    VerifyOutcome,
    _sign_pack,
    _write_json,
    _write_pack_with_manifest,
    evidence_pack_integrity_mod,
    evidence_pack_mod,
)


def test_verify_reports_covers_remaining_payload_contract_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_with_errors = _write_pack_with_manifest(
        tmp_path / "with-errors",
        with_error_report=True,
    )

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

    pack_clean_only = _write_pack_with_manifest(tmp_path / "clean-only")
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


def test_inspect_reuses_manifest_parse_and_file_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    _write_pack_with_manifest(pack_dir)
    original_load_json = evidence_pack_support_mod._load_json
    original_relative_paths = evidence_pack_support_mod._relative_file_paths
    manifest_loads = 0
    inventory_scans = 0

    def counted_load_json(path: Path):
        nonlocal manifest_loads
        manifest_loads += 1
        return original_load_json(path)

    def counted_relative_paths(path: Path):
        nonlocal inventory_scans
        inventory_scans += 1
        return original_relative_paths(path)

    monkeypatch.setattr(evidence_pack_support_mod, "_load_json", counted_load_json)
    monkeypatch.setattr(
        evidence_pack_support_mod, "_relative_file_paths", counted_relative_paths
    )

    result = evidence_pack_support_mod.inspect_evidence_pack(pack_dir)

    assert result.status is evidence_pack_support_mod.EvidencePackStatus.OK
    assert manifest_loads == 1
    assert inventory_scans == 1


def test_verify_manifest_provenance_skips_non_dict_invocation_and_materials(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    _write_pack_with_manifest(
        pack_dir,
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


def test_parse_checksums_rejects_duplicate_canonical_paths(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "checksums.sha256").write_text(
        f"{'a' * 64}  ./report.json\n{'b' * 64}  report.json\n",
        encoding="utf-8",
    )

    entries, errors = evidence_pack_mod._parse_checksums(pack_dir)

    assert entries == [
        ("a" * 64, "./report.json"),
        ("b" * 64, "report.json"),
    ]
    assert errors == [
        "checksums.sha256 line 2 duplicates path 'report.json'; "
        "each path must have exactly one checksum entry"
    ]


def test_verify_signature_success_without_manifest_fingerprint_returns_signer(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    _write_pack_with_manifest(pack_dir)
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
    _write_pack_with_manifest(pack_dir)
    expected_fingerprint = _sign_pack(pack_dir, tmp_path)
    errors, warnings, fingerprint = evidence_pack_mod._verify_signature(
        pack_dir, strict=False
    )
    assert errors == []
    assert warnings == []
    assert fingerprint == expected_fingerprint


def test_verify_signature_reuses_authenticated_manifest_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    _write_pack_with_manifest(pack_dir)
    _sign_pack(pack_dir, tmp_path)
    manifest_path = pack_dir / "manifest.json"
    manifest_reads = 0
    original_read_regular_file_bytes = (
        evidence_pack_integrity_mod.evidence_pack_json_mod.read_regular_file_bytes
    )

    def counted_read_regular_file_bytes(path: Path, *, label: str) -> bytes:
        nonlocal manifest_reads
        if path == manifest_path:
            manifest_reads += 1
        return original_read_regular_file_bytes(path, label=label)

    monkeypatch.setattr(
        evidence_pack_integrity_mod.evidence_pack_json_mod,
        "read_regular_file_bytes",
        counted_read_regular_file_bytes,
    )

    errors, warnings, _fingerprint = evidence_pack_mod._verify_signature(
        pack_dir, strict=False
    )

    assert errors == []
    assert warnings == []
    assert manifest_reads == 1


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
    _write_pack_with_manifest(pack_dir)
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
    _write_pack_with_manifest(pack_dir)

    result = evidence_pack_mod.inspect_evidence_pack(pack_dir)
    payload = result.payload

    assert result.status == evidence_pack_mod.EvidencePackStatus.OK
    assert payload["ok"] is True
    assert payload["integrity"]["extra_files"] == []
    assert (
        "manifest.signature.json missing; strict verification would fail."
        in payload["issues"]
    )


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
    assert exit_code == evidence_pack_mod.EvidencePackStatus.SIGNATURE
    assert any("signed manifest required" in error for error in payload["errors"])


def test_verify_evidence_pack_returns_signature_failure_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    _write_pack_with_manifest(pack_dir)

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
