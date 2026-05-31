from __future__ import annotations

from pathlib import Path

import pytest

from tests.reporting.evidence_pack.test_evidence_pack_helper_signature_and_manifest import (
    _write_manifest_and_checksums,
    _write_pack_scaffold,
    evidence_pack_mod,
)


def test_verify_evidence_pack_skip_verify_fails_closed_without_signature_override(
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
        "_verify_reports",
        lambda *args, **kwargs: pytest.fail(
            "_verify_reports should not run when skip_verify=True"
        ),
        raising=True,
    )
    monkeypatch.setattr(
        evidence_pack_mod,
        "unverified_provenance_allowed",
        lambda: False,
        raising=True,
    )

    result = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status == evidence_pack_mod.EvidencePackStatus.SIGNATURE
    assert result.payload["ok"] is False
    assert "verify" not in result.payload
    assert result.payload["warnings"] == []
    assert result.payload["errors"] == [
        "manifest.signature.json missing; signed manifest required by default."
    ]


def test_verify_evidence_pack_skip_verify_allows_explicit_unverified_provenance_override(
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
        "_verify_reports",
        lambda *args, **kwargs: pytest.fail(
            "_verify_reports should not run when skip_verify=True"
        ),
        raising=True,
    )
    monkeypatch.setattr(
        evidence_pack_mod,
        "unverified_provenance_allowed",
        lambda: True,
        raising=True,
    )

    result = evidence_pack_mod.verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status == evidence_pack_mod.EvidencePackStatus.OK
    assert result.payload["ok"] is True
    assert "verify" not in result.payload
    assert result.payload["warnings"] == [
        "manifest.signature.json missing; pack is unsigned."
    ]


def test_build_verify_result_includes_signer_and_verify_payload(tmp_path: Path) -> None:
    payload = evidence_pack_mod._build_verify_result(
        pack_dir=tmp_path / "pack",
        ok=False,
        strict=True,
        skip_verify=False,
        warnings=["warn"],
        errors=["err"],
        signer_fingerprint="ABC123",
        verify_payload={"ok": False},
        status=evidence_pack_mod.EvidencePackStatus.OK,
    )

    assert payload.payload["signer_fingerprint"] == "ABC123"
    assert payload.payload["verify"] == {"ok": False}


def test_evidence_pack_helper_paths_cover_counts_low_evidence_and_failed_readme() -> (
    None
):
    assert evidence_pack_mod._evidence_pack_counts_from_verification(
        {
            "clean_reports": 2,
            "error_injection_reports": 1,
            "failed_reports": 0,
        }
    ) == (2, 1, 0)
    assert (
        evidence_pack_mod._derive_evidence_pack_evidence_level(
            subject_present=False,
            checksums_bound=True,
            clean_reports=0,
            failed_reports=1,
            has_source_repo_ref=False,
            has_environment_ref=False,
        )
        == "low"
    )
    readme = evidence_pack_mod._render_evidence_pack_readme(
        evidence_level="low",
        clean_reports=2,
        error_reports=1,
        failed_reports=1,
        policy_profile="release",
        strict_ready=False,
        signer_fingerprint=None,
    )
    assert "Unexpected report verification failures" in readme


def test_build_verify_result_handles_invalid_manifest_json(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "manifest.json").write_text("{not-json", encoding="utf-8")

    payload = evidence_pack_mod._build_verify_result(
        pack_dir=pack_dir,
        ok=False,
        strict=False,
        skip_verify=True,
        warnings=[],
        errors=["boom"],
        signer_fingerprint=None,
        verify_payload=None,
        status=evidence_pack_mod.EvidencePackStatus.FORMAT,
    )

    assert payload.payload["evidence_level"] is None
    assert "signer_fingerprint" not in payload.payload
    assert "verify" not in payload.payload
