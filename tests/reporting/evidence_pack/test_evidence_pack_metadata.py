from __future__ import annotations

import json
from pathlib import Path

from invarlock import evidence_pack_support as mod
from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_evidence_pack_counts_from_verification_extracts_only_integer_counts() -> None:
    assert mod._evidence_pack_counts_from_verification(None) == (None, None, None)
    assert mod._evidence_pack_counts_from_verification(
        {
            "clean_reports": 2,
            "error_injection_reports": "3",
            "failed_reports": 0,
        }
    ) == (2, None, 0)


def test_derive_evidence_pack_evidence_level_requires_full_high_evidence_set() -> None:
    assert (
        mod._derive_evidence_pack_evidence_level(
            subject_present=True,
            checksums_bound=True,
            clean_reports=2,
            failed_reports=0,
            has_source_repo_ref=True,
            has_environment_ref=True,
        )
        == "high"
    )
    assert (
        mod._derive_evidence_pack_evidence_level(
            subject_present=True,
            checksums_bound=True,
            clean_reports=1,
            failed_reports=1,
            has_source_repo_ref=False,
            has_environment_ref=False,
        )
        == "medium"
    )
    assert (
        mod._derive_evidence_pack_evidence_level(
            subject_present=False,
            checksums_bound=False,
            clean_reports=None,
            failed_reports=None,
            has_source_repo_ref=False,
            has_environment_ref=False,
        )
        == "low"
    )


def test_render_evidence_pack_readme_covers_failed_and_strict_variants() -> None:
    rendered = mod._render_evidence_pack_readme(
        evidence_level="high",
        clean_reports=4,
        error_reports=1,
        failed_reports=2,
        policy_profile="release",
        strict_ready=True,
        signer_fingerprint="abc123",
    )

    assert "Evidence level: high" in rendered
    assert "clean_reports=4" in rendered
    assert "error_injection_reports=1" in rendered
    assert "failed_reports=2" in rendered
    assert "profile=release" in rendered
    assert "Unexpected report verification failures were recorded" in rendered
    assert "The pack is ready for strict verification" in rendered
    assert "Signer fingerprint: abc123" in rendered


def test_render_evidence_pack_readme_covers_default_guidance_without_failures() -> None:
    rendered = mod._render_evidence_pack_readme(
        evidence_level="low",
        clean_reports=None,
        error_reports=None,
        failed_reports=0,
        policy_profile=None,
        strict_ready=False,
        signer_fingerprint=None,
    )

    assert "Evidence level: low" in rendered
    assert "clean_reports=unknown" in rendered
    assert "error_injection_reports=unknown" in rendered
    assert "failed_reports=0" in rendered
    assert "profile=unknown" in rendered
    assert "Nested report verification succeeded" in rendered
    assert "evidence-grade packaging" in rendered
    assert "Signer fingerprint" not in rendered


def test_collect_build_evidence_pack_errors_validates_optional_report_sidecars(
    tmp_path: Path,
) -> None:
    final_verdict = tmp_path / "final_verdict.json"
    report = tmp_path / "report" / "evaluation.report.json"
    runtime_manifest = report.parent / RUNTIME_MANIFEST_FILENAME
    sidecar = report.parent / "edit_metadata.json"
    _write_json(final_verdict, {"ok": True})
    _write_json(report, {"ok": True})
    _write_json(runtime_manifest, {"schema": "runtime"})
    sidecar.write_text("[", encoding="utf-8")

    errors = mod._collect_build_evidence_pack_errors(
        out_dir=tmp_path / "out",
        final_verdict_path=final_verdict,
        report_paths=[report],
        source_repo_path=None,
        environment_path=None,
        material_specs=[],
        signing_key_path=None,
    )

    assert any("edit_metadata.json is not valid JSON" in error for error in errors)


def test_copy_build_evidence_pack_artifacts_copies_optional_report_sidecars(
    tmp_path: Path,
) -> None:
    final_verdict = tmp_path / "final_verdict.json"
    report = tmp_path / "report" / "evaluation.report.json"
    runtime_manifest = report.parent / RUNTIME_MANIFEST_FILENAME
    sidecar = report.parent / "memory_report.json"
    out_dir = tmp_path / "pack"
    _write_json(final_verdict, {"ok": True})
    _write_json(report, {"ok": True})
    _write_json(runtime_manifest, {"schema": "runtime"})
    _write_json(sidecar, {"rss_mb": 12})

    _final_dest, rel_paths, material_refs = mod._copy_build_evidence_pack_artifacts(
        out_dir=out_dir,
        final_verdict_path=final_verdict,
        report_paths=[report],
        source_repo_path=None,
        environment_path=None,
        material_specs=[],
    )

    assert material_refs == []
    assert "reports/report-001/memory_report.json" in rel_paths
    assert (out_dir / "reports/report-001/memory_report.json").is_file()
