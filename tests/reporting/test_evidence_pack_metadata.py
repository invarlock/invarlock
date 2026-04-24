from __future__ import annotations

from invarlock import evidence_pack_metadata as mod


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
