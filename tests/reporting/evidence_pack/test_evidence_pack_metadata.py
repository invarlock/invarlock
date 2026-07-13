from __future__ import annotations

from invarlock import evidence_pack_support as mod


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
