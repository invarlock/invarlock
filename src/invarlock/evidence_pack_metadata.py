from __future__ import annotations

from typing import Any


def _evidence_pack_counts_from_verification(
    verification: dict[str, Any] | None,
) -> tuple[int | None, int | None, int | None]:
    if not isinstance(verification, dict):
        return None, None, None
    clean_reports = verification.get("clean_reports")
    error_reports = verification.get("error_injection_reports")
    failed_reports = verification.get("failed_reports")
    return (
        int(clean_reports) if isinstance(clean_reports, int) else None,
        int(error_reports) if isinstance(error_reports, int) else None,
        int(failed_reports) if isinstance(failed_reports, int) else None,
    )


def _derive_evidence_pack_evidence_level(
    *,
    subject_present: bool,
    checksums_bound: bool,
    clean_reports: int | None,
    failed_reports: int | None,
    has_source_repo_ref: bool,
    has_environment_ref: bool,
) -> str:
    if (
        subject_present
        and checksums_bound
        and isinstance(clean_reports, int)
        and clean_reports > 0
        and failed_reports == 0
        and has_source_repo_ref
        and has_environment_ref
    ):
        return "high"
    if (
        subject_present
        and checksums_bound
        and isinstance(clean_reports, int)
        and clean_reports > 0
    ):
        return "medium"
    return "low"


def _render_evidence_pack_readme(
    *,
    evidence_level: str,
    clean_reports: int | None,
    error_reports: int | None,
    failed_reports: int | None,
    policy_profile: str | None,
    strict_ready: bool,
    signer_fingerprint: str | None,
) -> str:
    lines = [
        "# InvarLock Evidence Pack",
        "",
        "This evidence pack bundles reports, summary reports, and metadata for offline",
        "verification. No model weights are included.",
        "",
        f"Evidence level: {evidence_level}",
        (
            "Review summary: "
            f"clean_reports={clean_reports if clean_reports is not None else 'unknown'}, "
            f"error_injection_reports={error_reports if error_reports is not None else 'unknown'}, "
            f"failed_reports={failed_reports if failed_reports is not None else 'unknown'}, "
            f"profile={policy_profile or 'unknown'}."
        ),
        "",
        "Why it might be wrong:",
    ]
    if failed_reports not in (None, 0):
        lines.append(
            "- Unexpected report verification failures were recorded; inspect results/verification_summary.json before trusting final conclusions."
        )
    else:
        lines.append(
            "- Nested report verification succeeded for the bundled clean reports, but reviewers should still inspect the underlying evaluation.report.json files."
        )
    lines.append(
        "- Error-injection reports are expected-failure evidence and should not be interpreted as clean PASS runs."
    )
    if strict_ready:
        lines.append(
            "- The pack is ready for strict verification; signed manifest and checksum sealing are present."
        )
    else:
        lines.append(
            "- By default this is evidence-grade packaging. For strong distributable evidence, require a signed manifest, strict verification, and a PASS final verdict."
        )
    if signer_fingerprint:
        lines.append(f"- Signer fingerprint: {signer_fingerprint}")

    lines.extend(
        [
            "",
            "## Verify",
            "",
            "1. Verify the manifest signature and strict pack integrity:",
            "   invarlock advanced evidence-pack verify <pack-dir> --strict",
            "",
            "2. Verify file checksums:",
            "   sha256sum -c checksums.sha256",
            "   # macOS: shasum -a 256 -c checksums.sha256",
            "",
            "3. Verify report integrity:",
            "   invarlock verify --json reports/**/evaluation.report.json",
            "",
            "Or use:",
            "  invarlock advanced evidence-pack verify <pack-dir> --strict",
            "Repo workflow alternative:",
            "  scripts/evidence_packs/verify_pack.sh --pack <pack-dir> --strict",
        ]
    )
    return "\n".join(lines) + "\n"
