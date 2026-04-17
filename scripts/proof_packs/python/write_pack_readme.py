from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _derive_evidence_level(
    *,
    subject_present: bool,
    clean_reports: int | None,
    failed_reports: int | None,
    has_source_repo: bool,
    has_environment: bool,
) -> str:
    if (
        subject_present
        and isinstance(clean_reports, int)
        and clean_reports > 0
        and failed_reports == 0
        and has_source_repo
        and has_environment
    ):
        return "high"
    if subject_present and isinstance(clean_reports, int) and clean_reports > 0:
        return "medium"
    return "low"


def _render_readme(
    *,
    evidence_level: str,
    clean_reports: int | None,
    error_reports: int | None,
    failed_reports: int | None,
    policy_profile: str | None,
) -> str:
    lines = [
        "# InvarLock Proof Pack",
        "",
        "This proof pack bundles reports, summary reports, and metadata for offline",
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
            "- Unexpected report verification failures were recorded; inspect results/verification_summary.json before trusting downstream conclusions."
        )
    else:
        lines.append(
            "- Nested report verification succeeded for the bundled clean reports, but reviewers should still inspect the underlying evaluation.report.json files."
        )
    lines.extend(
        [
            "- Error-injection reports are expected-failure evidence and should not be interpreted as clean PASS runs.",
            "- By default this is evidence-grade packaging. For strong distributable evidence, require a signed manifest, strict verification, and a PASS final verdict.",
            "",
            "## Verify",
            "",
            "1) Verify the manifest signature (if present):",
            "   invarlock advanced proof-pack verify <pack-dir> --strict",
            "",
            "2) Verify file checksums:",
            "   sha256sum -c checksums.sha256",
            "   # macOS: shasum -a 256 -c checksums.sha256",
            "",
            "3) Verify report integrity:",
            "   invarlock verify --json reports/**/evaluation.report.json",
            "",
            "Or use:",
            "  invarlock advanced proof-pack verify <pack-dir> [--strict]",
            "Repo workflow alternative:",
            "  scripts/proof_packs/verify_pack.sh --pack <pack-dir> [--strict]",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("Usage: write_pack_readme.py <pack_dir>", file=sys.stderr)
        return 2

    pack_dir = Path(argv[0])
    verification = _load_json(pack_dir / "results" / "verification_summary.json")
    clean_reports = (
        int(verification.get("clean_reports"))
        if isinstance(verification, dict)
        and isinstance(verification.get("clean_reports"), int)
        else None
    )
    error_reports = (
        int(verification.get("error_injection_reports"))
        if isinstance(verification, dict)
        and isinstance(verification.get("error_injection_reports"), int)
        else None
    )
    failed_reports = (
        int(verification.get("failed_reports"))
        if isinstance(verification, dict)
        and isinstance(verification.get("failed_reports"), int)
        else None
    )
    policy_profile = (
        str(verification.get("policy_profile"))
        if isinstance(verification, dict)
        and isinstance(verification.get("policy_profile"), str)
        else None
    )
    evidence_level = _derive_evidence_level(
        subject_present=(pack_dir / "results" / "final_verdict.json").is_file(),
        clean_reports=clean_reports,
        failed_reports=failed_reports,
        has_source_repo=(pack_dir / "metadata" / "source_repo.json").is_file(),
        has_environment=(pack_dir / "metadata" / "environment.json").is_file(),
    )
    (pack_dir / "README.md").write_text(
        _render_readme(
            evidence_level=evidence_level,
            clean_reports=clean_reports,
            error_reports=error_reports,
            failed_reports=failed_reports,
            policy_profile=policy_profile,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
