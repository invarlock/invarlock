from __future__ import annotations

from .report_summary import ReportManifestSummary


def render_evaluation_bundle_reviewer_summary(
    summary: ReportManifestSummary,
    *,
    evidence_level: str,
    has_guard_evidence: bool,
) -> str:
    """Render a short plain-text audit summary for evaluation bundles."""
    lines = [
        "InvarLock Evaluation Bundle Reviewer Summary",
        "",
        f"Evidence level: {evidence_level}",
        f"Overall status: {summary.overall_status}",
        (
            "What we tested: "
            f"model={summary.run_model or 'unknown'}, device={summary.device or 'unknown'}, "
            f"gates={summary.gates_passed}/{summary.gates_total}."
        ),
        "",
        "Why it might be wrong:",
    ]
    if has_guard_evidence:
        lines.append(
            "- Guard evidence sidecar is present, but reviewers should still compare it against the canonical evaluation report."
        )
    else:
        lines.append(
            "- No guard evidence sidecar was bundled, so this package only includes the rendered report artifacts."
        )
    lines.extend(
        [
            "- This bundle is a packaging surface, not a signed provenance envelope.",
            "",
            "Known rerun guidance:",
            "- Reproduce from the same run inputs and compare evaluation.report.json with evaluation_report.md for drift.",
            "",
            "Environment assumptions:",
            f"- Device: {summary.device or 'unknown'}",
            f"- Seed: {summary.seed if summary.seed is not None else 'unknown'}",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = ["render_evaluation_bundle_reviewer_summary"]
