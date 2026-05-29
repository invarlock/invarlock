"""Manifest construction and persistence for evaluation report bundles."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .report_evidence import build_guard_evidence_payload, maybe_dump_guard_evidence
from .report_summary import (
    ReportManifestSummary,
    build_report_manifest_summary,
    derive_report_manifest_evidence_level,
)
from .report_types import RunReport

_NON_FATAL_EXCEPTIONS = (AttributeError, OSError, TypeError, ValueError)


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


def write_report_manifest(
    *,
    report: RunReport,
    output_path: Path,
    evaluation_report: dict[str, Any],
    report_json_path: Path,
    report_md_path: Path,
    saved_files: dict[str, Path],
) -> None:
    """Write the bundle manifest and optional guard evidence as best effort."""
    try:
        summary = build_report_manifest_summary(
            cast(dict[str, Any], report), evaluation_report
        )
        manifest: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "files": {
                "evaluation_report_json": str(report_json_path),
                "evaluation_report_markdown": str(report_md_path),
            },
            "summary": {
                "run_model": summary.run_model,
                "device": summary.device,
                "seed": summary.seed,
                "overall_status": summary.overall_status,
                "primary_metric_ratio": summary.primary_metric_ratio,
                "gates_passed": summary.gates_passed,
                "gates_total": summary.gates_total,
            },
        }

        guard_payload = build_guard_evidence_payload(report)
        ev_file = maybe_dump_guard_evidence(output_path, guard_payload)
        has_guard_evidence = ev_file is not None and ev_file.exists()
        evidence_level = derive_report_manifest_evidence_level(
            summary, has_guard_evidence=has_guard_evidence
        )
        manifest["evidence_level"] = evidence_level
        if has_guard_evidence and ev_file is not None:
            manifest["evidence"] = {"guards_evidence": str(ev_file)}

        reviewer_summary_path = output_path / "reviewer_summary.txt"
        reviewer_summary_path.write_text(
            render_evaluation_bundle_reviewer_summary(
                summary,
                evidence_level=evidence_level,
                has_guard_evidence=has_guard_evidence,
            ),
            encoding="utf-8",
        )
        manifest["files"]["reviewer_summary_txt"] = str(reviewer_summary_path)
        saved_files["reviewer_summary"] = reviewer_summary_path

        manifest_path = output_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        saved_files["manifest"] = manifest_path
    except _NON_FATAL_EXCEPTIONS:
        # Manifest generation is best-effort.
        pass


__all__ = ["render_evaluation_bundle_reviewer_summary", "write_report_manifest"]
