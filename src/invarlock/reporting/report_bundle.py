from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from invarlock.core.backend_inventory import write_backend_inventory_sidecar
from invarlock.core.guard_evidence import (
    build_guard_evidence_payload,
    maybe_dump_guard_evidence,
)
from invarlock.core.report_inputs import ReportInputError, resolve_report_input_path
from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME

from .render_markdown import render_report_markdown
from .report_schema import validate_report
from .report_summary import (
    ReportManifestSummary,
    build_report_manifest_summary,
    derive_report_manifest_evidence_level,
)
from .report_types import RunReport
from .run_report_formatters import to_html, to_json, to_markdown

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


def save_report(
    report: RunReport,
    output_dir: str | Path,
    formats: list[str] | None = None,
    compare: RunReport | None = None,
    filename_prefix: str = "report",
) -> dict[str, Path]:
    """Persist raw run-report artifacts to disk."""

    if formats is None:
        formats = ["json", "markdown", "html"]
    if "report" in formats:
        raise ValueError(
            "Evaluation bundle persistence moved to save_evaluation_bundle()"
        )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: dict[str, Path] = {}
    suffix = "_comparison" if compare else ""

    if "json" in formats:
        json_path = output_path / f"{filename_prefix}{suffix}.json"
        json_path.write_text(to_json(report), encoding="utf-8")
        saved_files["json"] = json_path

    if "markdown" in formats:
        md_path = output_path / f"{filename_prefix}{suffix}.md"
        md_path.write_text(to_markdown(report, compare), encoding="utf-8")
        saved_files["markdown"] = md_path

    if "html" in formats:
        html_path = output_path / f"{filename_prefix}{suffix}.html"
        html_path.write_text(to_html(report, compare), encoding="utf-8")
        saved_files["html"] = html_path

    return saved_files


def save_evaluation_bundle(
    *,
    run_report: RunReport,
    output_dir: str | Path,
    evaluation_report: dict[str, Any],
    source_run_path: str | Path | None = None,
    render_optional: bool = True,
) -> dict[str, Path]:
    """Persist a prebuilt evaluation bundle and related manifest artifacts."""
    if not validate_report(evaluation_report):
        raise ValueError("Invalid evaluation report structure")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: dict[str, Path] = {}

    report_json = json.dumps(evaluation_report, indent=2, ensure_ascii=False)
    report_json_path = output_path / "evaluation.report.json"
    report_json_path.write_text(report_json, encoding="utf-8")
    saved_files["report"] = report_json_path

    report_md_path: Path | None = None
    if render_optional:
        report_md = render_report_markdown(evaluation_report)
        report_md_path = output_path / "evaluation_report.md"
        report_md_path.write_text(report_md, encoding="utf-8")
        saved_files["report_md"] = report_md_path

    if source_run_path is not None:
        try:
            resolved_run_path = resolve_report_input_path(
                source_run_path,
                expected_kind="run",
            )
        except ReportInputError:
            resolved_run_path = None
        if resolved_run_path is not None:
            run_inventory_path = resolved_run_path.parent / "backend_inventory.json"
            if run_inventory_path.is_file():
                copied_inventory_path = output_path / "backend_inventory.json"
                if run_inventory_path.resolve() != copied_inventory_path.resolve():
                    shutil.copy2(run_inventory_path, copied_inventory_path)
                saved_files["backend_inventory"] = copied_inventory_path

            runtime_manifest_path = resolved_run_path.parent / RUNTIME_MANIFEST_FILENAME
            if runtime_manifest_path.is_file():
                copied_manifest_path = output_path / RUNTIME_MANIFEST_FILENAME
                try:
                    manifest_payload = json.loads(
                        runtime_manifest_path.read_text(encoding="utf-8")
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    manifest_payload = None
                if isinstance(manifest_payload, dict):
                    raw_report_payload = manifest_payload.get("report")
                    report_payload = (
                        dict(raw_report_payload)
                        if isinstance(raw_report_payload, dict)
                        else {}
                    )
                    report_payload["filename"] = report_json_path.name
                    report_payload["path"] = str(report_json_path)
                    report_payload["sha256"] = hashlib.sha256(
                        report_json_path.read_bytes()
                    ).hexdigest()
                    manifest_payload["report"] = report_payload
                    copied_manifest_path.write_text(
                        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                elif runtime_manifest_path.resolve() != copied_manifest_path.resolve():
                    shutil.copy2(runtime_manifest_path, copied_manifest_path)
                saved_files["runtime_manifest"] = copied_manifest_path

    if "backend_inventory" not in saved_files:
        backend_inventory_path = write_backend_inventory_sidecar(
            evaluation_report,
            output_path,
        )
        if backend_inventory_path is not None:
            saved_files["backend_inventory"] = backend_inventory_path

    if render_optional and report_md_path is not None:
        write_report_manifest(
            report=run_report,
            output_path=output_path,
            evaluation_report=evaluation_report,
            report_json_path=report_json_path,
            report_md_path=report_md_path,
            saved_files=saved_files,
        )

    return saved_files


__all__ = [
    "render_evaluation_bundle_reviewer_summary",
    "save_evaluation_bundle",
    "save_report",
    "write_report_manifest",
]
