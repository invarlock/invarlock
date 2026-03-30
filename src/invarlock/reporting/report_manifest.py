"""Manifest construction and persistence for evaluation report bundles."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .evidence import maybe_dump_guard_evidence
from .report_evidence import build_guard_evidence_payload
from .report_summary import build_report_manifest_summary
from .report_types import RunReport

_NON_FATAL_EXCEPTIONS = (AttributeError, OSError, TypeError, ValueError)


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
        maybe_dump_guard_evidence(output_path, guard_payload)

        ev_file = output_path / "guards_evidence.json"
        if ev_file.exists():
            manifest["evidence"] = {"guards_evidence": str(ev_file)}

        manifest_path = output_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        saved_files["manifest"] = manifest_path
    except _NON_FATAL_EXCEPTIONS:
        # Manifest generation is best-effort.
        pass


__all__ = ["write_report_manifest"]
