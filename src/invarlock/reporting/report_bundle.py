from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .render import render_report_markdown
from .report_manifest import write_report_manifest
from .report_schema import validate_report
from .report_types import RunReport


def save_evaluation_bundle(
    *,
    run_report: RunReport,
    output_dir: str | Path,
    evaluation_report: dict[str, Any],
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

    report_md = render_report_markdown(evaluation_report)
    report_md_path = output_path / "evaluation_report.md"
    report_md_path.write_text(report_md, encoding="utf-8")
    saved_files["report_md"] = report_md_path

    write_report_manifest(
        report=run_report,
        output_path=output_path,
        evaluation_report=evaluation_report,
        report_json_path=report_json_path,
        report_md_path=report_md_path,
        saved_files=saved_files,
    )

    return saved_files


__all__ = ["save_evaluation_bundle"]
