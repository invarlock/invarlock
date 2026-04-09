from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from invarlock.core.report_inputs import ReportInputError, resolve_report_input_path
from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME

from .render import render_report_markdown
from .report_manifest import write_report_manifest
from .report_schema import validate_report
from .report_types import RunReport


def save_evaluation_bundle(
    *,
    run_report: RunReport,
    output_dir: str | Path,
    evaluation_report: dict[str, Any],
    source_run_path: str | Path | None = None,
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

    if source_run_path is not None:
        try:
            resolved_run_path = resolve_report_input_path(
                source_run_path,
                expected_kind="run",
            )
        except ReportInputError:
            resolved_run_path = None
        if resolved_run_path is not None:
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
