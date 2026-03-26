from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .evidence import maybe_dump_guard_evidence
from .render import compute_console_validation_block
from .report import to_evaluation_report, to_html, to_json, to_markdown
from .report_types import RunReport


def save_report(
    report: RunReport,
    output_dir: str | Path,
    formats: list[str] | None = None,
    compare: RunReport | None = None,
    baseline: RunReport | None = None,
    filename_prefix: str = "report",
) -> dict[str, Path]:
    """Persist report artifacts to disk."""

    if formats is None:
        formats = ["json", "markdown", "html"]
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

    if "report" in formats:
        if baseline is None:
            raise ValueError(
                "Baseline report required for evaluation report generation"
            )

        report_json = to_evaluation_report(report, baseline, format="json")
        report_json_path = output_path / "evaluation.report.json"
        report_json_path.write_text(report_json, encoding="utf-8")
        saved_files["report"] = report_json_path

        report_md = to_evaluation_report(report, baseline, format="markdown")
        report_md_path = output_path / "evaluation_report.md"
        report_md_path.write_text(report_md, encoding="utf-8")
        saved_files["report_md"] = report_md_path

        _write_report_manifest(
            report=report,
            output_path=output_path,
            report_json=report_json,
            report_json_path=report_json_path,
            report_md_path=report_md_path,
            saved_files=saved_files,
        )

    return saved_files


def _write_report_manifest(
    *,
    report: RunReport,
    output_path: Path,
    report_json: str,
    report_json_path: Path,
    report_md_path: Path,
    saved_files: dict[str, Path],
) -> None:
    try:
        meta_obj: object = report.get("meta")
        meta_dict: dict[str, Any] = meta_obj if isinstance(meta_obj, dict) else {}
        manifest: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "files": {
                "evaluation_report_json": str(report_json_path),
                "evaluation_report_markdown": str(report_md_path),
            },
            "summary": {
                "run_model": meta_dict.get("model_id"),
                "device": meta_dict.get("device"),
                "seed": meta_dict.get("seed"),
            },
        }

        evaluation_report_obj = json.loads(report_json)
        if isinstance(evaluation_report_obj, dict):
            block = compute_console_validation_block(evaluation_report_obj)
            rows = block.get("rows", []) or []
            gates_total = len(rows)
            gates_passed = sum(
                1 for row in rows if isinstance(row, dict) and bool(row.get("ok"))
            )
            overall_status = "PASS" if block.get("overall_pass") else "FAIL"

            pm_ratio = None
            pm = evaluation_report_obj.get("primary_metric", {}) or {}
            if isinstance(pm, dict):
                ratio = pm.get("ratio_vs_baseline")
                if isinstance(ratio, int | float):
                    pm_ratio = float(ratio)

            manifest["summary"].update(
                {
                    "overall_status": overall_status,
                    "primary_metric_ratio": pm_ratio,
                    "gates_passed": gates_passed,
                    "gates_total": gates_total,
                }
            )

        guard_payload = _build_guard_evidence_payload(report)
        maybe_dump_guard_evidence(output_path, guard_payload)

        ev_file = output_path / "guards_evidence.json"
        if ev_file.exists():
            manifest["evidence"] = {"guards_evidence": str(ev_file)}

        manifest_path = output_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        saved_files["manifest"] = manifest_path
    except Exception:
        # Manifest generation is best-effort.
        pass


def _build_guard_evidence_payload(report: RunReport) -> dict[str, Any]:
    try:
        guard_ctx = report.get("guards") or []
    except Exception:
        guard_ctx = []

    if not isinstance(guard_ctx, list) or not guard_ctx:
        return {"guards_decisions": []}

    tiny: list[dict[str, object]] = []
    for guard in guard_ctx:
        if not isinstance(guard, dict):
            continue
        entry: dict[str, object] = {}
        policy = guard.get("policy") or {}
        if isinstance(policy, dict):
            for key in (
                "deadband",
                "min_effect_lognll",
                "max_caps",
                "sigma_quantile",
            ):
                if key in policy:
                    entry[key] = policy[key]
        if guard.get("name"):
            entry["name"] = guard.get("name")
        if entry:
            tiny.append(entry)
    return {"guards_decisions": tiny}


__all__ = ["save_report"]
