from __future__ import annotations

from pathlib import Path

from .report_types import RunReport
from .run_report_formatters import to_html, to_json, to_markdown


def save_report(
    report: RunReport,
    output_dir: str | Path,
    formats: list[str] | None = None,
    compare: RunReport | None = None,
    filename_prefix: str = "report",
) -> dict[str, Path]:
    """Persist run-report artifacts to disk."""

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


__all__ = ["save_report"]
