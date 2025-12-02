"""
InvarLock Unified Report Generation
==============================

Generate comprehensive reports from RunReport data in multiple formats.
Supports side-by-side comparison for bare vs guarded edit analysis.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from invarlock.cli._evidence import maybe_dump_guard_evidence

from .certificate import (
    make_certificate,
)
from .normalizer import normalize_run_report
from .render import render_certificate_markdown
from .report_types import RunReport, validate_report


def to_json(report: RunReport, indent: int = 2) -> str:
    """
    Convert RunReport to formatted JSON string.

    Args:
        report: RunReport to convert
        indent: JSON indentation level

    Returns:
        Formatted JSON string
    """
    if not validate_report(report):
        raise ValueError("Invalid RunReport structure")

    # Create a clean copy for JSON serialization
    json_data = dict(report)

    # Ensure all values are JSON serializable
    json_data = _sanitize_for_json(json_data)

    return json.dumps(json_data, indent=indent, ensure_ascii=False)


def to_markdown(
    report: RunReport | dict[str, Any],
    compare: RunReport | dict[str, Any] | None = None,
    title: str | None = None,
) -> str:
    """
    Convert RunReport to Markdown format with optional comparison.

    Args:
        report: Primary RunReport to convert
        compare: Optional second report for side-by-side comparison
        title: Optional title for the report

    Returns:
        Formatted Markdown string
    """
    # Normalize external dicts to canonical RunReport
    rp: RunReport = normalize_run_report(report) if isinstance(report, dict) else report
    cmp: RunReport | None = (
        normalize_run_report(compare) if isinstance(compare, dict) else compare
    )

    if not validate_report(rp):
        raise ValueError("Invalid primary RunReport structure")
    if cmp and not validate_report(cmp):
        raise ValueError("Invalid comparison RunReport structure")

    lines = []

    # Title
    if title:
        lines.append(f"# {title}")
    elif compare:
        lines.append("# InvarLock Evaluation Report Comparison")
    else:
        lines.append("# InvarLock Evaluation Report")

    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*")
    lines.append("")

    if cmp:
        lines.extend(_generate_comparison_markdown(rp, cmp))
    else:
        lines.extend(_generate_single_markdown(rp))

    return "\n".join(lines)


def to_html(
    report: RunReport | dict[str, Any],
    compare: RunReport | dict[str, Any] | None = None,
    title: str | None = None,
    include_css: bool = True,
) -> str:
    """
    Convert RunReport to HTML format with optional comparison.

    Args:
        report: Primary RunReport to convert
        compare: Optional second report for side-by-side comparison
        title: Optional title for the report
        include_css: Whether to include CSS styling

    Returns:
        Formatted HTML string
    """
    rp: RunReport = normalize_run_report(report) if isinstance(report, dict) else report
    cmp: RunReport | None = (
        normalize_run_report(compare) if isinstance(compare, dict) else compare
    )
    if not validate_report(rp):
        raise ValueError("Invalid primary RunReport structure")
    if cmp and not validate_report(cmp):
        raise ValueError("Invalid comparison RunReport structure")

    html_title = html.escape(
        title
        or ("InvarLock Report Comparison" if compare else "InvarLock Evaluation Report")
    )

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"    <title>{html_title}</title>",
    ]

    if include_css:
        html_parts.append(_get_default_css())

    html_parts.extend(
        [
            "</head>",
            "<body>",
            "    <div class='container'>",
            f"        <h1>{html_title}</h1>",
            f"        <p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>",
        ]
    )

    if cmp:
        html_parts.extend(_generate_comparison_html(rp, cmp))
    else:
        html_parts.extend(_generate_single_html(rp))

    html_parts.extend(["    </div>", "</body>", "</html>"])

    return "\n".join(html_parts)


def to_certificate(report: RunReport, baseline: RunReport, format: str = "json") -> str:
    """
    Convert RunReport to certificate format.

    Args:
        report: Primary RunReport to certify
        baseline: Baseline RunReport for comparison
        format: Output format ("json" or "markdown")

    Returns:
        Formatted certificate string
    """
    if not validate_report(report):
        raise ValueError("Invalid primary RunReport structure")

    if not _validate_baseline_or_report(baseline):
        raise ValueError("Invalid baseline RunReport structure")

    # Generate certificate
    certificate = make_certificate(report, baseline)

    if format == "json":
        return json.dumps(certificate, indent=2, ensure_ascii=False)
    elif format == "markdown":
        return render_certificate_markdown(certificate)
    else:
        raise ValueError(f"Unsupported certificate format: {format}")


def save_report(
    report: RunReport,
    output_dir: str | Path,
    formats: list[str] | None = None,
    compare: RunReport | None = None,
    baseline: RunReport | None = None,
    filename_prefix: str = "report",
) -> dict[str, Path]:
    """
    Save RunReport in multiple formats to a directory.

    Args:
        report: RunReport to save
        output_dir: Directory to save reports in
        formats: List of formats to generate ("json", "markdown", "html", "cert")
        compare: Optional comparison report
        baseline: Optional baseline report for certificate generation
        filename_prefix: Prefix for generated filenames

    Returns:
        Dictionary mapping format names to generated file paths
    """
    if formats is None:
        formats = ["json", "markdown", "html"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    suffix = "_comparison" if compare else ""

    if "json" in formats:
        json_path = output_path / f"{filename_prefix}{suffix}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(to_json(report))
        saved_files["json"] = json_path

    if "markdown" in formats:
        md_path = output_path / f"{filename_prefix}{suffix}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(to_markdown(report, compare))
        saved_files["markdown"] = md_path

    if "html" in formats:
        html_path = output_path / f"{filename_prefix}{suffix}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(to_html(report, compare))
        saved_files["html"] = html_path

    if "cert" in formats:
        if baseline is None:
            raise ValueError("Baseline report required for certificate generation")

        # Generate certificate JSON in canonical path/name
        cert_json_path = output_path / "evaluation.cert.json"
        with open(cert_json_path, "w", encoding="utf-8") as f:
            f.write(to_certificate(report, baseline, format="json"))
        saved_files["cert"] = cert_json_path

        # Also emit a markdown variant for human consumption
        cert_md_path = output_path / f"{filename_prefix}_certificate.md"
        with open(cert_md_path, "w", encoding="utf-8") as f:
            f.write(to_certificate(report, baseline, format="markdown"))
        saved_files["cert_md"] = cert_md_path

        # Emit a lightweight manifest to serve as an evidence bundle index
        try:
            from datetime import datetime as _dt

            manifest = {
                "generated_at": _dt.now().isoformat(),
                "files": {
                    "certificate_json": str(cert_json_path),
                    "certificate_markdown": str(cert_md_path),
                },
                "summary": {
                    "run_model": (report.get("meta", {}) or {}).get("model_id"),
                    "device": (report.get("meta", {}) or {}).get("device"),
                    "seed": (report.get("meta", {}) or {}).get("seed"),
                },
            }
            # Write debug evidence (tiny) when requested via env
            guard_payload = {}
            try:
                guard_ctx = report.get("guards") or []
            except Exception:
                guard_ctx = []
            if isinstance(guard_ctx, list) and guard_ctx:
                tiny: list[dict] = []
                for g in guard_ctx:
                    if isinstance(g, dict):
                        entry: dict[str, object] = {}
                        pol = g.get("policy") or {}
                        if isinstance(pol, dict):
                            for k in (
                                "deadband",
                                "min_effect_lognll",
                                "max_caps",
                                "sigma_quantile",
                            ):
                                if k in pol:
                                    entry[k] = pol[k]
                        if g.get("name"):
                            entry["name"] = g.get("name")
                        if entry:
                            tiny.append(entry)
                if tiny:
                    guard_payload = {"guards_decisions": tiny}
            # The helper will no-op unless INVARLOCK_EVIDENCE_DEBUG=1
            if guard_payload:
                maybe_dump_guard_evidence(output_path, guard_payload)
            else:
                maybe_dump_guard_evidence(output_path, {"guards_decisions": []})

            ev_file = Path(output_path) / "guards_evidence.json"
            if ev_file.exists():
                manifest["evidence"] = {"guards_evidence": str(ev_file)}

            (output_path / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            saved_files["manifest"] = output_path / "manifest.json"
        except Exception:
            # Non-fatal; manifest is best-effort
            pass

    return saved_files


# ── Private helper functions ──────────────────────────────────────────────


def _validate_baseline_or_report(baseline: RunReport | dict[str, Any]) -> bool:
    """
    Validate that a baseline is either a valid RunReport or a valid baseline format.

    Args:
        baseline: Baseline data to validate

    Returns:
        True if valid, False otherwise
    """
    # First try to validate as a RunReport
    if isinstance(baseline, dict) and validate_report(cast(RunReport, baseline)):
        return True

    # If not a RunReport, check if it's a valid baseline format
    try:
        # Check for baseline schema (v1 only)
        if isinstance(baseline, dict):
            schema_version = baseline.get("schema_version")
            if schema_version in ["baseline-v1"]:
                # Validate required baseline fields
                required_keys = {"meta", "metrics"}
                if all(key in baseline for key in required_keys):
                    # Baseline must include primary_metric with at least a final value
                    metrics = baseline.get("metrics", {})
                    pm = (
                        metrics.get("primary_metric")
                        if isinstance(metrics, dict)
                        else None
                    )
                    if isinstance(pm, dict) and (pm.get("final") is not None):
                        return True
        return False
    except (KeyError, TypeError):
        return False


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize data for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, int | float | str | bool | type(None)):
        return obj
    elif hasattr(obj, "isoformat"):  # datetime
        return obj.isoformat()
    else:
        return str(obj)


def _generate_single_markdown(report: RunReport) -> list[str]:
    """Generate markdown for a single report."""
    lines = []

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Model**: {report['meta']['model_id']}")
    lines.append(f"- **Edit**: {report['edit']['name']}")
    # Primary metric (canonical)
    pm = (
        report.get("metrics", {}).get("primary_metric")
        if isinstance(report.get("metrics"), dict)
        else None
    )
    if isinstance(pm, dict) and pm:
        kind = str(pm.get("kind") or "primary")
        prev = pm.get("preview")
        fin = pm.get("final")
        ratio = pm.get("ratio_vs_baseline")
        parts = [f"- **Primary Metric** ({kind})"]
        if isinstance(prev, int | float):
            parts.append(f"preview={prev:.3f}")
        if isinstance(fin, int | float):
            parts.append(f"final={fin:.3f}")
        if isinstance(ratio, int | float):
            parts.append(f"ratio_vs_baseline={ratio:.3f}")
        lines.append(" — ".join(parts) if len(parts) > 1 else parts[0])
    else:
        # When primary_metric is absent, do not attempt legacy fallbacks
        lines.append("- **Primary Metric**: unavailable")
    lines.append(
        f"- **Parameters Changed**: {report['edit']['deltas']['params_changed']:,}"
    )
    lines.append(
        f"- **Latency**: {report['metrics']['latency_ms_per_tok']:.2f} ms/token"
    )
    lines.append(f"- **Memory**: {report['metrics']['memory_mb_peak']:.1f} MB")
    lines.append("")

    # Model Information
    lines.append("## Model Information")
    lines.append("")
    lines.append(f"- **Model ID**: {report['meta']['model_id']}")
    lines.append(f"- **Adapter**: {report['meta']['adapter']}")
    lines.append(f"- **Device**: {report['meta']['device']}")
    lines.append(f"- **Commit**: {report['meta']['commit'][:8]}...")
    lines.append(f"- **Timestamp**: {report['meta']['ts']}")
    lines.append("")

    # Evaluation Data
    lines.append("## Evaluation Configuration")
    lines.append("")
    lines.append(f"- **Dataset**: {report['data']['dataset']}")
    lines.append(f"- **Split**: {report['data']['split']}")
    lines.append(f"- **Sequence Length**: {report['data']['seq_len']}")
    lines.append(f"- **Preview Samples**: {report['data']['preview_n']}")
    lines.append(f"- **Final Samples**: {report['data']['final_n']}")
    lines.append("")

    # Performance Metrics
    if isinstance(pm, dict) and pm:
        lines.append("## Primary Metric")
        lines.append("")
        lines.append("| Kind | Preview | Final | Ratio vs Baseline |")
        lines.append("|------|---------|-------|-------------------|")
        kind = str(pm.get("kind") or "primary")
        prev = pm.get("preview")
        fin = pm.get("final")
        ratio = pm.get("ratio_vs_baseline")

        def _fmt(x):
            return f"{x:.3f}" if isinstance(x, int | float) else "N/A"

        lines.append(f"| {kind} | {_fmt(prev)} | {_fmt(fin)} | {_fmt(ratio)} |")
        lines.append("")
        # Append system metrics
        lines.append("## System Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(
            f"| Latency (ms/token) | {report['metrics']['latency_ms_per_tok']:.2f} |"
        )
        lines.append(
            f"| Peak Memory (MB) | {report['metrics']['memory_mb_peak']:.1f} |"
        )
        lines.append("")
    else:
        lines.append("## Performance Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        # Primary metric block is unavailable; show system metrics only
        lines.append(
            f"| Latency (ms/token) | {report['metrics']['latency_ms_per_tok']:.2f} |"
        )
        lines.append(
            f"| Peak Memory (MB) | {report['metrics']['memory_mb_peak']:.1f} |"
        )
        lines.append("")

    # Edit Details
    lines.append("## Edit Details")
    lines.append("")
    lines.append(f"- **Edit Type**: {report['edit']['name']}")
    lines.append(f"- **Plan Digest**: `{report['edit']['plan_digest'][:16]}...`")
    lines.append("")
    lines.append("### Parameter Changes")
    lines.append("")
    deltas = report["edit"]["deltas"]
    lines.append("| Change Type | Count |")
    lines.append("|-------------|-------|")
    lines.append(f"| Parameters Changed | {deltas['params_changed']:,} |")
    lines.append(f"| Layers Modified | {deltas['layers_modified']} |")
    if deltas["sparsity"] is not None:
        lines.append(f"| Overall Sparsity | {deltas['sparsity']:.3f} |")
    lines.append("")

    # Guard Reports
    if report["guards"]:
        lines.append("## Guard Reports")
        lines.append("")
        for guard in report["guards"]:
            lines.append(f"### {guard['name']}")
            lines.append("")

            # Guard metrics
            if guard["metrics"]:
                lines.append("**Metrics:**")
                for metric, value in guard["metrics"].items():
                    lines.append(f"- {metric}: {value}")
                lines.append("")

            # Actions taken
            if guard["actions"]:
                lines.append("**Actions:**")
                for action in guard["actions"]:
                    lines.append(f"- {action}")
                lines.append("")

            # Violations
            if guard["violations"]:
                lines.append("**Violations:**")
                for violation in guard["violations"]:
                    lines.append(f"- ⚠️ {violation}")
                lines.append("")

    # Status Flags
    lines.append("## Status")
    lines.append("")

    # Determine overall status
    guards_passed = all(
        len(guard.get("violations", [])) == 0 for guard in report["guards"]
    )
    has_rollback = report["flags"]["rollback_reason"] is not None
    guard_recovery = report["flags"]["guard_recovered"]

    if has_rollback:
        lines.append(f"- 🔄 **ROLLBACK**: {report['flags']['rollback_reason']}")
        lines.append("- ❌ Pipeline did not complete successfully")
    elif guard_recovery:
        lines.append("- ✅ Guard recovery was triggered")
        lines.append("- ⚠️ Some guards detected issues but were resolved")
    elif guards_passed:
        lines.append("- ✅ **SUCCESS**: Pipeline completed successfully")
        lines.append(f"- 🛡️ All {len(report['guards'])} guards passed validation")
        lines.append("- 📊 Model modifications were approved and finalized")
    else:
        lines.append("- ⚠️ Some guards reported violations")
        lines.append("- 🔍 Review guard reports above for details")

    # Add performance summary based on primary metric
    metrics_map_sum = cast(dict[str, Any], dict(report["metrics"]))
    pm_sum = (
        metrics_map_sum.get("primary_metric")
        if isinstance(metrics_map_sum, dict)
        else None
    )
    ratio_val = None
    if isinstance(pm_sum, dict):
        rv = pm_sum.get("ratio_vs_baseline")
        if isinstance(rv, int | float):
            ratio_val = float(rv)
    params_changed = report["edit"]["deltas"]["params_changed"]

    lines.append("")
    lines.append("### Summary")
    lines.append(f"- **Parameters Modified**: {params_changed:,}")
    if isinstance(ratio_val, float):
        lines.append(f"- **Performance Impact**: PM ratio {ratio_val:.3f}")

    if params_changed > 0 and isinstance(ratio_val, float):
        impact = (
            "significant"
            if ratio_val > 1.1
            else "minimal"
            if ratio_val < 1.05
            else "moderate"
        )
        lines.append(
            f"- **Assessment**: {impact.title()} model changes with {impact} performance impact"
        )

    return lines


def _generate_comparison_markdown(report1: RunReport, report2: RunReport) -> list[str]:
    """Generate side-by-side comparison markdown."""
    lines = []

    # Comparison Summary
    lines.append("## Comparison Summary")
    lines.append("")
    lines.append("| Metric | Report 1 | Report 2 | Delta |")
    lines.append("|--------|----------|----------|-------|")

    # Compare primary metric (final) when present in both reports
    pm1 = (
        report1.get("metrics", {}).get("primary_metric")
        if isinstance(report1.get("metrics"), dict)
        else None
    )
    pm2 = (
        report2.get("metrics", {}).get("primary_metric")
        if isinstance(report2.get("metrics"), dict)
        else None
    )
    if isinstance(pm1, dict) and isinstance(pm2, dict):
        k1 = str(pm1.get("kind") or "primary")
        k2 = str(pm2.get("kind") or "primary")
        label = f"Primary Metric ({k1})" if k1 == k2 else "Primary Metric"
        f1 = pm1.get("final")
        f2 = pm2.get("final")
        if isinstance(f1, int | float) and isinstance(f2, int | float):
            delta = f2 - f1
            sym = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
            lines.append(f"| {label} | {f1:.3f} | {f2:.3f} | {sym} {delta:+.3f} |")
    # If primary metrics are missing, omit the comparison row rather than falling back

    # Latency comparison
    lat1 = report1["metrics"]["latency_ms_per_tok"]
    lat2 = report2["metrics"]["latency_ms_per_tok"]
    lat_delta = lat2 - lat1
    lat_symbol = "📈" if lat_delta > 0 else "📉" if lat_delta < 0 else "➡️"
    lines.append(
        f"| Latency (ms/tok) | {lat1:.2f} | {lat2:.2f} | {lat_symbol} {lat_delta:+.2f} |"
    )

    # Memory comparison
    mem1 = report1["metrics"]["memory_mb_peak"]
    mem2 = report2["metrics"]["memory_mb_peak"]
    mem_delta = mem2 - mem1
    mem_symbol = "📈" if mem_delta > 0 else "📉" if mem_delta < 0 else "➡️"
    lines.append(
        f"| Memory (MB) | {mem1:.1f} | {mem2:.1f} | {mem_symbol} {mem_delta:+.1f} |"
    )

    lines.append("")

    # Side-by-side details
    lines.append("## Detailed Comparison")
    lines.append("")
    lines.append("### Model Information")
    lines.append("")
    lines.append("| Aspect | Report 1 | Report 2 |")
    lines.append("|--------|----------|----------|")
    lines.append(
        f"| Model | {report1['meta']['model_id']} | {report2['meta']['model_id']} |"
    )
    lines.append(f"| Edit | {report1['edit']['name']} | {report2['edit']['name']} |")
    lines.append(
        f"| Device | {report1['meta']['device']} | {report2['meta']['device']} |"
    )
    lines.append("")

    # Parameter changes comparison
    lines.append("### Parameter Changes")
    lines.append("")
    lines.append("| Change Type | Report 1 | Report 2 | Delta |")
    lines.append("|-------------|----------|----------|-------|")

    delta1 = report1["edit"]["deltas"]
    delta2 = report2["edit"]["deltas"]

    for key in ["params_changed", "layers_modified"]:
        val1_obj = delta1.get(key, 0)
        val2_obj = delta2.get(key, 0)
        if isinstance(val1_obj, int | float | str):
            try:
                val1_i = int(val1_obj)
            except Exception:
                val1_i = 0
        else:
            val1_i = 0
        if isinstance(val2_obj, int | float | str):
            try:
                val2_i = int(val2_obj)
            except Exception:
                val2_i = 0
        else:
            val2_i = 0
        diff = val2_i - val1_i
        diff_str = f"{diff:+,}" if key == "params_changed" else f"{diff:+d}"
        lines.append(
            f"| {key.replace('_', ' ').title()} | {val1_i:,} | {val2_i:,} | {diff_str} |"
        )

    lines.append("")

    # Guard comparison
    if report1["guards"] or report2["guards"]:
        lines.append("### Guard Reports")
        lines.append("")

        # Get all guard names
        guard_names1 = {g["name"] for g in report1["guards"]}
        guard_names2 = {g["name"] for g in report2["guards"]}
        all_guards = sorted(guard_names1 | guard_names2)

        for guard_name in all_guards:
            lines.append(f"#### {guard_name}")
            lines.append("")

            guard1 = next(
                (g for g in report1["guards"] if g["name"] == guard_name), None
            )
            guard2 = next(
                (g for g in report2["guards"] if g["name"] == guard_name), None
            )

            status1 = "Present" if guard1 else "Absent"
            status2 = "Present" if guard2 else "Absent"

            lines.append(f"- **Report 1**: {status1}")
            lines.append(f"- **Report 2**: {status2}")

            if guard1 and guard2:
                if guard1["violations"] or guard2["violations"]:
                    lines.append("")
                    lines.append("**Violations:**")
                    lines.append(f"- Report 1: {len(guard1['violations'])} violations")
                    lines.append(f"- Report 2: {len(guard2['violations'])} violations")

            lines.append("")

    return lines


def _generate_single_html(report: RunReport) -> list[str]:
    """Generate HTML for a single report."""
    lines = []

    # Convert markdown to HTML structure
    md_lines = _generate_single_markdown(report)

    # This is a simplified conversion - in a full implementation,
    # you might use a proper markdown-to-HTML converter
    lines.append("        <div class='report-content'>")

    in_table = False
    for line in md_lines:
        line = line.strip()

        if line.startswith("# "):
            lines.append(f"            <h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines.append(f"            <h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            lines.append(f"            <h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("| ") and "|" in line[1:]:
            if not in_table:
                lines.append("            <table class='metrics-table'>")
                in_table = True

            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if all(cell.startswith("-") for cell in cells):
                continue  # Skip separator row

            if cells[0] in ["Metric", "Change Type", "Aspect"]:
                lines.append("                <thead><tr>")
                for cell in cells:
                    lines.append(f"                    <th>{html.escape(cell)}</th>")
                lines.append("                </tr></thead><tbody>")
            else:
                lines.append("                <tr>")
                for cell in cells:
                    lines.append(f"                    <td>{html.escape(cell)}</td>")
                lines.append("                </tr>")
        elif line.startswith("- "):
            if in_table:
                lines.append("                </tbody></table>")
                in_table = False
            lines.append(f"            <li>{html.escape(line[2:])}</li>")
        elif line == "":
            if in_table:
                lines.append("                </tbody></table>")
                in_table = False
        else:
            if in_table:
                lines.append("                </tbody></table>")
                in_table = False
            if line:
                lines.append(f"            <p>{html.escape(line)}</p>")

    if in_table:
        lines.append("                </tbody></table>")

    lines.append("        </div>")

    return lines


def _generate_comparison_html(report1: RunReport, report2: RunReport) -> list[str]:
    """Generate HTML for comparison reports."""
    lines = []

    # Similar to single report but with comparison layout
    md_lines = _generate_comparison_markdown(report1, report2)

    lines.append("        <div class='comparison-content'>")

    # Convert markdown lines to HTML (simplified)
    in_table = False
    for line in md_lines:
        line = line.strip()

        if line.startswith("## "):
            lines.append(f"            <h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            lines.append(f"            <h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("| ") and "|" in line[1:]:
            if not in_table:
                lines.append("            <table class='comparison-table'>")
                in_table = True

            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if all(cell.startswith("-") for cell in cells):
                continue

            if "Metric" in cells[0] or "Aspect" in cells[0]:
                lines.append("                <thead><tr>")
                for cell in cells:
                    lines.append(f"                    <th>{html.escape(cell)}</th>")
                lines.append("                </tr></thead><tbody>")
            else:
                lines.append("                <tr>")
                for cell in cells:
                    lines.append(f"                    <td>{html.escape(cell)}</td>")
                lines.append("                </tr>")
        elif line == "":
            if in_table:
                lines.append("                </tbody></table>")
                in_table = False
        else:
            if in_table:
                lines.append("                </tbody></table>")
                in_table = False
            if line:
                lines.append(f"            <p>{html.escape(line)}</p>")

    if in_table:
        lines.append("                </tbody></table>")

    lines.append("        </div>")

    return lines


def _get_default_css() -> str:
    """Get default CSS styling for HTML reports."""
    return """    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }
        h3 { color: #7f8c8d; margin-top: 25px; }
        .timestamp { color: #95a5a6; font-style: italic; margin-bottom: 30px; }
        .metrics-table, .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }
        .metrics-table th, .metrics-table td,
        .comparison-table th, .comparison-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        .metrics-table th, .comparison-table th {
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }
        .metrics-table tr:hover, .comparison-table tr:hover {
            background-color: #f8f9fa;
        }
        li { margin: 5px 0; }
        code {
            background: #f1f2f6;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Consolas', monospace;
        }
        .comparison-content { display: block; }
        @media (max-width: 768px) {
            .container { padding: 15px; }
            .metrics-table, .comparison-table { font-size: 14px; }
        }
    </style>"""


# Export public API
__all__ = ["to_json", "to_markdown", "to_html", "to_certificate", "save_report"]
