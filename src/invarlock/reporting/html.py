"""
Structured HTML exporter for evaluation reports.

The canonical content still comes from the Markdown renderer, but the HTML shell
adds summary cues and quick links so reviewers can orient faster in a browser.
"""

from __future__ import annotations

import re
from html import escape
from importlib import import_module
from typing import Any

from .branding import BRAND_TAGLINE
from .render_markdown import render_report_markdown
from .report_schema import validate_report
from .report_summary import compute_console_validation_block

markdown_module: Any | None = None
try:
    markdown_module = import_module("markdown")
except ImportError:  # pragma: no cover - optional dependency
    markdown_module = None


_STATUS_BADGES = {
    "\u2705 PASS": '<span class="badge pass">PASS</span>',
    "\u2705 OK": '<span class="badge pass">OK</span>',
    "\u274c FAIL": '<span class="badge fail">FAIL</span>',
    "\u26a0\ufe0f WARN": '<span class="badge warn">WARN</span>',
    "\u26a0 WARN": '<span class="badge warn">WARN</span>',
}
_HEADING_RE = re.compile(r"<h([23]) id=\"([^\"]+)\">(.*?)</h\1>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def _apply_status_badges(html_body: str) -> str:
    updated = html_body
    for token, replacement in _STATUS_BADGES.items():
        updated = updated.replace(token, replacement)
    return updated


def _escape_raw_html(md: str) -> str:
    """Disable raw HTML passthrough while preserving normal Markdown syntax."""

    return escape(md, quote=False)


def _strip_tags(value: str) -> str:
    text = _TAG_RE.sub(" ", value)
    return " ".join(text.split()).strip()


def _extract_outline_entries(html_body: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for _level, anchor, inner_html in _HEADING_RE.findall(html_body):
        label = _strip_tags(inner_html)
        if label:
            entries.append((anchor, label))
    return entries


def _render_outline(entries: list[tuple[str, str]]) -> str:
    if not entries:
        return ""
    items = "".join(
        f'<li><a href="#{escape(anchor)}">{escape(label)}</a></li>'
        for anchor, label in entries
    )
    return (
        '<aside class="report-outline" aria-label="Report quick links">'
        '<p class="outline-eyebrow">Quick Links</p>'
        "<ul>"
        f"{items}"
        "</ul>"
        "</aside>"
    )


def _summary_items(evaluation_report: dict[str, Any]) -> list[tuple[str, str, str]]:
    validation = evaluation_report.get("validation")
    validation_map = validation if isinstance(validation, dict) else {}
    primary_metric = evaluation_report.get("primary_metric")
    metric_map = primary_metric if isinstance(primary_metric, dict) else {}
    provenance = evaluation_report.get("provenance")
    provenance_map = provenance if isinstance(provenance, dict) else {}
    linked_runs_ready = False
    edited = provenance_map.get("edited")
    baseline = provenance_map.get("baseline")
    if isinstance(edited, dict) and isinstance(baseline, dict):
        linked_runs_ready = bool(edited.get("report_path")) and bool(
            baseline.get("report_path")
        )
    if "overall_pass" in validation_map:
        overall_pass = bool(validation_map.get("overall_pass"))
    else:
        overall_pass = bool(
            compute_console_validation_block(evaluation_report).get("overall_pass")
        )
    overall_status = "PASS" if overall_pass else "FAIL"
    metric_kind = str(metric_map.get("kind") or "primary metric")
    return [
        ("Overall", overall_status, overall_status.lower()),
        ("Primary Metric", metric_kind, "plain"),
        (
            "Linked Run Reports",
            "Ready" if linked_runs_ready else "Unavailable",
            "plain",
        ),
        ("Workflow", "Verify -> Review -> Share", "plain"),
    ]


def _render_summary_strip(evaluation_report: dict[str, Any]) -> str:
    cards: list[str] = []
    for label, value, tone in _summary_items(evaluation_report):
        badge = ""
        if tone in {"pass", "fail", "warn"}:
            badge = f" summary-chip-{tone}"
        cards.append(
            f'<article class="summary-chip{badge}"><p>{escape(label)}</p>'
            f"<strong>{escape(value)}</strong></article>"
        )
    return f'<section class="summary-strip">{"".join(cards)}</section>'


def render_report_html(evaluation_report: dict[str, Any]) -> str:
    """Render an evaluation report as an HTML document with quick navigation."""

    if not validate_report(evaluation_report):
        raise ValueError("Invalid evaluation report structure")
    md = render_report_markdown(evaluation_report)
    outline = ""
    if markdown_module is None:
        body = f'<div class="report-body"><pre class="invarlock-md">{escape(md)}</pre></div>'
    else:
        html_body = markdown_module.markdown(
            _escape_raw_html(md),
            extensions=["tables", "fenced_code", "toc"],
            extension_configs={"toc": {"permalink": False}},
        )
        html_body = _apply_status_badges(html_body)
        outline = _render_outline(_extract_outline_entries(html_body))
        body = f'<div class="report-body"><div class="invarlock-md">{html_body}</div></div>'
    summary_strip = _render_summary_strip(evaluation_report)
    shell = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>InvarLock Evaluation Report</title>"
        "<style>"
        ":root{--pass:#1f7a46;--fail:#b42318;--warn:#a15c07;--ink:#17212b;"
        "--muted:#52606d;--bg:#eef3f8;--bg-accent:#dfe9f3;--panel:#f7fafc;"
        "--panel-strong:#ffffff;--border:#cbd5e1;--shadow:rgba(15,23,42,0.12);"
        "--accent:#0f5f8c;--accent-soft:#d8ebf7}"
        "@media (prefers-color-scheme: dark){"
        ":root{--pass:#3fb36b;--fail:#ff7b72;--warn:#f2b44f;--ink:#e6edf3;"
        "--muted:#9fb0c0;--bg:#0f1722;--bg-accent:#162334;--panel:#111c2a;"
        "--panel-strong:#162334;--border:#2c3e50;--shadow:rgba(0,0,0,0.35);"
        "--accent:#7cc7ff;--accent-soft:#1b3044}}"
        "body{margin:0;min-height:100vh;padding:30px;color:var(--ink);"
        'font-family:"Avenir Next","Segoe UI Variable","Segoe UI",sans-serif;'
        "line-height:1.65;background:"
        "radial-gradient(circle at top right,var(--bg-accent),transparent 26%),"
        "linear-gradient(180deg,var(--bg),var(--panel))}"
        ".report-shell{max-width:1180px;margin:0 auto}"
        ".report-header{margin:0 auto 18px;padding:24px 26px;border:1px solid var(--border);"
        "border-radius:20px;background:linear-gradient(135deg,var(--panel-strong),var(--panel));"
        "box-shadow:0 18px 48px var(--shadow)}"
        ".eyebrow{margin:0 0 8px 0;font-size:0.78rem;font-weight:700;letter-spacing:0.12em;"
        "text-transform:uppercase;color:var(--accent)}"
        ".report-header h1{margin:0;font-size:2rem;line-height:1.1;"
        'font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif}'
        ".report-header p{margin:10px 0 0 0;max-width:52rem;color:var(--muted)}"
        ".brand-lockup{display:flex;align-items:center;gap:10px;margin-bottom:8px}"
        ".brand-mark{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:8px;"
        "background:var(--accent);color:var(--panel-strong);font-weight:800;letter-spacing:0}"
        ".summary-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;"
        "margin:0 0 18px 0}"
        ".summary-chip{padding:14px 16px;border-radius:18px;border:1px solid var(--border);"
        "background:var(--panel-strong);box-shadow:0 10px 24px var(--shadow)}"
        ".summary-chip p{margin:0 0 6px 0;font-size:0.78rem;font-weight:700;"
        "letter-spacing:0.06em;text-transform:uppercase;color:var(--muted)}"
        ".summary-chip strong{font-size:1rem;line-height:1.3}"
        ".summary-chip-pass strong{color:var(--pass)}"
        ".summary-chip-fail strong{color:var(--fail)}"
        ".summary-chip-warn strong{color:var(--warn)}"
        ".report-grid{display:grid;grid-template-columns:minmax(220px,260px) minmax(0,1fr);gap:18px;align-items:start}"
        ".report-outline{position:sticky;top:24px;padding:18px;border-radius:20px;"
        "border:1px solid var(--border);background:var(--panel-strong);box-shadow:0 18px 48px var(--shadow)}"
        ".outline-eyebrow{margin:0 0 10px 0;font-size:0.78rem;font-weight:700;"
        "letter-spacing:0.08em;text-transform:uppercase;color:var(--accent)}"
        ".report-outline ul{list-style:none;padding:0;margin:0;display:grid;gap:8px}"
        ".report-outline a{display:block;padding:8px 10px;border-radius:12px;text-decoration:none;"
        "color:var(--ink);background:transparent}"
        ".report-outline a:hover{background:var(--accent-soft)}"
        ".report-body{padding:28px;background:var(--panel-strong);border:1px solid var(--border);"
        "border-radius:24px;box-shadow:0 18px 48px var(--shadow)}"
        ".invarlock-md{max-width:960px;margin:0 auto}"
        ".invarlock-md>:first-child{margin-top:0}"
        'h1,h2,h3{line-height:1.2;font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif;scroll-margin-top:24px}'
        "h1{margin-top:0}h2,h3{margin-top:1.5em}"
        "p,li{color:var(--ink)}"
        "a{color:var(--accent)}"
        "table{border-collapse:collapse;width:100%;margin:14px 0;background:var(--panel-strong)}"
        "th,td{border:1px solid var(--border);padding:8px 10px;text-align:left;vertical-align:top}"
        "th{background:var(--panel);font-weight:700}"
        "code,pre{background:var(--panel);border-radius:10px}"
        "code{padding:0.12rem 0.35rem}"
        "pre{padding:14px;overflow:auto;border:1px solid var(--border)}"
        "blockquote{margin:16px 0;padding:0 0 0 16px;border-left:4px solid var(--border);"
        "color:var(--muted)}"
        "hr{border:none;border-top:1px solid var(--border);margin:24px 0}"
        ".badge{display:inline-block;padding:2px 8px;border-radius:999px;"
        "font-size:0.75rem;font-weight:700;letter-spacing:0.02em;color:#fff}"
        ".badge.pass{background:var(--pass)}"
        ".badge.fail{background:var(--fail)}"
        ".badge.warn{background:var(--warn)}"
        "@media (max-width:940px){.summary-strip{grid-template-columns:repeat(2,minmax(0,1fr))}"
        ".report-grid{grid-template-columns:1fr}.report-outline{position:static}}"
        "@media (max-width:720px){body{padding:18px}.report-header{padding:18px}"
        ".report-header h1{font-size:1.7rem}.summary-strip{grid-template-columns:1fr}"
        ".report-body{padding:18px}th,td{padding:7px 8px}}"
        "@media print{body{background:#fff;padding:0}.report-header,.summary-chip,.report-outline,.report-body{"
        "box-shadow:none;border-color:#d0d7de}.report-grid{grid-template-columns:1fr}"
        ".report-outline{display:none}.report-body{border-radius:0;padding:0}a{color:inherit;text-decoration:none}"
        ".badge{color:#000;border:1px solid #000;background:transparent}}"
        "</style>"
        '</head><body><div class="report-shell">'
        '<header class="report-header">'
        '<div class="brand-lockup"><span class="brand-mark" aria-hidden="true">IL</span>'
        '<p class="eyebrow">InvarLock</p></div>'
        "<h1>Evaluation Report</h1>"
        f"<p>{escape(BRAND_TAGLINE)} Browser-first rendering of the canonical evaluation bundle, with quick links for faster reviewer navigation.</p>"
        "</header>"
        f"{summary_strip}"
        '<section class="report-grid">'
        f"{outline}"
        f"{body}"
        "</section>"
        "</div></body></html>"
    )
    return shell


__all__ = ["render_report_html"]
