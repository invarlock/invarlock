"""
Minimal HTML exporter for reports.

This implementation wraps the Markdown rendering in a simple HTML template so
that the numbers and core content remain identical across formats.
"""

from __future__ import annotations

from html import escape
from importlib import import_module
from typing import Any

from .render import render_report_markdown
from .report_schema import validate_report

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


def _apply_status_badges(html_body: str) -> str:
    updated = html_body
    for token, replacement in _STATUS_BADGES.items():
        updated = updated.replace(token, replacement)
    return updated


def render_report_html(evaluation_report: dict[str, Any]) -> str:
    """Render an evaluation report as a simple HTML document.

    Uses the Markdown renderer and converts to HTML when available, falling back
    to a <pre> block when the markdown dependency is missing.
    """
    if not validate_report(evaluation_report):
        raise ValueError("Invalid evaluation report structure")
    md = render_report_markdown(evaluation_report)
    if markdown_module is None:
        body = f'<pre class="invarlock-md">{escape(md)}</pre>'
    else:
        html_body = markdown_module.markdown(md, extensions=["tables", "fenced_code"])
        html_body = _apply_status_badges(html_body)
        body = f'<div class="invarlock-md">{html_body}</div>'
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>InvarLock Evaluation Report</title>"
        "<style>"
        ":root{--pass:#1f7a46;--fail:#b42318;--warn:#a15c07;--ink:#17212b;"
        "--muted:#52606d;--bg:#eef3f8;--bg-accent:#dfe9f3;--panel:#f7fafc;"
        "--panel-strong:#ffffff;--border:#cbd5e1;--shadow:rgba(15,23,42,0.12);"
        "--accent:#0f5f8c}"
        "@media (prefers-color-scheme: dark){"
        ":root{--pass:#3fb36b;--fail:#ff7b72;--warn:#f2b44f;--ink:#e6edf3;"
        "--muted:#9fb0c0;--bg:#0f1722;--bg-accent:#162334;--panel:#111c2a;"
        "--panel-strong:#162334;--border:#2c3e50;--shadow:rgba(0,0,0,0.35);"
        "--accent:#7cc7ff}}"
        "body{margin:0;min-height:100vh;padding:32px;color:var(--ink);"
        'font-family:"Avenir Next","Segoe UI Variable","Segoe UI",sans-serif;'
        "line-height:1.65;background:"
        "radial-gradient(circle at top right,var(--bg-accent),transparent 28%),"
        "linear-gradient(180deg,var(--bg),var(--panel))}"
        ".report-shell{max-width:1080px;margin:0 auto}"
        ".report-header{margin:0 auto 20px;padding:22px 24px;border:1px solid var(--border);"
        "border-radius:20px;background:linear-gradient(135deg,var(--panel-strong),var(--panel));"
        "box-shadow:0 18px 48px var(--shadow)}"
        ".eyebrow{margin:0 0 8px 0;font-size:0.78rem;font-weight:700;letter-spacing:0.12em;"
        "text-transform:uppercase;color:var(--accent)}"
        ".report-header h1{margin:0;font-size:2rem;line-height:1.1;"
        'font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif}'
        ".report-header p{margin:10px 0 0 0;max-width:48rem;color:var(--muted)}"
        ".report-card{padding:28px;background:var(--panel-strong);border:1px solid var(--border);"
        "border-radius:24px;box-shadow:0 18px 48px var(--shadow)}"
        ".invarlock-md{max-width:960px;margin:0 auto}"
        ".invarlock-md>:first-child{margin-top:0}"
        'h1,h2,h3{line-height:1.2;font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif}'
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
        "@media (max-width:720px){body{padding:18px}.report-header{padding:18px}"
        ".report-header h1{font-size:1.7rem}.report-card{padding:18px}th,td{padding:7px 8px}}"
        "@media print{body{background:#fff;padding:0}.report-header,.report-card{box-shadow:none;"
        "border:0;border-radius:0;padding:0}.report-header{margin-bottom:16px}"
        ".invarlock-md{max-width:none}a{color:inherit;text-decoration:none}"
        ".badge{color:#000;border:1px solid #000;background:transparent}}"
        "</style>"
        '</head><body><div class="report-shell">'
        '<header class="report-header">'
        '<p class="eyebrow">InvarLock</p>'
        "<h1>Evaluation Report</h1>"
        "<p>Readable HTML rendering of the canonical evaluation report bundle.</p>"
        '</header><main class="report-card">' + body + "</main></div></body></html>"
    )


__all__ = ["render_report_html"]
