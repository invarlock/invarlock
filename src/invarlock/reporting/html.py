"""
Minimal HTML exporter for certificates.

This implementation wraps the Markdown rendering in a simple HTML template so
that the numbers and core content remain identical across formats.
"""

from __future__ import annotations

from html import escape
from typing import Any

from .render import render_certificate_markdown

markdown_module: Any | None = None
try:
    import markdown as _markdown  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    _markdown = None
else:
    markdown_module = _markdown


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


def render_certificate_html(certificate: dict[str, Any]) -> str:
    """Render a certificate as a simple HTML document.

    Uses the Markdown renderer and converts to HTML when available, falling back
    to a <pre> block when the markdown dependency is missing.
    """
    md = render_certificate_markdown(certificate)
    if markdown_module is None:
        body = f'<pre class="invarlock-md">{escape(md)}</pre>'
    else:
        html_body = markdown_module.markdown(md, extensions=["tables", "fenced_code"])
        html_body = _apply_status_badges(html_body)
        body = f'<div class="invarlock-md">{html_body}</div>'
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>InvarLock Evaluation Certificate</title>"
        "<style>"
        ":root{--pass:#2da44e;--fail:#cf222e;--warn:#bf8700;--ink:#1f2328;"
        "--muted:#57606a;--panel:#f6f8fa;--border:#d0d7de}"
        "body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;"
        "color:var(--ink);background:linear-gradient(180deg,#fff, #f6f8fa);"
        "margin:0;padding:32px}"
        ".invarlock-md{max-width:960px;margin:0 auto;padding:24px;background:#fff;"
        "border:1px solid var(--border);border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,0.05)}"
        "h1,h2,h3{margin-top:1.4em}h1{margin-top:0}"
        "table{border-collapse:collapse;width:100%;margin:12px 0}"
        "th,td{border:1px solid var(--border);padding:6px 8px;text-align:left}"
        "code,pre{background:var(--panel);border-radius:8px}"
        "pre{padding:12px;overflow:auto}"
        ".badge{display:inline-block;padding:2px 8px;border-radius:999px;"
        "font-size:0.75rem;font-weight:700;letter-spacing:0.02em;color:#fff}"
        ".badge.pass{background:var(--pass)}"
        ".badge.fail{background:var(--fail)}"
        ".badge.warn{background:var(--warn)}"
        "@media print{body{background:#fff;padding:0}.invarlock-md{box-shadow:none;"
        "border:0}a{color:inherit;text-decoration:none}.badge{color:#000;"
        "border:1px solid #000;background:transparent}}"
        "</style>"
        "</head><body>" + body + "</body></html>"
    )


__all__ = ["render_certificate_html"]
