"""
Minimal HTML exporter for certificates.

This implementation wraps the Markdown rendering in a simple HTML template so
that the numbers and core content remain identical across formats.
"""

from __future__ import annotations

from html import escape
from typing import Any

from .render import render_certificate_markdown


def render_certificate_html(certificate: dict[str, Any]) -> str:
    """Render a certificate as a simple HTML document.

    Uses the Markdown renderer and embeds the content in a <pre> block to ensure
    stable parity for snapshot tests without extra dependencies.
    """
    md = render_certificate_markdown(certificate)
    body = f'<pre class="invarlock-md">{escape(md)}</pre>'
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>InvarLock Safety Certificate</title>"
        "<style>body{font-family:ui-monospace,Menlo,monospace;white-space:pre-wrap}</style>"
        "</head><body>" + body + "</body></html>"
    )


__all__ = ["render_certificate_html"]
