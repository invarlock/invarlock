"""Shared presentation branding for human-readable report renderers."""

from __future__ import annotations

from invarlock import __version__
from invarlock.public_contracts import REPORT_SCHEMA_VERSION

BRAND_NAME = "InvarLock"
BRAND_TAGLINE = "Auditable verification for edited model checkpoints."


def html_brand_mark(*, class_name: str = "brand-mark-svg") -> str:
    """Return the self-contained SVG mark used in exported HTML reports."""

    return (
        f'<svg class="{class_name}" viewBox="0 0 512 512" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">'
        '<path d="M174 150 H126 V362 H174" fill="none" stroke="currentColor" '
        'stroke-width="34" stroke-linecap="round" stroke-linejoin="round" '
        'opacity="0.72"/>'
        '<path d="M338 150 H386 V362 H338" fill="none" stroke="currentColor" '
        'stroke-width="34" stroke-linecap="round" stroke-linejoin="round" '
        'opacity="0.72"/>'
        '<path d="M156 292 C 198 212, 230 212, 256 252 S 314 332, 356 242" '
        'fill="none" stroke="var(--brand-mark-accent,currentColor)" '
        'stroke-width="32" stroke-linecap="round" opacity="0.62"/>'
        '<path d="M156 292 C 198 212, 230 212, 256 252 S 314 332, 356 242" '
        'fill="none" stroke="currentColor" stroke-width="14" '
        'stroke-linecap="round" opacity="0.48"/>'
        '<circle cx="256" cy="252" r="9" fill="currentColor" opacity="0.62"/>'
        "</svg>"
    )


def version_label() -> str:
    """Return the package version label used in human-readable reports."""

    return __version__ if isinstance(__version__, str) and __version__ else "unknown"


def markdown_brand_line(*, schema_version: str | None = None) -> str:
    """Return a compact Markdown brand/version line."""

    schema = schema_version or REPORT_SCHEMA_VERSION
    return f"*{BRAND_NAME} {version_label()} · schema {schema} · {BRAND_TAGLINE}*"


__all__ = [
    "BRAND_NAME",
    "BRAND_TAGLINE",
    "html_brand_mark",
    "markdown_brand_line",
    "version_label",
]
