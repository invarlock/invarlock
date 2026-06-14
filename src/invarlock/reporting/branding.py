"""Shared presentation branding for human-readable report renderers."""

from __future__ import annotations

from invarlock import __version__
from invarlock.public_contracts import REPORT_SCHEMA_VERSION

BRAND_NAME = "InvarLock"
BRAND_TAGLINE = "Auditable verification for edited model checkpoints."


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
    "markdown_brand_line",
    "version_label",
]
