"""
Evaluation report tooling (`invarlock.reporting`).

Provides the evaluation report schema, builder, and renderers.
"""

from __future__ import annotations

from .html import render_report_html
from .render import render_report_markdown
from .report_builder import make_report, validate_report
from .report_schema import REPORT_JSON_SCHEMA, REPORT_SCHEMA_VERSION

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "REPORT_JSON_SCHEMA",
    "make_report",
    "render_report_markdown",
    "render_report_html",
    "validate_report",
]
