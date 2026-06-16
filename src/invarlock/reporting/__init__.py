"""
Evaluation report tooling (`invarlock.reporting`).

Provides the evaluation report schema, builder, and renderers.
"""

from __future__ import annotations

from .report_schema import REPORT_JSON_SCHEMA, REPORT_SCHEMA_VERSION, validate_report


def make_report(*args, **kwargs):
    from .report_make import make_report as _make_report

    return _make_report(*args, **kwargs)


def render_report_markdown(*args, **kwargs):
    from .render_markdown import render_report_markdown as _render_report_markdown

    return _render_report_markdown(*args, **kwargs)


def render_report_html(*args, **kwargs):
    from .html import render_report_html as _render_report_html

    return _render_report_html(*args, **kwargs)


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "REPORT_JSON_SCHEMA",
    "make_report",
    "render_report_markdown",
    "render_report_html",
    "validate_report",
]
