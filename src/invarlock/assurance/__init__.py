"""Assurance namespace (`invarlock.assurance`)."""

from __future__ import annotations

from typing import Any

from invarlock.reporting.report_types import RunReport

try:  # pragma: no cover - shim to reporting modules
    from invarlock.reporting.report_builder import make_report
    from invarlock.reporting.report_schema import REPORT_SCHEMA_VERSION, validate_report

    # Prefer direct import from render for rendering APIs
    from invarlock.reporting.render import render_report_markdown
except Exception:  # pragma: no cover - provide soft stubs
    REPORT_SCHEMA_VERSION = "v1"

    def make_report(
        report: RunReport,
        baseline: RunReport | dict[str, Any],
    ) -> dict[str, Any]:
        raise ImportError("invarlock.reporting.report_builder not available")

    def render_report_markdown(evaluation_report: dict[str, Any]) -> str:
        raise ImportError("invarlock.reporting.report_builder not available")

    def validate_report(evaluation_report: dict[str, Any]) -> bool:
        raise ImportError("invarlock.reporting.report_schema not available")


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "make_report",
    "render_report_markdown",
    "validate_report",
]
