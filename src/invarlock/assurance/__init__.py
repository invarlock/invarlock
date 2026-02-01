"""Assurance namespace (`invarlock.assurance`)."""

from __future__ import annotations

from typing import Any

from invarlock.reporting.report_types import RunReport

try:  # pragma: no cover - shim to reporting modules
    from invarlock.reporting.certificate import (
        REPORT_SCHEMA_VERSION,
        make_certificate,
        validate_certificate,
    )

    # Prefer direct import from render for rendering APIs
    from invarlock.reporting.render import render_certificate_markdown
except Exception:  # pragma: no cover - provide soft stubs
    REPORT_SCHEMA_VERSION = "v1"

    def make_certificate(
        report: RunReport,
        baseline: RunReport | dict[str, Any],
    ) -> dict[str, Any]:
        raise ImportError("invarlock.reporting.certificate not available")

    def render_certificate_markdown(certificate: dict[str, Any]) -> str:
        raise ImportError("invarlock.reporting.certificate not available")

    def validate_certificate(certificate: dict[str, Any]) -> bool:
        raise ImportError("invarlock.reporting.certificate not available")


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "make_certificate",
    "render_certificate_markdown",
    "validate_certificate",
]
