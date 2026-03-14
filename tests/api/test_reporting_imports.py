from __future__ import annotations

from invarlock.reporting.render import render_report_markdown
from invarlock.reporting.report_builder import make_report
from invarlock.reporting.report_schema import REPORT_SCHEMA_VERSION, validate_report


def test_canonical_reporting_imports_are_available() -> None:
    assert isinstance(REPORT_SCHEMA_VERSION, str)
    assert callable(make_report)
    assert callable(validate_report)
    assert callable(render_report_markdown)
