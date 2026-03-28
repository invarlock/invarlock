from __future__ import annotations

import importlib.util

from invarlock.reporting.render import render_report_markdown
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_schema import REPORT_SCHEMA_VERSION, validate_report


def test_canonical_reporting_imports_are_available() -> None:
    assert isinstance(REPORT_SCHEMA_VERSION, str)
    assert callable(make_report)
    assert callable(validate_report)
    assert callable(render_report_markdown)


def test_removed_reporting_facades_are_not_importable() -> None:
    assert importlib.util.find_spec("invarlock.reporting.report_builder") is None
    assert importlib.util.find_spec("invarlock.reporting.report_make_support") is None
