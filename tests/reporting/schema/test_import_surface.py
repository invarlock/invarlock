from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys

from invarlock.reporting.rendering.markdown import render_report_markdown
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
    assert importlib.util.find_spec("invarlock.reporting.verify_checks") is None
    assert importlib.util.find_spec("invarlock.reporting.render") is None
    assert importlib.util.find_spec("invarlock.reporting.report_files") is None


def test_reporting_package_root_import_is_light_and_source_compatible(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("INVARLOCK_LIGHT_IMPORT", raising=False)

    for mod in [
        "invarlock.reporting",
        "invarlock.reporting.html",
        "invarlock.reporting.rendering.markdown",
        "invarlock.reporting.report_make",
    ]:
        sys.modules.pop(mod, None)

    mod = importlib.import_module("invarlock.reporting")

    assert hasattr(mod, "REPORT_SCHEMA_VERSION")
    assert callable(mod.make_report)
    assert callable(mod.render_report_markdown)
    assert callable(mod.render_report_html)
    assert callable(mod.validate_report)

    from invarlock.reporting import (  # noqa: PLC0415
        make_report,
        render_report_html,
        render_report_markdown,
        validate_report,
    )

    assert make_report is mod.make_report
    assert render_report_markdown is mod.render_report_markdown
    assert render_report_html is mod.render_report_html
    assert validate_report is mod.validate_report
