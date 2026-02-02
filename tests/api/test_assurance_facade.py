import importlib


def test_invarlock_assurance_imports_and_exports():
    module = importlib.import_module("invarlock.assurance")

    # Symbols should exist on the facade
    assert hasattr(module, "REPORT_SCHEMA_VERSION")
    assert hasattr(module, "make_report")
    assert hasattr(module, "validate_report")
    assert hasattr(module, "render_report_markdown")

    # Types of exported items
    assert isinstance(module.REPORT_SCHEMA_VERSION, str)
    assert callable(module.make_report)
    assert callable(module.validate_report)
    assert callable(module.render_report_markdown)

    # Direct import form should also work
    from invarlock.assurance import (  # noqa: F401
        REPORT_SCHEMA_VERSION,
        make_report,
        render_report_markdown,
        validate_report,
    )
