import importlib


def test_invarlock_assurance_imports_and_exports():
    module = importlib.import_module("invarlock.assurance")

    # Symbols should exist on the facade
    assert hasattr(module, "CERTIFICATE_SCHEMA_VERSION")
    assert hasattr(module, "make_certificate")
    assert hasattr(module, "validate_certificate")
    assert hasattr(module, "render_certificate_markdown")

    # Types of exported items
    assert isinstance(module.CERTIFICATE_SCHEMA_VERSION, str)
    assert callable(module.make_certificate)
    assert callable(module.validate_certificate)
    assert callable(module.render_certificate_markdown)

    # Direct import form should also work
    from invarlock.assurance import (  # noqa: F401
        CERTIFICATE_SCHEMA_VERSION,
        make_certificate,
        render_certificate_markdown,
        validate_certificate,
    )
