from __future__ import annotations

from pathlib import Path


def test_docs_reference_is_schema_v1():
    p = Path("docs/reference/certificate-schema.md")
    text = p.read_text("utf-8")
    assert "Certificate Schema (v1)" in text
