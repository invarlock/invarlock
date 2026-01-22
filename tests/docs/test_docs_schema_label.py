from __future__ import annotations

from pathlib import Path


def test_docs_reference_is_schema_v1():
    p = Path("docs/reference/certificates.md")
    text = p.read_text("utf-8")
    assert 'schema_version = "v1"' in text
