from __future__ import annotations

import re
from pathlib import Path


def test_assurance_docs_are_numbered_except_glossary() -> None:
    assurance_docs = sorted(Path("docs/assurance").glob("*.md"))
    offenders = [
        path.name
        for path in assurance_docs
        if path.name != "glossary.md" and re.match(r"^\d{2}-", path.name) is None
    ]

    assert offenders == []
