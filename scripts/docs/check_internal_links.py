#!/usr/bin/env python3
"""Check that relative markdown links resolve within a single file."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ANCHOR_PATTERN = re.compile(r"\[[^\]]+\]\((#[^)]+)\)")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_internal_links.py <markdown-file>")
        return 1

    md_path = Path(sys.argv[1])
    if not md_path.exists():
        print(f"File not found: {md_path}")
        return 1

    text = md_path.read_text(encoding="utf-8")

    def slugify(s: str) -> str:
        s = s.lower()
        # Keep alphanumerics, spaces and hyphens; drop punctuation like ()[]{}
        import re as _re

        s = _re.sub(r"[^a-z0-9\-\s]", "", s)
        s = _re.sub(r"\s+", "-", s.strip())
        s = _re.sub(r"-+", "-", s)
        return s

    anchors = {
        slugify(match) for match in re.findall(r"^#+\s*(.*)", text, flags=re.MULTILINE)
    }

    missing: list[str] = []
    for match in ANCHOR_PATTERN.finditer(text):
        anchor = match.group(1)[1:].strip()
        slug = slugify(anchor)
        if slug not in anchors:
            missing.append(slug)

    if missing:
        print(f"{md_path}: missing anchors -> {missing}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
