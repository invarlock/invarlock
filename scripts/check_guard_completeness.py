#!/usr/bin/env python3
"""Verify that each core guard has a dedicated section in the documentation."""

from __future__ import annotations

import sys
from pathlib import Path

GUARD_HEADINGS = {
    "### Invariants Guard",
    "### Spectral Guard",
    "### RMT Guard",
    "### Variance Guard",
}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    guards_doc = repo_root / "docs" / "reference" / "guards.md"

    if not guards_doc.exists():
        print("Guard reference documentation not found.")
        return 1

    text = guards_doc.read_text(encoding="utf-8")
    missing = [heading for heading in GUARD_HEADINGS if heading not in text]

    if missing:
        print("Missing guard sections in docs/reference/guards.md:")
        for heading in missing:
            print(f"  - {heading}")
        return 1

    print("All core guard sections are present in docs/reference/guards.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
