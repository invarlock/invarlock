#!/usr/bin/env python3
"""Verify that each core guard has a dedicated section in the documentation."""

from __future__ import annotations

import sys

from common_io import read_text, repo_root_from

GUARD_HEADINGS = {
    "### Invariants Guard",
    "### Spectral Guard",
    "### RMT Guard",
    "### Variance Guard",
}


def main() -> int:
    repo_root = repo_root_from(__file__)
    guards_doc = repo_root / "docs" / "reference" / "guards.md"

    if not guards_doc.exists():
        print("Guard reference documentation not found.")
        return 1

    text = read_text(guards_doc)
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
