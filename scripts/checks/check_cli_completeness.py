#!/usr/bin/env python3
"""Check that core CLI commands are documented."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from common_io import read_text, repo_root_from

COMMANDS = {
    "invarlock evaluate",
    "invarlock report",
    "invarlock verify",
    "invarlock doctor",
    "invarlock advanced",
}


def gather_documented_commands(doc_root: Path) -> set[str]:
    documented: set[str] = set()
    for md_file in doc_root.rglob("*.md"):
        text = read_text(md_file)
        for command in COMMANDS:
            if re.search(rf"\b{re.escape(command)}\b", text):
                documented.add(command)
    return documented


def main() -> int:
    repo_root = repo_root_from(__file__)
    docs_root = repo_root / "docs"
    if not docs_root.exists():
        print("Docs directory not found; skipping CLI completeness check.")
        return 0

    documented = gather_documented_commands(docs_root)
    missing = COMMANDS.difference(documented)

    if missing:
        print("The following CLI commands are not documented:")
        for command in sorted(missing):
            print(f"  - {command}")
        return 1

    print("All core CLI commands are documented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
