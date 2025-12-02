#!/usr/bin/env python3
"""Check that core CLI commands are documented."""

from __future__ import annotations

import re
import sys
from pathlib import Path

COMMANDS = {
    "invarlock run",
    "invarlock report",
    "invarlock plugins",
    "invarlock doctor",
}


def gather_documented_commands(doc_root: Path) -> set[str]:
    documented: set[str] = set()
    for md_file in doc_root.rglob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        for command in COMMANDS:
            if re.search(rf"\b{re.escape(command)}\b", text):
                documented.add(command)
    return documented


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
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
