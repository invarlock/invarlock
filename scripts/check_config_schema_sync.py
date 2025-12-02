#!/usr/bin/env python3
"""Basic sanity check to ensure config schema fragments are present in docs."""

from __future__ import annotations

import sys
from pathlib import Path

EXPECTED_KEYS = {
    "model:",
    "dataset:",
    "edit:",
    "auto:",
    "guards:",
    "eval:",
    "output:",
}


def doc_contains_keys(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return all(key in text for key in EXPECTED_KEYS)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "README.md",
        repo_root / "docs" / "README.md",
    ]

    for candidate in candidates:
        if doc_contains_keys(candidate):
            print(
                f"Configuration schema documented in {candidate.relative_to(repo_root)}"
            )
            return 0

    print("Configuration schema snippets not found in documentation.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
