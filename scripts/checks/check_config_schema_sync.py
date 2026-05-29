#!/usr/bin/env python3
"""Basic sanity check to ensure config schema fragments are present in docs."""

from __future__ import annotations

import sys
from pathlib import Path

from common_io import path_contains_all, repo_root_from

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
    return path_contains_all(path, EXPECTED_KEYS)


def main() -> int:
    repo_root = repo_root_from(__file__)
    candidates = [
        repo_root / "docs" / "reference" / "config-schema.md",
        repo_root / "docs" / "README.md",
        repo_root / "README.md",
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
