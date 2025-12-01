#!/usr/bin/env python3
"""Ensure documented version numbers match the package version."""

from __future__ import annotations

import re
import sys
from pathlib import Path

VERSION_PATTERN = re.compile(r'__version__\s*=\s*"([^"]+)"')


def get_package_version(repo_root: Path) -> str:
    init_path = repo_root / "src" / "invarlock" / "__init__.py"
    content = init_path.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(content)
    if not match:
        raise RuntimeError(
            "Could not determine package version from src/invarlock/__init__.py"
        )
    return match.group(1)


def check_file_contains(path: Path, version: str) -> bool:
    if not path.exists():
        return False
    return version in path.read_text(encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    version = get_package_version(repo_root)

    targets = [
        repo_root / "README.md",
        repo_root / "docs" / "README.md",
    ]

    missing: list[str] = []
    for target in targets:
        if not check_file_contains(target, version):
            missing.append(str(target.relative_to(repo_root)))

    if missing:
        print(f"Version {version} not found in:")
        for item in missing:
            print(f"  - {item}")
        return 1

    print(f"Documentation version strings match package version {version}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
