#!/usr/bin/env python3
"""Ensure documented version numbers match the package version.

We intentionally avoid hardcoding the version in end-user docs like README.md
and docs/README.md to prevent drift (the repo already publishes version badges
and exposes `invarlock --version`). Instead, we validate version consistency
across canonical metadata files used for packaging/citation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

VERSION_PATTERN = re.compile(r'__version__\s*=\s*"([^"]+)"')
PYPROJECT_VERSION_PATTERN = re.compile(r'^\s*version\s*=\s*"([^"]+)"\s*$', re.M)
CITATION_VERSION_PATTERN = re.compile(
    r"^\s*version:\s*([0-9]+\.[0-9]+\.[0-9]+)\s*$", re.M
)


def get_package_version(repo_root: Path) -> str:
    init_path = repo_root / "src" / "invarlock" / "__init__.py"
    content = init_path.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(content)
    if not match:
        raise RuntimeError(
            "Could not determine package version from src/invarlock/__init__.py"
        )
    return match.group(1)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    version = get_package_version(repo_root)

    pyproject = repo_root / "pyproject.toml"
    citation_cff = repo_root / "CITATION.cff"

    missing: list[str] = []
    pyproject_text = pyproject.read_text(encoding="utf-8")
    pyproject_match = PYPROJECT_VERSION_PATTERN.search(pyproject_text)
    if not pyproject_match or pyproject_match.group(1) != version:
        missing.append(str(pyproject.relative_to(repo_root)))

    citation_text = citation_cff.read_text(encoding="utf-8")
    citation_match = CITATION_VERSION_PATTERN.search(citation_text)
    if not citation_match or citation_match.group(1) != version:
        missing.append(str(citation_cff.relative_to(repo_root)))

    if missing:
        print(f"Version {version} does not match in:")
        for item in missing:
            print(f"  - {item}")
        return 1

    print(f"Metadata version strings match package version {version}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
