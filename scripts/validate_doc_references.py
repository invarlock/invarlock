#!/usr/bin/env python3
"""
Validate documentation references.

Ensures that relative links inside the documentation tree point to existing
files. External links (http/https/mailto) are ignored so we do not trigger
network requests during CI.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RE_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:")


def iter_markdown_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.md") if p.is_file())


def resolve_relative_link(markdown_file: Path, link: str, docs_root: Path) -> Path:
    # Strip fragments
    target = link.split("#", 1)[0].strip()
    if not target:
        return Path()

    if target.startswith("/"):
        return (docs_root / target.lstrip("/")).resolve()

    return (markdown_file.parent / target).resolve()


def validate_file(markdown_file: Path, docs_root: Path) -> list[str]:
    errors: list[str] = []
    text = markdown_file.read_text(encoding="utf-8")

    for match in RE_LINK.finditer(text):
        raw_link = match.group(1).strip()
        if (
            not raw_link
            or raw_link.startswith(EXTERNAL_PREFIXES)
            or raw_link.startswith("#")
        ):
            continue

        candidate = resolve_relative_link(markdown_file, raw_link, docs_root)
        if not candidate:
            continue

        # Allow directory references (e.g., pointing to a folder)
        if candidate.exists():
            continue

        # Also allow implicit .md extension
        if candidate.suffix == "" and candidate.with_suffix(".md").exists():
            continue

        errors.append(f"{markdown_file.relative_to(docs_root)} -> {raw_link}")

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    docs_root = repo_root / "docs"

    if not docs_root.exists():
        print("Docs directory not found; skipping validation.")
        return 0

    all_errors: list[str] = []

    for md_file in iter_markdown_files(docs_root):
        all_errors.extend(validate_file(md_file, docs_root))

    if all_errors:
        print("Broken documentation links detected:")
        for item in all_errors:
            print(f"  - {item}")
        return 1

    print("All documentation references resolved successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
