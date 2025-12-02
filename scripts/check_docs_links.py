#!/usr/bin/env python3
"""Static checker for Markdown links in docs/.

Verifies that relative links resolve to existing files (ignoring anchors) and
that code snippets referenced via pymdownx.snippets are present.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS_ROOT = Path("docs")
LINK_RE = re.compile(r"\]\(([^)]+)\)")
SNIPPET_RE = re.compile(r"--8<--\s*\"([^\"]+)\"")


def _is_external(link: str) -> bool:
    lowered = link.lower()
    return lowered.startswith(("http://", "https://", "mailto:", "tel:"))


def _check_links() -> list[str]:
    missing: list[str] = []
    for path in DOCS_ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for snippet in SNIPPET_RE.findall(text):
            resolved = (DOCS_ROOT / snippet).resolve()
            if not resolved.exists():
                missing.append(f"{path}: missing snippet -> {snippet}")
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip()
            if not target or target.startswith("#") or _is_external(target):
                continue
            if target.startswith("!include"):
                continue
            # Handle snippet include syntax: --8<-- "path"
            if target.startswith("--8<--"):
                continue
            target_path = target.split("#", 1)[0]
            resolved = (path.parent / target_path).resolve()
            if not resolved.exists():
                missing.append(f"{path}: broken link -> {target}")
    return missing


def main() -> None:
    missing = _check_links()
    if missing:
        for entry in missing:
            print(entry)
        raise SystemExit(f"Found {len(missing)} broken documentation links")
    print("Documentation links valid")


if __name__ == "__main__":
    main()
