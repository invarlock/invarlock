#!/usr/bin/env python3
"""Ensure Python snippets bundled with the documentation are syntactically valid."""

from __future__ import annotations

import sys
from pathlib import Path


def iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    docs_root = repo_root / "docs"

    if not docs_root.exists():
        print("Docs directory not found; skipping validation.")
        return 0

    python_files = iter_python_files(docs_root)
    if not python_files:
        print("No Python documentation examples found.")
        return 0

    errors: list[str] = []
    for path in python_files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(repo_root)}: {exc}")

    if errors:
        print("Python example validation failed:")
        for item in errors:
            print(f"  - {item}")
        return 1

    print(f"Validated {len(python_files)} Python example(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
