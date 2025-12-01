#!/usr/bin/env python3
"""Validate YAML example files referenced in the documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def iter_yaml_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.suffix in {".yaml", ".yml"} and p.is_file()
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    docs_root = repo_root / "docs"

    if not docs_root.exists():
        print("Docs directory not found; nothing to validate.")
        return 0

    yaml_files = iter_yaml_files(docs_root)
    if not yaml_files:
        print("No YAML documentation examples found.")
        return 0

    errors: list[str] = []
    for path in yaml_files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            errors.append(f"{path.relative_to(repo_root)}: {exc}")

    if errors:
        print("Invalid YAML examples detected:")
        for item in errors:
            print(f"  - {item}")
        return 1

    print(f"Validated {len(yaml_files)} YAML example(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
