#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _load_terms(glossary_path: Path) -> list[str]:
    terms: list[str] = []
    for line in glossary_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            term = line[4:].strip()
            if term:
                terms.append(term)
    return terms


def _count_term(text: str, term: str) -> int:
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    return len(pattern.findall(text))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest glossary link targets by scanning docs for glossary terms."
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=Path("docs"),
        help="Docs root to scan (default: docs)",
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        default=Path("docs/assurance/glossary.md"),
        help="Glossary markdown file (default: docs/assurance/glossary.md)",
    )
    args = parser.parse_args()

    glossary = args.glossary
    docs_root = args.docs
    if not glossary.exists():
        raise SystemExit(f"Glossary not found: {glossary}")
    if not docs_root.exists():
        raise SystemExit(f"Docs root not found: {docs_root}")

    terms = _load_terms(glossary)
    if not terms:
        raise SystemExit("No glossary terms found.")

    for md_file in sorted(docs_root.rglob("*.md")):
        if md_file.resolve() == glossary.resolve():
            continue
        text = md_file.read_text(encoding="utf-8")
        hits = []
        for term in terms:
            count = _count_term(text, term)
            if count:
                hits.append(f"{term} ({count})")
        if hits:
            joined = ", ".join(hits)
            print(f"{md_file}: {joined}")


if __name__ == "__main__":
    main()
