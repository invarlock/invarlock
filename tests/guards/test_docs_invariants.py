from __future__ import annotations

import os
from pathlib import Path


def _iter_doc_files(root: Path):
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            if fn.endswith((".md", ".mdx", ".txt")):
                yield Path(dp) / fn


def test_docs_invariants_pm_first_strings():
    docs_root = Path(__file__).resolve().parents[2] / "docs" / "reference"
    assert docs_root.exists()

    banned = [
        "PPL Ratio",
        "PPL vs Baseline",
        "PPL\n",
        "explain-gates",
        "export-html",
        "--source ",
        "--edited ",
        "cert-v2",
        "cert-v3",
    ]

    offenders: list[str] = []
    skip_files = {
        "cli.md",
        "exporting-certificates-html.md",
        "certificate-schema-v2.md",
        "certificate-schema-v3.md",
        "certificate_telemetry.md",
    }
    for path in _iter_doc_files(docs_root):
        if path.name in skip_files:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        # Only scan reference pages; allow prose mentions of 'perplexity' (lowercase)
        lower = text  # case-sensitive banned checks
        for needle in banned:
            if needle in lower:
                offenders.append(f"{path}: contains '{needle}'")

    assert not offenders, "Legacy UI strings found in docs:\n" + "\n".join(offenders)
