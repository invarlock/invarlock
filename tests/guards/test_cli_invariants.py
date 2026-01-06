from __future__ import annotations

import os
from pathlib import Path


def _iter_source_files(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith((".py", ".md", ".txt")):
                yield Path(dirpath) / fn


def test_cli_invariants_ban_removed_strings():
    # Restrict scan to code paths only
    src_root = Path(__file__).resolve().parents[2] / "src"
    assert src_root.exists()

    banned = [
        "explain-gates",  # replaced by: report explain
        "export-html",  # replaced by: report html
        "--source ",  # replaced by: --baseline
        "--edited ",  # replaced by: --subject
    ]

    offenders: list[str] = []
    for path in _iter_source_files(src_root):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        lower = text.lower()
        for needle in banned:
            if needle in lower:
                offenders.append(f"{path}: contains '{needle}'")

    assert not offenders, "Removed CLI strings found in code:\n" + "\n".join(offenders)
