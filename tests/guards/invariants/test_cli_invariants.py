from __future__ import annotations

import os
from pathlib import Path

from tests._repo_root import REPO_ROOT


def _iter_source_files(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith((".py", ".md", ".txt")):
                yield Path(dirpath) / fn


def test_cli_invariants_ban_removed_strings():
    # Restrict scan to code paths only
    src_root = REPO_ROOT / "src"
    assert src_root.exists()

    banned = [
        "explain-gates",  # replaced by: report explain
        "export-html",  # replaced by: report html
        "--source ",  # replaced by: --baseline
        "--edited ",  # replaced by: --subject
    ]

    offenders: list[str] = []
    unreadable: list[str] = []
    for path in _iter_source_files(src_root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            unreadable.append(
                f"{path}: unable to read file ({exc.__class__.__name__}: {exc})"
            )
            continue
        lower = text.lower()
        for needle in banned:
            if needle in lower:
                offenders.append(f"{path}: contains '{needle}'")

    assert not unreadable, "Unreadable CLI source files:\n" + "\n".join(unreadable)
    assert not offenders, "Removed CLI strings found in code:\n" + "\n".join(offenders)
