from __future__ import annotations

import os
from pathlib import Path


def test_no_render_helpers_imported_from_certificate():
    """Ensure code imports render helpers from invarlock.reporting.render, not certificate.

    This guards future modules from re-introducing imports like:
    from invarlock.reporting.certificate import render_certificate_markdown
    """
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    banned_snippets = [
        "from invarlock.reporting.certificate import render_certificate_markdown",
        "from invarlock.reporting.certificate import compute_console_validation_block",
        "from invarlock.reporting.certificate import _load_console_labels",
        "from invarlock.reporting.certificate import _compute_certificate_hash",
        "from invarlock.reporting.certificate import build_console_summary_pack",
    ]

    offenders: list[str] = []
    for root, _dirs, files in os.walk(src_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = Path(root) / fn
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            for needle in banned_snippets:
                if needle in text:
                    offenders.append(f"{p}: banned import -> {needle}")

    assert not offenders, "\n".join(offenders)
