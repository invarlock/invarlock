from __future__ import annotations

import re
from pathlib import Path


def _get_make_target_block(text: str, target: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(target)}\s*:\s*(?:##.*)?$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return None
    start = m.end()
    # Collect subsequent tab-indented recipe lines until next target or EOF
    lines = []
    for line in text[start:].splitlines():
        if not line:
            # keep blank lines within recipe
            lines.append(line)
            continue
        if re.match(r"^[A-Za-z0-9_.-]+\s*:\s*", line):
            break
        lines.append(line)
    return "\n".join(lines)


def test_verify_target_runs_docs_api_refs_check() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    data = makefile.read_text(encoding="utf-8")
    block = _get_make_target_block(data, "verify")
    assert block is not None, "verify target not found in Makefile"
    # Either always run or behind an env flag is acceptable; require presence
    # of the script path within the verify recipe body.
    assert "scripts/validate_docs_api_refs.py" in block, (
        "verify target should include docs API refs validation (optionally gated)"
    )
