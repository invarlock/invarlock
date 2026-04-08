from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "invarlock"


def _count_pattern(pattern: str) -> int:
    total = 0
    compiled = re.compile(pattern, flags=re.MULTILINE)
    for path in SRC_ROOT.rglob("*.py"):
        total += len(compiled.findall(path.read_text(encoding="utf-8")))
    return total


def test_except_exception_pass_budget_does_not_regress() -> None:
    count = _count_pattern(r"except\s+Exception(?:\s+as\s+\w+)?:\r?\n\s+pass")
    assert count <= 426, (
        f"broad-catch pass count regressed to {count} (budget 426)"
    )
