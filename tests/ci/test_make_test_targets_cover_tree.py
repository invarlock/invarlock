from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"

EXPECTED_TEST_DIR_TARGETS = {
    "adapters",
    "calibration",
    "ci",
    "cli",
    "core",
    "docs",
    "edits",
    "eval",
    "fuzzing",
    "guards",
    "integration",
    "lint",
    "observability",
    "plugins",
    "evidence_packs",
    "reporting",
    "runtime",
    "scripts",
}


def test_make_test_dir_targets_cover_executable_tree() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(r"^TEST_DIR_TARGETS := (.+)$", text, flags=re.MULTILINE)
    assert match is not None
    actual = set(match.group(1).split())
    assert actual == EXPECTED_TEST_DIR_TARGETS


def test_makefile_declares_runtime_and_reporting_targets() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    for target in ("test-runtime", "test-reporting"):
        assert re.search(rf"^{target}:", text, flags=re.MULTILINE), target
