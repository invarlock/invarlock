from __future__ import annotations

import ast
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


def _make_pytest_selectors() -> set[str]:
    text = MAKEFILE.read_text(encoding="utf-8")
    return set(
        re.findall(
            r"tests/[A-Za-z0-9_./-]+\.py(?:::[A-Za-z_][A-Za-z0-9_]*)?",
            text,
        )
    )


def test_make_test_dir_targets_cover_executable_tree() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(r"^TEST_DIR_TARGETS := (.+)$", text, flags=re.MULTILINE)
    assert match is not None
    actual = set(match.group(1).split())
    assert actual == EXPECTED_TEST_DIR_TARGETS

    grouped_targets = actual - {"integration"}
    missing_assignments = sorted(
        target
        for target in grouped_targets
        if f"test-{target}: TEST_DIR = {target}" not in text
    )
    missing_help = sorted(
        target
        for target in grouped_targets
        if f"test-{target}: ## Run tests/{target}" not in text
    )
    assert missing_assignments == []
    assert missing_help == []


def test_makefile_declares_runtime_and_reporting_targets() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    for target in ("test-runtime", "test-reporting"):
        assert re.search(rf"^{target}:", text, flags=re.MULTILINE), target


def test_makefile_pytest_file_and_node_selectors_resolve() -> None:
    missing_files: list[str] = []
    missing_nodes: list[str] = []

    for selector in sorted(_make_pytest_selectors()):
        relative_path, _, node_name = selector.partition("::")
        path = REPO_ROOT / relative_path
        if not path.is_file():
            missing_files.append(relative_path)
            continue
        if not node_name:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        if node_name not in function_names:
            missing_nodes.append(selector)

    assert missing_files == []
    assert missing_nodes == []
