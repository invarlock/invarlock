from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests._support_repository_contracts import MakefileContract


def _first_party_schema_paths(root: Path) -> list[Path]:
    return [
        *sorted((root / "contracts").glob("*.json")),
        *sorted((root / "examples").rglob("*.schema.json")),
    ]


def test_public_schema_ids_use_the_canonical_contract_namespace() -> None:
    root = Path(__file__).resolve().parents[2]
    schemas = sorted((root / "contracts").glob("*.schema.json"))
    assert schemas
    for path in schemas:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["$id"] == f"https://invarlock.dev/contracts/{path.name}"


def test_printable_identifier_patterns_reject_a_final_control_character() -> None:
    root = Path(__file__).resolve().parents[2]
    checked = 0

    def visit(value: object) -> None:
        nonlocal checked
        if isinstance(value, dict):
            pattern = value.get("pattern")
            if isinstance(pattern, str) and r"\x00-\x1f\x7f" in pattern:
                checked += 1
                assert re.search(pattern, "identifier\n") is None
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in _first_party_schema_paths(root):
        visit(json.loads(path.read_text(encoding="utf-8")))
    assert checked > 0


def test_anchored_contract_patterns_require_the_true_end_of_string() -> None:
    root = Path(__file__).resolve().parents[2]
    checked = 0

    def visit(value: object, path: Path) -> None:
        nonlocal checked
        if isinstance(value, dict):
            pattern = value.get("pattern")
            if isinstance(pattern, str) and pattern.startswith("^"):
                checked += 1
                assert not pattern.endswith("$"), (
                    f"{path} uses a terminal $ anchor that accepts a final newline"
                )
                assert pattern.endswith(r"(?![\s\S])"), (
                    f"{path} does not require the true end of the string"
                )
            for child in value.values():
                visit(child, path)
        elif isinstance(value, list):
            for child in value:
                visit(child, path)

    for path in _first_party_schema_paths(root):
        visit(json.loads(path.read_text(encoding="utf-8")), path)
    assert checked > 0


def test_make_target_parser_keeps_repeated_declarations_and_recipe_colons() -> None:
    makefile = MakefileContract(
        "verify: WORKERS = 2\n"
        "verify: test lint  ## Run checks\n"
        "\tpython -c 'print(\"result: ok\")'\n"
        "verify-fast: test\n"
        "\tpytest -q\n"
    )

    target = makefile.target("verify")

    assert target.declarations == (
        "verify: WORKERS = 2",
        "verify: test lint  ## Run checks",
    )
    assert target.prerequisites == ("test", "lint")
    assert "result: ok" in target.text
    assert "verify-fast" not in target.text


def test_make_target_parser_matches_complete_target_names() -> None:
    makefile = MakefileContract("coverage-fast:\n\ttrue\ncoverage:\n\tpytest\n")

    assert "pytest" in makefile.target("coverage").text
    assert "coverage-fast" not in makefile.target("coverage").text


def test_make_target_parser_rejects_an_absent_target() -> None:
    with pytest.raises(AssertionError, match="Make target 'missing' not found"):
        MakefileContract("test:\n\tpytest\n").target("missing")


def test_coverage_checks_untracked_runtime_files_without_requiring_staging(tmp_path):
    root = Path(__file__).resolve().parents[2]
    makefile = MakefileContract((root / "Makefile").read_text())
    collection = makefile.target("coverage").text
    recipe = makefile.target("coverage-check-files").text
    assert "--cov-fail-under=95" in collection
    assert "$(MAKE) coverage-core-report" in collection
    assert (
        "$(MAKE) coverage-check-files" in makefile.target("coverage-core-report").text
    )
    assert "--fail-under=95" in recipe
    assert "git ls-files" not in recipe
    program = recipe.split("-c '", 1)[1].split("' |", 1)[0]
    source = tmp_path / "src/invarlock"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "existing.py").write_text("pass\n")
    (source / "new_untracked.py").write_text("pass\n")
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        "src/invarlock/existing.py",
        "src/invarlock/new_untracked.py",
    ]
