from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests._support_repository_contracts import MakefileContract


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
    assert "$(MAKE) coverage-check-files" in collection
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
