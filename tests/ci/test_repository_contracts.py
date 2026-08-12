from __future__ import annotations

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
