from __future__ import annotations

from pathlib import Path

from tests._support_repository_contracts import MakefileContract

MAKE = MakefileContract.read(Path(__file__).resolve().parents[2] / "Makefile")
MAKEFILE = MAKE.text


def test_verify_composes_the_maintained_gates() -> None:
    block = MAKE.target("verify").text
    for target in (
        "repo-cruft-check",
        "public-evidence-audit",
        "examples-check",
        "contracts-check",
        "test",
        "cli-smoke-core",
        "lint",
        "docs-check-build",
    ):
        assert target in block

    assert "$(MAKE) -j $(VERIFY_TARGET_JOBS)" in block
    assert "$(MAKE) examples-check PYTEST_WORKERS=$(PYTEST_WORKERS)" in block


def test_verify_fast_includes_the_example_contract() -> None:
    block = MAKE.target("verify-fast").text
    assert "$(MAKE) -j $(VERIFY_TARGET_JOBS)" in block
    assert "$(MAKE) examples-check PYTEST_WORKERS=$(PYTEST_WORKERS)" in block


def test_docs_are_checked_by_established_tools() -> None:
    assert "npx --no-install markdownlint-cli2" in MAKEFILE
    assert "npx --no-install cspell" in MAKEFILE
    assert "scripts/checks/check_public_text.py" in MAKEFILE
    assert "$(MKDOCS) build --strict" in MAKEFILE
    assert "scripts/docs/" not in MAKEFILE


def test_docs_linters_discover_every_tracked_markdown_file() -> None:
    markdown_block = MAKE.target("docs-lint-markdown").text
    spell_block = MAKE.target("docs-lint-spell").text

    for block in (markdown_block, spell_block):
        assert "git ls-files -z -- ':(icase,glob)**/*.md'" in block
        assert "xargs -0" in block
        assert block.rstrip().endswith("--")
        assert "README.md CODE_OF_CONDUCT.md" not in block

    public_text_block = MAKE.target("docs-lint-public-text").text
    assert "scripts/checks/check_public_text.py" in public_text_block


def test_public_evidence_gate_is_canonical_index_only() -> None:
    block = MAKE.target("public-evidence-audit").text
    assert "check_public_evidence.py" in block
    assert "sync_packaged_public_evidence.py --check" in block
    assert "guard_scenario" not in MAKEFILE
