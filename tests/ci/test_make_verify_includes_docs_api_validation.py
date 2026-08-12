from __future__ import annotations

from pathlib import Path

MAKEFILE = (Path(__file__).resolve().parents[2] / "Makefile").read_text(
    encoding="utf-8"
)


def _block(name: str, next_name: str) -> str:
    return MAKEFILE.split(f"{name}:", 1)[1].split(f"{next_name}:", 1)[0]


def test_verify_composes_the_maintained_gates() -> None:
    block = _block("verify", "verify-fast")
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
    block = _block("verify-fast", "contracts-check")
    assert "$(MAKE) -j $(VERIFY_TARGET_JOBS)" in block
    assert "$(MAKE) examples-check PYTEST_WORKERS=$(PYTEST_WORKERS)" in block


def test_docs_are_checked_by_established_tools() -> None:
    assert "npx --no-install markdownlint-cli2" in MAKEFILE
    assert "npx --no-install cspell" in MAKEFILE
    assert "scripts/checks/check_public_text.py" in MAKEFILE
    assert "$(MKDOCS) build --strict" in MAKEFILE
    assert "scripts/docs/" not in MAKEFILE


def test_docs_linters_discover_every_tracked_markdown_file() -> None:
    markdown_block = _block("docs-lint-markdown", "docs-lint-spell")
    spell_block = _block("docs-lint-spell", "docs-lint-public-text")

    for block in (markdown_block, spell_block):
        assert "git ls-files -z -- ':(icase,glob)**/*.md'" in block
        assert "xargs -0" in block
        assert block.rstrip().endswith("--")
        assert "README.md CODE_OF_CONDUCT.md" not in block

    public_text_block = _block("docs-lint-public-text", "##@ Packaging and security")
    assert "scripts/checks/check_public_text.py" in public_text_block


def test_public_evidence_gate_is_canonical_index_only() -> None:
    block = _block("public-evidence-audit", "public-evidence-sync")
    assert "check_public_evidence.py" in block
    assert "sync_packaged_public_evidence.py --check" in block
    assert "guard_scenario" not in MAKEFILE
