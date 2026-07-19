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
        "example-scenarios-check",
        "contracts-check",
        "test",
        "cli-smoke-core",
        "lint",
        "docs-check-build",
    ):
        assert f"$(MAKE) {target}" in block


def test_verify_fast_includes_the_scenario_contract() -> None:
    block = _block("verify-fast", "contracts-check")
    assert "$(MAKE) example-scenarios-check" in block


def test_docs_are_checked_by_established_tools() -> None:
    assert "npx --no-install markdownlint-cli2" in MAKEFILE
    assert "npx --no-install cspell" in MAKEFILE
    assert "$(MKDOCS) build --strict" in MAKEFILE
    assert "scripts/docs/" not in MAKEFILE


def test_public_evidence_gate_is_canonical_index_only() -> None:
    block = _block("public-evidence-audit", "public-evidence-sync")
    assert "check_public_evidence.py" in block
    assert "sync_packaged_public_evidence.py --check" in block
    assert "guard_scenario" not in MAKEFILE
