from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_markdownlintignore_curated_docs_use_current_existing_paths() -> None:
    text = _read(".markdownlintignore")
    expected_curated_paths = [
        "docs/assurance/01-eval-math-derivation.md",
        "docs/assurance/02-coverage-and-pairing.md",
        "docs/assurance/03-bca-bootstrap.md",
        "docs/assurance/04-guard-contracts.md",
        "docs/assurance/05-spectral-fpr-derivation.md",
        "docs/assurance/06-rmt-epsilon-rule.md",
        "docs/assurance/07-ve-gate-power.md",
        "docs/assurance/08-determinism-contracts.md",
        "docs/reference/reports.md",
        "docs/user-guide/compare-and-evaluate.md",
        "docs/user-guide/reading-report.md",
    ]

    for rel_path in expected_curated_paths:
        assert f"!{rel_path}" in text, f"missing markdownlint curated path {rel_path}"
        assert (REPO_ROOT / rel_path).exists(), (
            f"curated markdownlint path missing: {rel_path}"
        )


def test_gitignore_keeps_current_output_paths() -> None:
    text = _read(".gitignore")
    required_patterns = [
        "/reports/",
        "/reports_*/",
        "/runs/",
        "/runs_cfg/",
        "/guards_evidence.json",
        "/tmp/",
        "/tmp_*/",
        "._*",
    ]

    for pattern in required_patterns:
        assert pattern in text, f"required gitignore pattern missing: {pattern}"


def test_public_docs_use_repo_and_package_native_wording_for_pack_verification() -> (
    None
):
    surfaces = [
        "README.md",
        "docs/reference/contracts.md",
        "docs/user-guide/evidence-packs.md",
        "docs/user-guide/getting-started.md",
    ]
    banned = [
        "wheel users",
        "installed-wheel users",
        "third parties",
    ]

    for rel_path in surfaces:
        text = _read(rel_path)
        for needle in banned:
            assert needle not in text, f"{needle} still present in {rel_path}"


def test_docs_node_toolchain_contract_is_explicit_and_ci_aligned() -> None:
    package_json = json.loads(_read("package.json"))
    engines = package_json.get("engines", {})
    assert engines.get("node") == ">=22.18.0"

    npmrc = _read(".npmrc")
    assert "engine-strict=true" in npmrc

    contributing = _read("CONTRIBUTING.md")
    assert "Node.js 22.18+ + npm" in contributing
    assert "npm ci` will fail early on older versions" in contributing

    workflows_doc = _read(".github/WORKFLOWS.md")
    assert "Node.js 22.18+" in workflows_doc
    assert "Node.js 18" not in workflows_doc
