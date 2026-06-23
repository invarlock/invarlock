from __future__ import annotations

import json
import re
import tomllib
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
        "docs/user-guide/knowledge-and-self-edit-workflows.md",
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
    package_lock = json.loads(_read("package-lock.json"))
    assert package_json.get("name") == package_lock.get("name")
    assert package_json.get("name") == package_lock.get("packages", {}).get("", {}).get(
        "name"
    )
    assert package_json.get("private") is True

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


def _notice_table_components(section: str) -> set[str]:
    text = _read("THIRD_PARTY_NOTICES.md")
    marker = f"## {section}"
    start = text.index(marker)
    next_start = text.find("\n## ", start + len(marker))
    body = text[start:] if next_start == -1 else text[start:next_start]
    components: set[str] = set()
    for line in body.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] in {"Component", "-----------"}:
            continue
        components.add(cells[0])
    return components


def test_third_party_direct_dependency_notices_match_pyproject() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    direct_dependencies = pyproject["project"]["dependencies"]
    expected = {
        re.split(r"[<>=!~;\\[]", dependency, maxsplit=1)[0]
        .strip()
        .lower()
        .replace("_", "-")
        for dependency in direct_dependencies
    }

    actual = {
        component.lower().replace("_", "-")
        for component in _notice_table_components("Direct Python Dependencies")
    }

    assert actual == expected
