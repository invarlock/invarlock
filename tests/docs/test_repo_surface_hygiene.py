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
    removed_paths = [
        "docs/assurance/01-logspace-rationale.md",
        "docs/assurance/02-eval-math-proof.md",
        "docs/assurance/03-coverage-and-pairing.md",
        "docs/assurance/04-bca-bootstrap.md",
        "docs/assurance/05-guard-contracts.md",
        "docs/assurance/06-spectral-fpr-derivation.md",
        "docs/assurance/07-rmt-epsilon-rule.md",
        "docs/assurance/08-ve-gate-power.md",
        "docs/assurance/09-determinism-contracts.md",
        "docs/reference/exporting-certificates-html.md",
        "docs/user-guide/compare-and-certify.md",
        "docs/user-guide/reading-certificate.md",
    ]

    for rel_path in expected_curated_paths:
        assert f"!{rel_path}" in text, f"missing markdownlint curated path {rel_path}"
        assert (REPO_ROOT / rel_path).exists(), (
            f"curated markdownlint path missing: {rel_path}"
        )

    for rel_path in removed_paths:
        assert rel_path not in text, (
            f"removed markdownlint path still present: {rel_path}"
        )


def test_gitignore_keeps_current_output_paths_and_drops_stale_legacy_scratch() -> None:
    text = _read(".gitignore")
    required_patterns = [
        "/reports/",
        "/reports_*/",
        "/runs/",
        "/runs_cfg/",
        "/guards_evidence.json",
        "/tmp/",
        "/tmp_*/",
    ]
    removed_patterns = [
        "/reports_report/",
        "/.certify_tmp/",
        "/certificates/",
        "*_certificate/",
        "cert-*.json",
        "*.cert",
        "fullLisk.txt",
        "*.clinerules",
        ".clinerules",
        "/demo_*/",
        "/mock_run/",
        "test_comprehensive_results*.json",
        "mps_full_stats/",
        "test_mps_stats/",
        "optuna_results/",
        "invarlock_comparison_results/",
    ]

    for pattern in required_patterns:
        assert pattern in text, f"required gitignore pattern missing: {pattern}"

    for pattern in removed_patterns:
        assert pattern not in text, f"stale gitignore pattern still present: {pattern}"


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
