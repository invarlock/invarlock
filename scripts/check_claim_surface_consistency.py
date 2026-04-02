#!/usr/bin/env python3
"""Enforce public claim-surface wording and support-scope consistency."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _check_required_snippets(
    failures: list[str], rel_path: str, required_snippets: list[str]
) -> None:
    text = _read(rel_path)
    for snippet in required_snippets:
        if snippet not in text:
            failures.append(f"{rel_path}: missing required snippet: {snippet!r}")


def _check_banned_snippets(
    failures: list[str], rel_path: str, banned_snippets: list[str]
) -> None:
    text = _read(rel_path)
    for snippet in banned_snippets:
        if snippet in text:
            failures.append(f"{rel_path}: banned snippet present: {snippet!r}")


def main() -> int:
    failures: list[str] = []

    required_by_file = {
        "docs/user-guide/quickstart.md": [
            "machine-readable evaluation report",
            "invarlock verify",
            "report html",
        ],
        "docs/user-guide/getting-started.md": [
            "invarlock evaluate",
            "invarlock verify",
            "report html",
            "Assurance Case",
        ],
        "docs/user-guide/example-reports.md": [
            "Machine-readable evaluation report",
        ],
        "SUPPORT.md": ["evaluation workflows"],
        "scripts/run_tiny_all_matrix.sh": ["Evaluation Matrix"],
        "tests/integration/scripts/test_tiny_matrix_checklist.py": [
            "Evaluation Matrix",
        ],
        "docs/README.md": [
            "Assurance Case",
            "Evaluation Math Derivation",
            "Published assurance basis covers GPT-2 and BERT profiles.",
            "Mistral 7B",
            "Qwen2 7B",
            "Qwen2.5 14B",
            "pilot calibration configs",
            "Model Family Catalog",
        ],
        "mkdocs.yml": [
            "Model Family Catalog: reference/model-family-catalog.md",
            "Assurance Case: assurance/00-assurance-case.md",
            "Evaluation Math Derivation: assurance/01-eval-math-derivation.md",
        ],
        "README.md": ["docs/assurance/00-assurance-case.md"],
        "docs/user-guide/reading-report.md": ["Assurance Case"],
        "docs/reference/index.md": ["Assurance claims and derivations"],
        "docs/reference/contracts.md": ["Model family catalog"],
        "docs/reference/model-family-catalog.md": [
            "support tier",
            "coverage state",
            "Declared Support",
            "Recommended Additions",
        ],
        "docs/assurance/00-assurance-case.md": [
            "assurance case",
            "assurance claims",
            "published assurance tiers",
        ],
        "docs/assurance/04-guard-contracts.md": [
            "Published assurance basis covers GPT-2 and BERT profiles.",
            "Mistral 7B",
            "Qwen2 7B",
            "Qwen2.5 14B",
            "not part of the published",
        ],
        "docs/reference/calibration.md": [
            "Published assurance basis covers GPT-2 and BERT profiles.",
            "Mistral 7B",
            "Qwen2 7B",
            "Qwen2.5 14B",
        ],
        "docs/reference/guards.md": [
            "GPT-2",
            "BERT profiles",
            "Mistral 7B",
            "Qwen2 7B",
            "Qwen2.5 14B",
            "published assurance basis",
        ],
        "docs/reference/model-adapters.md": [
            "Adapter availability is broader than the published assurance basis.",
            "GPT-2",
            "BERT",
            "Mistral 7B",
            "Qwen2 7B",
            "Qwen2.5 14B",
        ],
        "docs/user-guide/proof-packs.md": [
            "signed manifest",
            "strict verification",
            "PASS final verdict",
        ],
        "scripts/proof_packs/run_pack.sh": [
            "signed manifest",
            "strict verification",
            "PASS final verdict",
        ],
        "scripts/proof_packs/tests/test_run_pack.sh": [
            "signed manifest, strict verification, and a PASS final verdict",
        ],
        "Makefile": ["eval-loop:"],
    }

    banned_by_file = {
        "docs/user-guide/quickstart.md": ["machine-readable safety report"],
        "docs/user-guide/getting-started.md": [
            "make cert-loop",
            "safety guarantees",
            "[Safety Case]",
        ],
        "docs/user-guide/example-reports.md": ["Signed compliance payload"],
        "SUPPORT.md": ["certification flow"],
        "scripts/run_tiny_all_matrix.sh": [
            "Certification Matrix",
            "gpt2_cert_",
            "gpt2_editcert_quant8",
            "bert_mlm_cert",
            "distilbert_cls_cert",
        ],
        "tests/integration/scripts/test_tiny_matrix_checklist.py": [
            "Certification Matrix",
        ],
        "mkdocs.yml": ["Safety Case:"],
        "docs/README.md": ["Safety Case", "safety claim"],
        "docs/reference/index.md": ["Safety claims and proofs"],
        "docs/reference/architecture.md": ["Safety Case Overview"],
        "docs/reference/reports.md": ["Safety Case"],
        "docs/user-guide/reading-report.md": ["[Safety Case]"],
        "docs/user-guide/primary-metric-smoke.md": ["Evaluation Math Proof"],
        "docs/assurance/00-assurance-case.md": [
            "safety case",
            "safety claims",
            "safety tiers",
        ],
        "Makefile": ["cert-loop:"],
        "scripts/proof_packs/lib/task_functions.sh": ["Certification for "],
    }

    for rel_path, required_snippets in required_by_file.items():
        _check_required_snippets(failures, rel_path, required_snippets)
    for rel_path, banned_snippets in banned_by_file.items():
        _check_banned_snippets(failures, rel_path, banned_snippets)

    if failures:
        print("[check_claim_surface_consistency] FAIL", file=sys.stderr)
        for failure in failures:
            print(f" - {failure}", file=sys.stderr)
        return 1

    print("[check_claim_surface_consistency] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
