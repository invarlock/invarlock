from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

REMOVED_README_GUARANTEE_LABEL = "Statistical " + "guarantees"
REMOVED_REPORT_GUARANTEE_LABEL = "What the report " + "guarantees"
CLAIM_REQUIRED_BY_FILE = {
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
    "scripts/smoke/run_tiny_all_matrix.sh": ["Evaluation Matrix"],
    "tests/integration/scripts/test_tiny_matrix_checklist.py": [
        "Evaluation Matrix",
    ],
    "docs/README.md": [
        "Assurance Case",
        "Evaluation Math Derivation",
        "InvarLock maintains 39 evaluation lanes",
        "Evidence not yet created",
        "contracts/evidence_catalog_v1.json",
        "contracts/support_matrix.json",
        "Model Family Catalog",
    ],
    "mkdocs.yml": [
        "Model Family Catalog: reference/model-family-catalog.md",
        "Assurance Case: assurance/00-assurance-case.md",
        "Evaluation Math Derivation: assurance/01-eval-math-derivation.md",
    ],
    "README.md": ["docs/assurance/00-assurance-case.md"],
    "docs/user-guide/reading-report.md": ["Assurance Case"],
    "docs/reference/index.md": ["Assurance rationale and derivations"],
    "docs/reference/contracts.md": ["Model family catalog"],
    "docs/reference/model-family-catalog.md": [
        "Maintained evaluation lanes",
        "Support tier and current evidence status",
        "Implementation coverage",
        "Coverage states include",
        "Adding a maintained lane",
    ],
    "docs/assurance/00-assurance-case.md": [
        "assurance case",
        "assurance claims",
        "published assurance tiers",
    ],
    "docs/assurance/04-guard-contracts.md": [
        "contracts/support_matrix.json",
        "docs/README.md#support-matrix",
        "evidence status",
    ],
    "docs/reference/calibration.md": [
        "contracts/support_matrix.json",
        "docs/README.md#support-matrix",
        "Evidence not yet created",
    ],
    "docs/reference/guards.md": [
        "contracts/support_matrix.json",
        "maintained lanes",
        "evidence status",
    ],
    "docs/reference/model-adapters.md": [
        "Maintained catalog lanes span GPT-2/BERT",
        "FLAN-T5 through `hf_seq2seq`",
        "Gemma 4 image-text through `hf_multimodal` plus `vision_text`",
        "lanes such as OLMoE, Mixtral, and Qwen3 30B-A3B",
        "`contracts/support_matrix.json` as authoritative",
        "Model Family Catalog",
    ],
    "docs/user-guide/evidence-packs.md": [
        "signed manifest",
        "strict verification",
        "PASS final verdict",
    ],
    "Makefile": ["eval-loop:"],
}

CLAIM_BANNED_BY_FILE = {
    "README.md": [REMOVED_README_GUARANTEE_LABEL],
    "docs/user-guide/quickstart.md": ["machine-readable safety report"],
    "docs/user-guide/getting-started.md": [
        "make cert-loop",
        "safety guarantees",
        "[Safety Case]",
    ],
    "docs/user-guide/example-reports.md": ["Signed compliance payload"],
    "SUPPORT.md": ["certif" + "ication flow"],
    "scripts/smoke/run_tiny_all_matrix.sh": [
        "Certif" + "ication Matrix",
        "gpt2_cert_",
        "gpt2_editcert_quant8",
        "bert_mlm_cert",
        "distilbert_cls_cert",
    ],
    "tests/integration/scripts/test_tiny_matrix_checklist.py": [
        "Certif" + "ication Matrix",
    ],
    "mkdocs.yml": ["Safety Case:"],
    "docs/README.md": ["Safety Case", "safety claim"],
    "docs/reference/index.md": ["Safety claims and proofs"],
    "docs/reference/architecture.md": ["Safety Case Overview"],
    "docs/reference/reports.md": ["Safety Case", REMOVED_REPORT_GUARANTEE_LABEL],
    "docs/user-guide/reading-report.md": ["[Safety Case]"],
    "docs/user-guide/primary-metric-smoke.md": ["Evaluation Math Proof"],
    "docs/assurance/00-assurance-case.md": [
        "safety case",
        "safety claims",
        "safety tiers",
    ],
    "Makefile": ["cert-loop:"],
}


def _check_required_snippets(
    failures: list[str],
    root: Path,
    rel_path: str,
    required_snippets: list[str],
    *,
    read_text: Callable[[Path], str],
) -> None:
    text = read_text(root / rel_path)
    for snippet in required_snippets:
        if snippet not in text:
            failures.append(f"{rel_path}: missing required snippet: {snippet!r}")


def _check_banned_snippets(
    failures: list[str],
    root: Path,
    rel_path: str,
    banned_snippets: list[str],
    *,
    read_text: Callable[[Path], str],
) -> None:
    text = read_text(root / rel_path)
    for snippet in banned_snippets:
        if snippet in text:
            failures.append(f"{rel_path}: banned snippet present: {snippet!r}")


def check_claim_surface_consistency(
    *,
    root: Path,
    read_text: Callable[[Path], str],
) -> int:
    failures: list[str] = []

    for rel_path, required_snippets in CLAIM_REQUIRED_BY_FILE.items():
        _check_required_snippets(
            failures,
            root,
            rel_path,
            required_snippets,
            read_text=read_text,
        )
    for rel_path, banned_snippets in CLAIM_BANNED_BY_FILE.items():
        _check_banned_snippets(
            failures,
            root,
            rel_path,
            banned_snippets,
            read_text=read_text,
        )

    if failures:
        print("[check_claim_surface_consistency] FAIL", file=sys.stderr)
        for failure in failures:
            print(f" - {failure}", file=sys.stderr)
        return 1

    print("[check_claim_surface_consistency] OK")
    return 0


__all__ = [
    "CLAIM_BANNED_BY_FILE",
    "CLAIM_REQUIRED_BY_FILE",
    "check_claim_surface_consistency",
]
