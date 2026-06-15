from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

REMOVED_README_GUARANTEE_LABEL = "Statistical " + "guarantees"
REMOVED_REPORT_GUARANTEE_LABEL = "What the report " + "guarantees"
PUBLISHED_BASIS_INTRO = "Published assurance basis covers GPT-2, BERT, Mistral 7B"
PUBLISHED_BASIS_NAMES = [
    "Ministral 3 3B",
    "Ministral 3 8B",
    "Ministral 3 14B",
    "TinyLlama 1.1B",
    "Gemma 4 E2B text-only",
    "Gemma 4 E2B image-text",
    "Gemma 4 E4B image-text",
    "Granite 4.1 3B",
    "Granite 4.1 8B",
    "OLMo 2 7B",
    "OLMo 2 13B",
    "Qwen2 7B",
    "Qwen2.5 7B",
    "Qwen2.5 14B",
    "Qwen3 8B",
    "Qwen3.5 9B",
    "DeepSeek-R1-Distill-Qwen 7B",
    "DeepSeek-R1-0528-Qwen3 8B",
    "DeepSeek-R1-Distill-Qwen 14B",
    "Phi-4 text-only",
]

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
        PUBLISHED_BASIS_INTRO,
        "Mistral 7B",
        *PUBLISHED_BASIS_NAMES,
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
    "docs/reference/index.md": ["Assurance rationale and derivations"],
    "docs/reference/contracts.md": ["Model family catalog"],
    "docs/reference/model-family-catalog.md": [
        "support tier",
        "coverage state",
        "Declared Support",
        "<=14B Text Candidate Inventory",
        "Recommended Additions",
    ],
    "docs/assurance/00-assurance-case.md": [
        "assurance case",
        "assurance claims",
        "published assurance tiers",
    ],
    "docs/assurance/04-guard-contracts.md": [
        "published assurance basis",
        "contracts/support_matrix.json",
        "docs/README.md#support-matrix",
        "not part of the published",
    ],
    "docs/reference/calibration.md": [
        "published assurance basis",
        "contracts/support_matrix.json",
        "docs/README.md#support-matrix",
    ],
    "docs/reference/guards.md": [
        "contracts/support_matrix.json",
        "docs/README.md#support-matrix",
        "Mistral 7B",
        "published assurance basis",
    ],
    "docs/reference/model-adapters.md": [
        "Adapter availability is broader than the published evidence basis.",
        "`published_basis` lanes span GPT-2/BERT fixtures",
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
    "scripts/evidence_packs/run_pack.sh": [
        "signed manifest",
        "strict verification",
        "PASS final verdict",
    ],
    "scripts/evidence_packs/tests/test_run_pack.sh": [
        "signed manifest, strict verification, and a PASS final verdict",
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
    "scripts/evidence_packs/lib/tasks/task_functions.sh": ["Certif" + "ication for "],
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
