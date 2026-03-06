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
        "docs/user-guide/quickstart.md": ["machine-readable evaluation"],
        "docs/user-guide/getting-started.md": ["make eval-loop"],
        "docs/user-guide/example-reports.md": [
            "Machine-readable evaluation report",
        ],
        "SUPPORT.md": ["evaluation workflows"],
        "scripts/run_tiny_all_matrix.sh": ["Evaluation Matrix"],
        "tests/integration/scripts/test_tiny_matrix_checklist.py": [
            "Evaluation Matrix",
        ],
        "docs/README.md": [
            "Published assurance basis currently covers GPT-2 and BERT profiles.",
            "Mistral 7B",
            "Qwen2 7B",
            "pilot calibration configs",
        ],
        "docs/assurance/04-guard-contracts.md": [
            "Published assurance basis currently covers GPT-2 and BERT profiles.",
            "Mistral 7B",
            "Qwen2 7B",
            "not part of the published",
        ],
        "docs/reference/calibration.md": [
            "Published assurance basis currently covers GPT-2 and BERT profiles.",
            "Mistral 7B",
            "Qwen2 7B",
        ],
        "docs/reference/guards.md": [
            "GPT-2",
            "BERT profiles",
            "Mistral 7B",
            "Qwen2 7B",
            "published assurance basis",
        ],
        "docs/reference/model-adapters.md": [
            "Adapter availability is broader than the published assurance basis.",
            "GPT-2",
            "BERT",
            "Mistral 7B",
            "Qwen2 7B",
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
        "docs/user-guide/getting-started.md": ["make cert-loop"],
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
