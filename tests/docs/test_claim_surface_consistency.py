from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_claim_surface_consistency_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/docs/docs_check.py", "--claim-surface"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_knowledge_self_edit_workflow_page_keeps_bounded_scope() -> None:
    text = (
        REPO_ROOT / "docs" / "user-guide" / "knowledge-and-self-edit-workflows.md"
    ).read_text(encoding="utf-8")
    lower = text.lower()

    assert "external editor creates the subject" in lower
    assert "declared baseline against that subject" in lower
    assert "reporting context for the existing weight-edit regression contract" in lower
    assert "named profile would be required" in lower

    banned = [
        "guarantees locality",
        "guarantees robustness",
        "guarantees safety",
        "knowledge-edit assurance mode",
        "self-edit assurance mode",
        "knowledge-edit profile",
        "self-edit profile",
    ]
    for phrase in banned:
        assert phrase not in lower


def test_knowledge_self_edit_page_documents_realism_and_delta_privacy_boundaries() -> (
    None
):
    text = (
        REPO_ROOT / "docs" / "user-guide" / "knowledge-and-self-edit-workflows.md"
    ).read_text(encoding="utf-8")
    lower = text.lower()
    normalized = " ".join(lower.split())

    assert "evaluation realism" in lower
    assert "teacher-forced" in lower
    assert "live generation" in lower
    assert "regression signal" in lower
    assert "generation-mode lane" in normalized
    assert "raw deltas" in lower
    assert "adapter weights" in lower
    assert "hash-only" in lower

    banned = [
        "detects privacy leakage",
        "prevents privacy leakage",
        "certifies generation realism",
        "proves edit success",
    ]
    for phrase in banned:
        assert phrase not in lower
