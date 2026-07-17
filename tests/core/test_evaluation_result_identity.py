from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.evaluation_transaction import EvaluationTransactionResult
from invarlock.evidence_pack_contract import EvidencePackError


def test_evaluation_result_json_carries_exact_manifest_identity(tmp_path: Path) -> None:
    digest = "sha256:" + ("a" * 64)
    result = EvaluationTransactionResult(
        evidence_path=tmp_path / "evidence",
        comparison_id="comparison-123",
        pack_manifest_digest=digest,
    )

    payload = json.loads(result.as_json())

    assert payload == {
        "comparison_id": "comparison-123",
        "evidence": str(tmp_path / "evidence"),
        "format_version": "invarlock/evaluation-result-v1",
        "ok": True,
        "pack_manifest_digest": digest,
    }


def test_evaluation_result_rejects_a_non_digest_publication_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(EvidencePackError, match="published evidence manifest digest"):
        EvaluationTransactionResult(
            evidence_path=tmp_path / "evidence",
            comparison_id="comparison-123",
            pack_manifest_digest="not-a-digest",
        )
