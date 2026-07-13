from __future__ import annotations

import json
from pathlib import Path

DOC_PATH = Path("docs/reference/model-family-catalog.md")
CONTRACT_PATH = Path("contracts/model_family_catalog.json")


def test_model_family_catalog_doc_matches_current_contract_surface() -> None:
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert len(payload["declared_support"]) == 39
    assert "39 maintained lanes" in doc_text
    assert "Evidence not yet created" in doc_text
    assert "contracts/model_family_catalog.json" in doc_text
    assert "contracts/evidence_catalog_v1.json" in doc_text
    assert "contracts/support_matrix.json" in doc_text

    for heading in (
        "Maintained evaluation lanes",
        "Implementation coverage",
        "Adding a maintained lane",
    ):
        assert f"## {heading}" in doc_text

    for state in (
        "profile_first_class",
        "profile_shared_alias",
        "auto_or_loader_only",
        "loader_only",
    ):
        assert f"`{state}`" in doc_text
