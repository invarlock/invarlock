from __future__ import annotations

import json
from pathlib import Path

DOC_PATH = Path("docs/reference/model-family-catalog.md")
CONTRACT_PATH = Path("contracts/model_family_catalog.json")


def _section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    assert marker in text
    tail = text.split(marker, 1)[1]
    next_heading = tail.find("\n## ")
    if next_heading == -1:
        return tail
    return tail[:next_heading]


def test_model_family_catalog_doc_matches_contract_sections() -> None:
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert "support tier" in doc_text
    assert "coverage state" in doc_text
    assert "contracts/model_family_catalog.json" in doc_text

    section_map = {
        "Declared Support": payload["declared_support"],
        "Implemented Coverage": payload["implemented_coverage"],
        "Usage Only": payload["usage_only"],
        "Recommended Additions": payload["recommended_additions"],
    }

    for heading, entries in section_map.items():
        section_text = _section(doc_text, heading)
        for entry in entries:
            assert entry["display_name"] in section_text

    recommended_text = _section(doc_text, "Recommended Additions")
    for entry in payload["recommended_additions"]:
        assert entry["priority"] in recommended_text
        assert entry["planned_support_mode"] in recommended_text
