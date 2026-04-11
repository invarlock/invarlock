from __future__ import annotations

from pathlib import Path


def test_api_and_guard_docs_track_typed_guard_decisions() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    api_guide = (repo_root / "docs" / "reference" / "api-guide.md").read_text(
        encoding="utf-8"
    )
    guards_ref = (repo_root / "docs" / "reference" / "guards.md").read_text(
        encoding="utf-8"
    )
    assurance_doc = (
        repo_root / "docs" / "assurance" / "04-guard-contracts.md"
    ).read_text(encoding="utf-8")

    assert "return typed decisions (`allow`/`monitor`/`rollback`/`block`)." in api_guide
    assert "`validate(...)` should emit the typed decision vocabulary:" in api_guide
    assert '"decision": "monitor"' in api_guide
    assert "return action (warn/rollback/abort)." not in api_guide
    assert "`action: warn`" not in api_guide
    assert "`action: abort`" not in api_guide

    assert "`report.guards[].decision`" in guards_ref
    assert "`report.guards[].diagnostics`" in guards_ref
    assert "`report.meta.tier_policies`" in guards_ref
    assert "`report.guards[].action`" not in guards_ref
    assert "`report.guards[].actions`" not in guards_ref

    assert "guard emits a blocking decision" in assurance_doc
    assert "`action = abort`" not in assurance_doc
