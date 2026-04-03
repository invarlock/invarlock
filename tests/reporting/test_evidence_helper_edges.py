from __future__ import annotations

from pathlib import Path

from invarlock.reporting.evidence import maybe_dump_guard_evidence


def test_maybe_dump_guard_evidence_swallows_json_encoding_errors(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", "1")

    maybe_dump_guard_evidence(tmp_path, {"bad": object()})

    assert not (tmp_path / "guards_evidence.json").exists()
