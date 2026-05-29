from __future__ import annotations

import json
from pathlib import Path

from invarlock.reporting.report_evidence import maybe_dump_guard_evidence


def test_maybe_dump_guard_evidence_swallows_json_encoding_errors(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", "1")

    assert maybe_dump_guard_evidence(tmp_path, {"bad": object()}) is None

    assert not (tmp_path / "guards_evidence.json").exists()


def test_maybe_dump_guard_evidence_strips_debug_flag_whitespace(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", " 1 ")

    assert maybe_dump_guard_evidence(tmp_path, {"ok": True}) == (
        tmp_path / "guards_evidence.json"
    )

    payload = json.loads(
        (tmp_path / "guards_evidence.json").read_text(encoding="utf-8")
    )
    assert payload == {"ok": True}


def test_maybe_dump_guard_evidence_swallows_non_path_targets(monkeypatch) -> None:
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", "1")

    assert maybe_dump_guard_evidence(object(), {"ok": True}) is None
