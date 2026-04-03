from __future__ import annotations

import json
from pathlib import Path

from invarlock.reporting.evidence import maybe_dump_guard_evidence


def test_maybe_dump_guard_evidence_noops_when_debug_is_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("INVARLOCK_EVIDENCE_DEBUG", raising=False)

    maybe_dump_guard_evidence(tmp_path, {"ok": True})

    assert not (tmp_path / "guards_evidence.json").exists()


def test_maybe_dump_guard_evidence_writes_json_when_debug_is_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", "1")

    maybe_dump_guard_evidence(tmp_path, {"guard": "spectral", "ok": True})

    payload = json.loads(
        (tmp_path / "guards_evidence.json").read_text(encoding="utf-8")
    )
    assert payload == {"guard": "spectral", "ok": True}


def test_maybe_dump_guard_evidence_accepts_string_target_dir(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", "1")

    maybe_dump_guard_evidence(str(tmp_path), {"guard": "variance", "ok": False})

    payload = json.loads(
        (tmp_path / "guards_evidence.json").read_text(encoding="utf-8")
    )
    assert payload == {"guard": "variance", "ok": False}


def test_maybe_dump_guard_evidence_swallows_filesystem_errors(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INVARLOCK_EVIDENCE_DEBUG", "1")
    target = tmp_path / "not-a-directory"
    target.write_text("sentinel\n", encoding="utf-8")

    maybe_dump_guard_evidence(target, {"ok": True})

    assert target.read_text(encoding="utf-8") == "sentinel\n"
