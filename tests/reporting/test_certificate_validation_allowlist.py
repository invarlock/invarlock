from __future__ import annotations

from pathlib import Path

from invarlock.reporting import certificate as cert


def test_load_validation_allowlist_returns_default_when_missing(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False, raising=False)
    allowlist = cert._load_validation_allowlist()
    assert allowlist == set(cert._VALIDATION_ALLOWLIST_DEFAULT)


def test_load_validation_allowlist_handles_non_list_payload(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True, raising=False)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding="utf-8": "{}",
        raising=False,
    )
    allowlist = cert._load_validation_allowlist()
    assert allowlist == set(cert._VALIDATION_ALLOWLIST_DEFAULT)
