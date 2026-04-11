from __future__ import annotations

from invarlock.reporting import report_validation_allowlist as allowlist_mod


def test_load_validation_allowlist_returns_default_when_missing(monkeypatch):
    monkeypatch.setattr(
        allowlist_mod,
        "load_json_contract",
        lambda _filename: (_ for _ in ()).throw(FileNotFoundError),
    )
    allowlist = allowlist_mod.load_validation_allowlist()
    assert allowlist == set(allowlist_mod.DEFAULT_VALIDATION_ALLOWLIST)


def test_load_validation_allowlist_handles_non_list_payload(monkeypatch):
    monkeypatch.setattr(allowlist_mod, "load_json_contract", lambda _filename: {})
    allowlist = allowlist_mod.load_validation_allowlist()
    assert allowlist == set(allowlist_mod.DEFAULT_VALIDATION_ALLOWLIST)
