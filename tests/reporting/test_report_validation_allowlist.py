from __future__ import annotations

import pytest

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


def test_load_validation_allowlist_strict_raises_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        allowlist_mod,
        "load_json_contract",
        lambda _filename: (_ for _ in ()).throw(FileNotFoundError),
    )

    with pytest.raises(
        allowlist_mod.ValidationAllowlistContractError,
        match="Failed to load validation key contract",
    ):
        allowlist_mod.load_validation_allowlist_strict()


def test_load_validation_allowlist_strict_rejects_non_list_payload(
    monkeypatch,
) -> None:
    monkeypatch.setattr(allowlist_mod, "load_json_contract", lambda _filename: {})

    with pytest.raises(
        allowlist_mod.ValidationAllowlistContractError,
        match="non-empty JSON array of strings",
    ):
        allowlist_mod.load_validation_allowlist_strict()
