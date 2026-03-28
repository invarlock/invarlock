from __future__ import annotations

from invarlock.reporting.report_console import (
    _CONSOLE_LABELS_DEFAULT,
    load_console_labels,
)


def test_console_labels_fallback_when_contract_file_missing(monkeypatch):
    def _missing(_filename: str):
        raise FileNotFoundError

    monkeypatch.setattr(
        "invarlock.reporting.report_console.load_json_contract", _missing
    )
    labels = load_console_labels()
    # Default allow-list should include common rows
    assert any("Primary Metric" in lab for lab in labels)
    assert any("Spectral" in lab for lab in labels)
    assert any("Rmt" in lab or "RMT" in lab for lab in labels)


def test_console_labels_handles_invalid_payload(monkeypatch):
    monkeypatch.setattr(
        "invarlock.reporting.report_console.load_json_contract",
        lambda _filename: {"not": "a list"},
    )
    labels = load_console_labels()
    assert labels == _CONSOLE_LABELS_DEFAULT
