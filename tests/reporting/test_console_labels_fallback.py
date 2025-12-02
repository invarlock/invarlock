from __future__ import annotations

from pathlib import Path

from invarlock.reporting.render import _CONSOLE_LABELS_DEFAULT, _load_console_labels


def test_console_labels_fallback_when_contract_file_missing(monkeypatch):
    # Make the loader think the contracts file does not exist
    class _PathProxy(type(Path())):  # type: ignore[misc]
        def resolve(self):  # type: ignore[override]
            # Return a dummy path; existence will be forced False
            return self

        def exists(self):  # type: ignore[override]
            return False

        def read_text(self, *args, **kwargs):  # type: ignore[override]
            raise FileNotFoundError

    monkeypatch.setattr("invarlock.reporting.certificate.Path", _PathProxy)

    labels = _load_console_labels()
    # Default allow-list should include common rows
    assert any("Primary Metric" in lab for lab in labels)
    assert any("Spectral" in lab for lab in labels)
    assert any("Rmt" in lab or "RMT" in lab for lab in labels)


def test_console_labels_handles_invalid_payload(monkeypatch, tmp_path):
    target = Path(__file__).resolve().parents[2] / "contracts" / "console_labels.json"
    fake_file = tmp_path / "console_labels.json"
    fake_file.write_text("not a list", encoding="utf-8")

    real_read = Path.read_text

    def fake_exists(self):  # type: ignore[override]
        try:
            return self.resolve() == target
        except Exception:
            return False

    def fake_read(self, *args, **kwargs):  # type: ignore[override]
        try:
            if self.resolve() == target:
                return fake_file.read_text()
        except Exception:
            pass
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", fake_exists, raising=False)
    monkeypatch.setattr(Path, "read_text", fake_read, raising=False)

    labels = _load_console_labels()
    assert labels == _CONSOLE_LABELS_DEFAULT
