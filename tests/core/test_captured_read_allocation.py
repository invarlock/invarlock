from __future__ import annotations

import os

import pytest

from invarlock import captured_contracts as contracts


def _observe_reads(monkeypatch, *, before_read=None, after_read=None):
    original = os.fdopen
    requested = []

    class Reader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def read(self, size):
            requested.append(size)
            if before_read is not None:
                before_read()
            result = self.handle.read(size)
            if after_read is not None:
                after_read()
            return result

    monkeypatch.setattr(os, "fdopen", lambda *a, **k: Reader(original(*a, **k)))
    return requested


@pytest.mark.parametrize(
    "payload,limit", [(b"", 0), (b"small", 5), (b"small", 128 << 20)]
)
def test_read_allocation_follows_pinned_size_and_accepts_valid_boundary_files(
    tmp_path, monkeypatch, payload, limit
):
    path = (tmp_path / "payload").resolve()
    path.write_bytes(payload)
    requested = _observe_reads(monkeypatch)
    assert contracts.read_file(path, limit) == payload
    assert requested and max(requested) <= len(payload) + 1


@pytest.mark.parametrize("change", ["grow", "grow-beyond-limit", "shrink", "replace"])
def test_size_bounded_read_still_rejects_changes_during_read(
    tmp_path, monkeypatch, change
):
    path = (tmp_path / "payload").resolve()
    path.write_bytes(b"original")

    def mutate():
        if change == "grow":
            path.write_bytes(b"original+")
        elif change == "grow-beyond-limit":
            path.write_bytes(b"original" * 20)
        elif change == "shrink":
            path.write_bytes(b"tiny")
        else:
            replacement = path.with_name("replacement")
            replacement.write_bytes(b"replaced")
            replacement.replace(path)

    _observe_reads(monkeypatch, before_read=mutate)
    with pytest.raises(contracts.CapturedContractError, match="changed|exceeds"):
        contracts.read_file(path, 32)


def test_same_size_edit_after_read_still_fails_identity_check(tmp_path, monkeypatch):
    path = (tmp_path / "payload").resolve()
    path.write_bytes(b"original")
    _observe_reads(monkeypatch, after_read=lambda: path.write_bytes(b"modified"))
    with pytest.raises(contracts.CapturedIntegrityError, match="changed while reading"):
        contracts.read_file(path, 32)


def test_oversize_file_is_rejected_before_opening_a_payload(tmp_path, monkeypatch):
    path = (tmp_path / "payload").resolve()
    path.write_bytes(b"too large")
    with contracts.secure_directory(path.parent) as parent:
        monkeypatch.setattr(
            os, "open", lambda *a, **k: pytest.fail("opened oversized file")
        )
        with pytest.raises(contracts.CapturedContractError, match="exceeds"):
            contracts._read_at(parent, path.name, 4)
