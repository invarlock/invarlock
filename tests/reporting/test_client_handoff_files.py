"""Copy limits and concurrent edits must abort before package publication."""

import os

import pytest

from invarlock import client_review_files as copies
from invarlock.filesystem.paths import pinned_directory


@pytest.mark.parametrize(
    ("limit", "value", "message"),
    [
        ("MAX_BYTES", 3, "copy limit"),
        ("MAX_ENTRIES", 1, "entry limit"),
        ("MAX_DEPTH", 0, "nesting"),
    ],
)
def test_copy_limits(tmp_path, monkeypatch, limit, value, message):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/file").write_bytes(b"1234")
    monkeypatch.setattr(copies, limit, value)
    with pinned_directory(source) as fd, pytest.raises(ValueError, match=message):
        copies.copy_evidence(fd, tmp_path / "copy")


@pytest.mark.parametrize(
    "member", ["fifo", "link", "directory-link", "bad\\\\name", "bad\\nname"]
)
def test_special_files_and_nonportable_names_are_rejected(tmp_path, member):
    source = tmp_path / "source"
    source.mkdir()
    path = source / member
    if member == "fifo":
        os.mkfifo(path)
    elif member in {"link", "directory-link"}:
        path.symlink_to(tmp_path if member == "directory-link" else __file__)
    else:
        path.write_bytes(b"x")
    with pinned_directory(source) as fd, pytest.raises(ValueError):
        copies.copy_evidence(fd, tmp_path / "copy")


@pytest.mark.parametrize(
    ("replacement", "message"),
    [(b"longer content", "grew"), (b"x", "shrank"), (b"other", "changed while")],
)
def test_mutation_during_read_is_detected(tmp_path, monkeypatch, replacement, message):
    source = tmp_path / "source"
    source.mkdir()
    file = source / "file"
    file.write_bytes(b"first")
    read = os.read
    mutated = False

    def change(fd, count):
        nonlocal mutated
        if not mutated:
            mutated = True
            file.write_bytes(replacement)
        return read(fd, count)

    monkeypatch.setattr(copies.os, "read", change)
    with pinned_directory(source) as fd, pytest.raises(ValueError, match=message):
        copies.copy_evidence(fd, tmp_path / "copy")


def test_replacement_before_open_is_detected(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    file = source / "file"
    file.write_bytes(b"original")
    real_open = os.open

    def replace(path, flags, *args, **kwargs):
        if path == "file":
            file.unlink()
            file.write_bytes(b"new")
        return real_open(path, flags, *args, **kwargs)

    with pinned_directory(source) as fd:
        monkeypatch.setattr(copies.os, "open", replace)
        with pytest.raises(ValueError, match="before copying"):
            copies.copy_evidence(fd, tmp_path / "copy")


def test_added_member_during_directory_scan_is_detected(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    real_list = os.listdir

    def change(fd):
        result = real_list(fd)
        (source / "new").write_bytes(b"new")
        return result

    with pinned_directory(source) as fd:
        monkeypatch.setattr(copies.os, "listdir", change)
        with pytest.raises(ValueError, match="directory changed"):
            copies.copy_evidence(fd, tmp_path / "copy")
