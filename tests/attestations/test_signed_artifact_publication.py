from __future__ import annotations

import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from invarlock import acceptance_attestation, evidence_receipt
from invarlock.filesystem import atomic_file


@pytest.fixture(
    params=[acceptance_attestation, evidence_receipt], ids=["attestation", "receipt"]
)
def writer(request: pytest.FixtureRequest) -> tuple[Callable, type[ValueError]]:
    module = request.param
    error = (
        acceptance_attestation.AcceptanceAttestationError
        if module is acceptance_attestation
        else evidence_receipt.EvidenceReceiptError
    )
    return module._write_no_clobber, error


@pytest.mark.parametrize("replacement", [False, True], ids=["absent", "other-writer"])
def test_partial_signed_artifact_is_never_published_or_used_for_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: tuple[Callable, type[ValueError]],
    replacement: bool,
) -> None:
    destination = tmp_path / "signed.json"
    payload = b"complete signed artifact"
    real_fdopen = atomic_file.os.fdopen

    class PartialWrite:
        def __init__(self, descriptor: int, mode: str, **kwargs) -> None:
            self.stream = real_fdopen(descriptor, mode, **kwargs)

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, value: bytes) -> None:
            assert value == payload
            self.stream.write(value[:7])
            self.stream.flush()
            assert not destination.exists()
            staged = list(tmp_path.glob(".invarlock-write-*/payload"))
            assert len(staged) == 1
            assert staged[0].read_bytes() == value[:7]
            if replacement:
                destination.write_bytes(b"other writer")
            raise OSError("partial write failure")

    monkeypatch.setattr(atomic_file.os, "fdopen", PartialWrite)
    write, error = writer
    with pytest.raises(error, match="could not write"):
        write(destination, payload)
    if replacement:
        assert destination.read_bytes() == b"other writer"
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob(".invarlock-write-*"))


@pytest.mark.parametrize("replacement", [False, True], ids=["absent", "other-writer"])
def test_signed_artifact_fsync_failure_never_exposes_or_deletes_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: tuple[Callable, type[ValueError]],
    replacement: bool,
) -> None:
    destination = tmp_path / "signed.json"
    payload = b"complete signed artifact"

    def fail_sync(descriptor: int) -> None:
        assert stat.S_ISREG(atomic_file.os.fstat(descriptor).st_mode)
        assert not destination.exists()
        staged = list(tmp_path.glob(".invarlock-write-*/payload"))
        assert len(staged) == 1
        assert staged[0].read_bytes() == payload
        if replacement:
            destination.write_bytes(b"other writer")
        raise OSError("durable write failure")

    monkeypatch.setattr(atomic_file.os, "fsync", fail_sync)
    write, error = writer
    with pytest.raises(error, match="could not write"):
        write(destination, payload)
    if replacement:
        assert destination.read_bytes() == b"other writer"
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob(".invarlock-write-*"))


@pytest.mark.parametrize("replacement", [False, True], ids=["absent", "other-writer"])
def test_signed_artifact_chmod_changes_only_the_owned_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: tuple[Callable, type[ValueError]],
    replacement: bool,
) -> None:
    destination = tmp_path / "signed.json"
    real_fchmod = atomic_file.os.fchmod
    changed: list[tuple[int, int]] = []

    def chmod_owned(descriptor: int, mode: int) -> None:
        assert mode == 0o444
        assert not destination.exists()
        before = atomic_file.os.fstat(descriptor)
        assert stat.S_ISREG(before.st_mode)
        changed.append((before.st_dev, before.st_ino))
        if replacement:
            destination.write_bytes(b"other writer")
            destination.chmod(0o640)
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(atomic_file.os, "fchmod", chmod_owned)
    write, error = writer
    if replacement:
        with pytest.raises(error, match="already exists"):
            write(destination, b"ours")
        assert destination.read_bytes() == b"other writer"
        assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    else:
        write(destination, b"ours")
        assert destination.read_bytes() == b"ours"
        after = destination.stat()
        assert changed == [(after.st_dev, after.st_ino)]
        assert stat.S_IMODE(after.st_mode) == 0o444
    assert len(changed) == 1
    assert not list(tmp_path.glob(".invarlock-write-*"))


def test_signed_artifact_replacement_after_publication_is_never_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: tuple[Callable, type[ValueError]],
) -> None:
    destination = tmp_path / "signed.json"
    previous = tmp_path / "previous.json"
    real_publish = atomic_file._rename_no_replace

    def publish_then_replace(**kwargs):
        result = real_publish(**kwargs)
        assert result == 0
        assert destination.read_bytes() == b"ours"
        destination.rename(previous)
        destination.write_bytes(b"other writer")
        return result

    monkeypatch.setattr(atomic_file, "_rename_no_replace", publish_then_replace)
    write, error = writer
    with pytest.raises(error, match="could not write"):
        write(destination, b"ours")
    assert destination.read_bytes() == b"other writer"
    assert previous.read_bytes() == b"ours"
    assert not list(tmp_path.glob(".invarlock-write-*"))
