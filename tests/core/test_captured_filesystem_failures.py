"""Descriptor-bound capture and publication under real filesystem mutations."""

import errno
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest

from invarlock import captured_contracts as contracts
from invarlock import captured_evidence_publication as publication
from invarlock import captured_verification as verification
from invarlock.filesystem import AtomicDirectoryPublicationError
from tests.core.test_captured_contract_freeze import (
    _authenticate,
    _digest,
    _pack,
    _pem,
    _value,
    _verify_options,
)


@pytest.fixture
def publication_args(tmp_path):
    key = tmp_path / "signer.pem"
    key.write_bytes(_pem("signer"))
    return {
        "baseline": _value("baseline.json"),
        "subject": _value("subject-pass.json"),
        "policy": _value("policy.json"),
        "comparison": _value("comparison-pass.json"),
        "normalized_request": _value("request-pass.json"),
        "request_digest": _digest("request-pass.json"),
        "signing_key_path": key,
        "unsigned": False,
    }


@pytest.mark.parametrize("component", ["directory", "file", "snapshot-directory"])
@pytest.mark.parametrize(
    "error_number", [errno.ELOOP, errno.ENOENT, errno.ENOTDIR, errno.EACCES]
)
def test_open_failure_classifies_replacement_separately_from_io(
    tmp_path, monkeypatch, component, error_number
):
    target = tmp_path / "target"
    if component == "file":
        target.write_bytes(b"source")
    else:
        target.mkdir()
    original = os.open
    opened = []
    error = OSError(error_number, os.strerror(error_number))

    def fail_target(name, flags, *args, **kwargs):
        if name == "target":
            raise error
        descriptor = original(name, flags, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(contracts.os, "open", fail_target)
    expected = (
        PermissionError
        if error_number == errno.EACCES
        else contracts.CapturedIntegrityError
    )
    with pytest.raises(expected) as failure:
        if component == "directory":
            with contracts.secure_directory(target):
                pytest.fail("unsafe directory opened")
        elif component == "file":
            contracts.read_file(target, 100)
        else:
            with contracts.secure_directory(tmp_path) as parent:
                contracts._open_snapshot_directory(parent, "target")
    assert (
        (failure.value is error)
        if error_number == errno.EACCES
        else (failure.value.__cause__ is error)
    )
    for descriptor in opened:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(descriptor)


@pytest.mark.parametrize("directory", [False, True])
def test_replacement_between_stat_and_open_closes_all_descriptors(
    tmp_path, monkeypatch, directory
):
    target, displaced = tmp_path / "target", tmp_path / "displaced"
    if directory:
        target.mkdir()
    else:
        target.write_bytes(b"original")
    original = os.open
    opened = []

    def replace_then_open(name, flags, *args, **kwargs):
        if name == "target":
            target.rename(displaced)
            if directory:
                target.mkdir()
            else:
                target.write_bytes(b"replacement")
        descriptor = original(name, flags, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(contracts.os, "open", replace_then_open)
    with pytest.raises(contracts.CapturedIntegrityError, match="changed while opening"):
        if directory:
            with contracts.secure_directory(target):
                pytest.fail("replacement accepted")
        else:
            contracts.read_file(target, 100)
    for descriptor in opened:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(descriptor)


def test_directory_removed_during_use_is_not_a_stable_source(tmp_path):
    target = tmp_path / "removed"
    target.mkdir()
    with pytest.raises(
        contracts.CapturedIntegrityError, match="directory source was replaced"
    ):
        with contracts.secure_directory(target) as descriptor:
            target.rmdir()
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(descriptor)


@pytest.mark.parametrize("mutation", ["growth", "unlink", "replace"])
def test_file_mutation_during_bounded_read_is_rejected(tmp_path, monkeypatch, mutation):
    target = tmp_path / "source.json"
    target.write_bytes(b"{}\n")
    original_fdopen, original_stat = os.fdopen, os.stat
    changed = False

    def grow_before_read(descriptor, *args, **kwargs):
        nonlocal changed
        with target.open("ab") as output:
            output.write(b" " * 100)
        changed = True
        return original_fdopen(descriptor, *args, **kwargs)

    calls = 0

    def replace_before_final_stat(name, *args, **kwargs):
        nonlocal calls, changed
        if name == target.name:
            calls += 1
            if calls == 2:
                target.rename(tmp_path / "old-source")
                if mutation == "replace":
                    target.write_bytes(b"{}\n")
                changed = True
        return original_stat(name, *args, **kwargs)

    if mutation == "growth":
        monkeypatch.setattr(contracts.os, "fdopen", grow_before_read)
    else:
        monkeypatch.setattr(contracts.os, "stat", replace_before_final_stat)
    with pytest.raises(
        contracts.CapturedContractError, match="byte limit|changed while reading"
    ):
        contracts.read_file(target, 32)
    assert changed


@pytest.mark.parametrize(
    "mutation", ["missing-directory", "missing-payload", "missing-manifest"]
)
def test_incomplete_inventory_is_rejected_without_payload_reads(
    tmp_path, monkeypatch, mutation
):
    pack = _pack(tmp_path)
    target = {
        "missing-directory": pack / "inputs",
        "missing-payload": pack / "records/baseline.json",
        "missing-manifest": pack / "manifest.json",
    }[mutation]
    target.rename(tmp_path / "removed")
    read = Mock(wraps=contracts._read_at)
    monkeypatch.setattr(contracts, "_read_at", read)
    with pytest.raises(
        contracts.CapturedContractError, match="incomplete|manifest is missing"
    ):
        with contracts.captured_snapshot(pack):
            pytest.fail("incomplete inventory accepted")
    assert {call.args[1] for call in read.call_args_list} == {"manifest.json"}


@pytest.mark.parametrize("mutation", ["duplicate", "vanished"])
def test_inventory_readdir_races_fail_closed(tmp_path, monkeypatch, mutation):
    pack = _pack(tmp_path)
    original = os.scandir
    changed = False

    @contextmanager
    def unstable_entries(descriptor):
        nonlocal changed
        with original(descriptor) as entries:
            collected = list(entries)
            for entry in collected:
                if entry.name == "baseline.json":
                    if mutation == "duplicate":
                        # Directory iteration can repeat entries under concurrent renames.
                        collected.append(entry)
                    else:
                        (pack / "records/baseline.json").unlink()
                    changed = True
                    break
            yield iter(collected)

    monkeypatch.setattr(contracts.os, "scandir", unstable_entries)
    with pytest.raises(
        contracts.CapturedIntegrityError, match="repeated an entry|changed during scan"
    ):
        with contracts.captured_snapshot(pack):
            pytest.fail("unstable inventory accepted")
    assert changed


@pytest.mark.parametrize("open_number", [1, 2])
def test_subdirectory_replaced_between_inventory_and_open(
    tmp_path, monkeypatch, open_number
):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    original = contracts._open_snapshot_directory
    calls = 0

    def replace(root, name):
        nonlocal calls
        if name == "records":
            calls += 1
            if calls == open_number:
                (pack / "records").rename(tmp_path / "old-records")
                (pack / "records").mkdir()
        return original(root, name)

    monkeypatch.setattr(contracts, "_open_snapshot_directory", replace)
    with pytest.raises(
        verification.CapturedVerificationError,
        match="directory.*changed|directory source was replaced",
    ):
        verification.verify_captured_evidence(pack, **options)
    authenticated = _authenticate(pack, options)
    assert authenticated.ok
    assert authenticated.statement["verdict"]["integrity_ok"] is False


@pytest.mark.parametrize("mutation", ["removed", "symlink", "new-inode"])
def test_payload_changes_after_inventory_cannot_reach_replay(
    tmp_path, monkeypatch, mutation
):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    original = contracts._read_at
    target = pack / "records/baseline.json"

    def replace(parent, name, limit):
        if name == "baseline.json":
            displaced = tmp_path / "baseline.json"
            target.rename(displaced)
            if mutation == "symlink":
                target.symlink_to(displaced)
            elif mutation == "new-inode":
                target.write_bytes(displaced.read_bytes())
        return original(parent, name, limit)

    monkeypatch.setattr(contracts, "_read_at", replace)
    replay = Mock(side_effect=AssertionError("unstable payload reached replay"))
    monkeypatch.setattr(verification, "compare_runs", replay)
    with pytest.raises(
        verification.CapturedVerificationError, match="changed during snapshot"
    ):
        verification.verify_captured_evidence(pack, **options)
    replay.assert_not_called()
    authenticated = _authenticate(pack, options)
    assert authenticated.ok
    assert authenticated.statement["replay_status"] == "not_started"


def test_late_payload_mutation_revokes_positive_receipt(tmp_path, monkeypatch):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    original = verification.compare_runs

    def mutate_after_replay(*args, **kwargs):
        result = original(*args, **kwargs)
        with (pack / "records/baseline.json").open("ab") as output:
            output.write(b" ")
        return result

    monkeypatch.setattr(verification, "compare_runs", mutate_after_replay)
    with pytest.raises(
        verification.CapturedVerificationError, match="inventory changed after capture"
    ):
        verification.verify_captured_evidence(pack, **options)
    authenticated = _authenticate(pack, options)
    assert authenticated.ok
    assert authenticated.statement["replay_status"] == "completed"
    assert authenticated.statement["verdict"]["integrity_ok"] is False
    assert authenticated.statement["scoring_assurance"] is None


def test_manifest_disappearing_after_detection_has_no_receipt(tmp_path, monkeypatch):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    original = contracts._inventory

    def remove_manifest(root, signed):
        (pack / "manifest.json").unlink()
        return original(root, signed)

    monkeypatch.setattr(contracts, "_inventory", remove_manifest)
    with pytest.raises(
        verification.CapturedVerificationError, match="manifest changed after detection"
    ):
        verification.verify_captured_evidence(pack, **options)
    assert not options["receipt_path"].exists()


@pytest.mark.parametrize("path", [".", "..", "/"])
def test_atomic_write_requires_a_filename(path):
    with pytest.raises(contracts.CapturedContractError, match="must name a file"):
        contracts.atomic_write(Path(path), b"never published")


def test_atomic_write_rejects_parent_traversal_before_creating_output(tmp_path):
    with pytest.raises(contracts.CapturedContractError, match="parent traversal"):
        contracts.atomic_write(tmp_path / "new" / ".." / "receipt.json", b"receipt")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mutation", ["removed", "replaced"])
def test_atomic_rollback_preserves_original_failure_and_foreign_output(
    tmp_path, monkeypatch, mutation
):
    destination = tmp_path / "receipt.json"
    original = os.fsync
    error = OSError(errno.EIO, "directory synchronization failed")

    def change_published_output(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            destination.unlink()
            if mutation == "replaced":
                destination.write_bytes(b"another writer's output")
            raise error
        original(descriptor)

    monkeypatch.setattr(contracts.os, "fsync", change_published_output)
    with pytest.raises(OSError) as failure:
        contracts.atomic_write(destination, b"our complete output")
    assert failure.value is error
    if mutation == "replaced":
        assert destination.read_bytes() == b"another writer's output"
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob(".captured-*"))


@pytest.mark.parametrize(
    "operation",
    ["dup", "mkdir", "write-open", "file-fsync", "directory-fsync", "publish"],
)
def test_publication_io_failure_cleans_staging_and_descriptors(
    tmp_path, monkeypatch, publication_args, operation
):
    destination = tmp_path / "output" / "evidence"
    error = OSError(errno.EIO, "injected publication failure")
    duplicates = []
    original_dup, original_mkdir = os.dup, os.mkdir
    original_open, original_fsync = os.open, os.fsync

    def fail_dup(descriptor):
        if operation == "dup":
            raise error
        result = original_dup(descriptor)
        duplicates.append(result)
        return result

    def fail_mkdir(name, *args, **kwargs):
        if operation == "mkdir" and str(name).startswith(".captured-evidence-"):
            raise error
        return original_mkdir(name, *args, **kwargs)

    def fail_open(name, flags, *args, **kwargs):
        if operation == "write-open" and flags & os.O_WRONLY:
            raise error
        return original_open(name, flags, *args, **kwargs)

    def fail_fsync(descriptor):
        directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        if operation == ("directory-fsync" if directory else "file-fsync"):
            raise error
        return original_fsync(descriptor)

    monkeypatch.setattr(publication.os, "dup", fail_dup)
    monkeypatch.setattr(publication.os, "mkdir", fail_mkdir)
    monkeypatch.setattr(publication.os, "open", fail_open)
    monkeypatch.setattr(publication.os, "fsync", fail_fsync)
    if operation == "publish":
        monkeypatch.setattr(
            publication,
            "publish_directory_no_replace",
            Mock(
                side_effect=AtomicDirectoryPublicationError("unsupported atomic rename")
            ),
        )
    with pytest.raises(publication.CapturedEvidenceError, match="could not publish"):
        publication.publish_captured_evidence(destination, **publication_args)
    assert not destination.exists()
    assert not list(destination.parent.iterdir())
    for descriptor in duplicates:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(descriptor)


def test_destination_created_at_publication_instant_is_not_clobbered(
    tmp_path, monkeypatch, publication_args
):
    destination = tmp_path / "evidence"
    original = publication.publish_directory_no_replace

    def collide(staging, target):
        target.mkdir()
        (target / "keep").write_bytes(b"concurrent writer")
        return original(staging, target)

    monkeypatch.setattr(publication, "publish_directory_no_replace", collide)
    with pytest.raises(publication.CapturedEvidenceError, match="already exists"):
        publication.publish_captured_evidence(destination, **publication_args)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == {
        "keep": b"concurrent writer"
    }
    assert not list(tmp_path.glob(".captured-evidence-*"))


def test_staging_name_collision_does_not_remove_foreign_files(
    tmp_path, monkeypatch, publication_args
):
    staging = tmp_path / (".captured-evidence-" + "0" * 32)
    staging.mkdir()
    marker = staging / "keep"
    marker.write_bytes(b"foreign staging content")
    monkeypatch.setattr(publication.secrets, "token_hex", Mock(return_value="0" * 32))
    destination = tmp_path / "evidence"
    with pytest.raises(publication.CapturedEvidenceError, match="could not publish"):
        publication.publish_captured_evidence(destination, **publication_args)
    assert not destination.exists()
    assert marker.read_bytes() == b"foreign staging content"
