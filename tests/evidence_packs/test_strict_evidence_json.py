from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import invarlock.evidence_pack_json as strict_json
from invarlock.evidence_pack_json import (
    StrictJsonError,
    copy_regular_file_snapshot,
    load_json,
    load_json_object,
    parse_json_bytes,
    read_json_object_snapshot,
    read_jsonl_snapshot,
    read_regular_file_bytes,
    sha256_prefixed,
)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"key":1,"key":2}', "duplicate key"),
        (b'{"value":NaN}', "non-standard constant"),
        (b'{"value":Infinity}', "non-standard constant"),
        (b'{"value":1e9999}', "non-finite number"),
        (b'{"value":', "not valid JSON"),
        (b'{"value":"\xff"}', "not UTF-8 JSON"),
    ],
)
def test_strict_json_rejects_ambiguous_or_nonstandard_bytes(
    payload: bytes, message: str
) -> None:
    with pytest.raises(StrictJsonError, match=message):
        parse_json_bytes(payload, label="evidence input")


def test_strict_json_rejects_excessive_nesting_without_leaking_recursion() -> None:
    payload = (b'{"x":' * 100_000) + b"null" + (b"}" * 100_000)

    with pytest.raises(StrictJsonError, match="not valid JSON"):
        parse_json_bytes(payload, label="nested observation")


def test_regular_file_reader_is_bounded_and_rejects_unsafe_nodes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.json"
    source.write_bytes(b'{"ok":true}')

    assert read_regular_file_bytes(source, label="input") == b'{"ok":true}'
    assert (
        read_regular_file_bytes(source, label="input", max_bytes=11) == b'{"ok":true}'
    )
    for invalid_limit in (0, -1, True, 1.5):
        with pytest.raises(StrictJsonError, match="positive integer"):
            read_regular_file_bytes(
                source,
                label="input",
                max_bytes=invalid_limit,  # type: ignore[arg-type]
            )
    with pytest.raises(StrictJsonError, match="size limit"):
        read_regular_file_bytes(source, label="input", max_bytes=10)

    link = tmp_path / "input-link.json"
    link.symlink_to(source)
    with pytest.raises(StrictJsonError, match="must not be a symlink"):
        read_regular_file_bytes(link, label="input")
    with pytest.raises(StrictJsonError, match="regular file"):
        read_regular_file_bytes(tmp_path, label="input")
    with pytest.raises(StrictJsonError, match="unavailable"):
        read_regular_file_bytes(tmp_path / "missing.json", label="input")


def test_snapshot_copy_is_exact_no_clobber_and_preserves_requested_mode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"authenticated bytes")
    destination = tmp_path / "snapshot.bin"

    copy_regular_file_snapshot(source, destination, label="artifact", mode=0o440)

    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o440
    with pytest.raises(StrictJsonError, match="could not be copied safely"):
        copy_regular_file_snapshot(source, destination, label="artifact")

    linked_source = tmp_path / "linked-source.bin"
    linked_source.symlink_to(source)
    with pytest.raises(StrictJsonError, match="must not be a symlink"):
        copy_regular_file_snapshot(
            linked_source, tmp_path / "unsafe.bin", label="artifact"
        )


def test_snapshot_copy_enforces_its_streaming_byte_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"12345")
    destination = tmp_path / "snapshot.bin"

    with pytest.raises(StrictJsonError, match="4-byte size limit"):
        copy_regular_file_snapshot(
            source,
            destination,
            label="artifact",
            max_bytes=4,
        )

    assert not destination.exists()

    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    copy_regular_file_snapshot(
        empty,
        destination,
        label="empty artifact",
        max_bytes=0,
    )
    assert destination.read_bytes() == b""


def test_json_object_snapshot_binds_exact_bytes_and_requires_an_object(
    tmp_path: Path,
) -> None:
    source = tmp_path / "object.json"
    exact = b'{"candidate":"subject","score":0.5}\n'
    source.write_bytes(exact)

    snapshot, payload = read_json_object_snapshot(source, label="candidate")

    assert snapshot == exact
    assert payload == {"candidate": "subject", "score": 0.5}
    assert load_json(source, label="candidate") == payload
    assert load_json_object(source, label="candidate") == payload
    assert sha256_prefixed(snapshot) == "sha256:" + hashlib.sha256(exact).hexdigest()

    source.write_text("[]\n", encoding="utf-8")
    with pytest.raises(StrictJsonError, match="JSON object"):
        read_json_object_snapshot(source, label="candidate")


def test_jsonl_snapshot_rejects_empty_blank_and_ambiguous_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"id":"one"}\n{"id":"two"}\n')

    exact, records = read_jsonl_snapshot(path, label="paired records")

    assert exact == path.read_bytes()
    assert records == [{"id": "one"}, {"id": "two"}]

    path.write_bytes(b"")
    with pytest.raises(StrictJsonError, match="no JSON records"):
        read_jsonl_snapshot(path, label="paired records")
    path.write_bytes(b'{"id":"one"}\n\n')
    with pytest.raises(StrictJsonError, match="blank row at line 2"):
        read_jsonl_snapshot(path, label="paired records")
    path.write_bytes(b'{"id":"one","id":"two"}\n')
    with pytest.raises(StrictJsonError, match="duplicate key"):
        read_jsonl_snapshot(path, label="paired records")


def test_regular_file_reader_rejects_named_pipe_without_opening_it(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are unavailable")
    fifo = tmp_path / "unsafe.fifo"
    os.mkfifo(fifo)

    with pytest.raises(StrictJsonError, match="regular file"):
        read_regular_file_bytes(fifo, label="input")


def test_regular_file_reader_rejects_metadata_drift_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mutable.json"
    source.write_bytes(b'{"status":"observation"}\n')
    real_fstat = strict_json.os.fstat
    calls = 0

    def drifting_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        file_stat = real_fstat(descriptor)
        if calls >= 2:
            values = list(file_stat)
            values[9] = file_stat.st_ctime + 1
            return os.stat_result(values)
        return file_stat

    monkeypatch.setattr(strict_json.os, "fstat", drifting_fstat)

    with pytest.raises(StrictJsonError, match="changed while being read"):
        read_regular_file_bytes(source, label="observation")


def test_snapshot_copy_removes_partial_destination_when_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mutable.bin"
    source.write_bytes(b"authenticated observation bytes")
    destination = tmp_path / "snapshot.bin"
    real_fstat = strict_json.os.fstat
    calls = 0

    def drifting_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        file_stat = real_fstat(descriptor)
        if calls >= 2:
            values = list(file_stat)
            values[9] = file_stat.st_ctime + 1
            return os.stat_result(values)
        return file_stat

    monkeypatch.setattr(strict_json.os, "fstat", drifting_fstat)

    with pytest.raises(StrictJsonError, match="changed while being copied"):
        copy_regular_file_snapshot(source, destination, label="observation")

    assert not destination.exists()
