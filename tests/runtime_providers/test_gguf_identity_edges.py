from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import pytest

from invarlock.runtime_providers import gguf_identity as identity


def _error(message: str):
    return pytest.raises(identity.GGUFIdentityError, match=message)


def _reader(tmp_path: Path, payload: bytes) -> tuple[identity._HeaderReader, int]:
    path = tmp_path / f"payload-{len(list(tmp_path.iterdir()))}"
    path.write_bytes(payload)
    descriptor = os.open(path, os.O_RDONLY)
    return identity._HeaderReader(descriptor, file_size=len(payload)), descriptor


def test_header_reader_bounds_and_early_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader, descriptor = _reader(tmp_path, b"a")
    try:
        with _error("truncated GGUF field"):
            reader.read_exact(-1, label="field")
        with _error("truncated GGUF field"):
            reader.read_exact(2, label="field")
        monkeypatch.setattr(identity, "_MAX_HEADER_BYTES", 0)
        with _error("header size"):
            reader.read_exact(1, label="field")
    finally:
        os.close(descriptor)


def test_artifact_path_opening_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InvalidPath:
        def __fspath__(self) -> str:
            raise OSError("invalid")

    with _error("path is invalid"):
        identity._open_regular_without_symlinks(InvalidPath())  # type: ignore[arg-type]
    with _error("identify a regular file"):
        identity._open_regular_without_symlinks(Path("/"))

    monkeypatch.delattr(identity.os, "O_NOFOLLOW")
    with _error("nofollow artifact opening"):
        identity._open_regular_without_symlinks(tmp_path / "value.gguf")


def test_artifact_root_and_file_open_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "value.gguf"
    path.write_bytes(b"GGUF")
    original_open = identity.os.open

    monkeypatch.setattr(
        identity.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("closed")),
    )
    with _error("root cannot be opened"):
        identity._open_regular_without_symlinks(path)

    monkeypatch.setattr(identity.os, "open", original_open)
    original_stat = identity.os.stat
    monkeypatch.setattr(
        identity.os,
        "stat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("gone")),
    )
    with _error("artifact is unavailable"):
        identity._open_regular_without_symlinks(path)
    monkeypatch.setattr(identity.os, "stat", original_stat)


@pytest.mark.parametrize(
    ("payload", "decoder", "message"),
    [
        (struct.pack("<Q", 1) + b"\xff", "key", "must be ASCII"),
        (struct.pack("<Q", 3) + b"Bad", "key", "not canonical"),
        (struct.pack("<Q", 1) + b"\xff", "tensor", "must be UTF-8"),
        (struct.pack("<Q", 0), "tensor", "non-empty and printable"),
        (struct.pack("<Q", 1) + b"\n", "tensor", "non-empty and printable"),
    ],
)
def test_metadata_keys_and_tensor_names_are_canonical(
    tmp_path: Path, payload: bytes, decoder: str, message: str
) -> None:
    reader, descriptor = _reader(tmp_path, payload)
    try:
        with _error(message):
            if decoder == "key":
                identity._decode_metadata_key(reader)
            else:
                identity._decode_tensor_name(reader)
    finally:
        os.close(descriptor)


def test_metadata_value_scalar_and_array_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    float_reader, float_fd = _reader(tmp_path, struct.pack("<f", math.nan))
    try:
        with _error("floating metadata must be finite"):
            identity._read_metadata_value(
                float_reader, 6, budget=identity._ParseBudget()
            )
    finally:
        os.close(float_fd)

    array_reader, array_fd = _reader(tmp_path, struct.pack("<I", 99))
    try:
        with _error("unsupported value type"):
            identity._read_metadata_value(
                array_reader, 9, budget=identity._ParseBudget()
            )
    finally:
        os.close(array_fd)

    nested_reader, nested_fd = _reader(tmp_path, b"")
    try:
        with _error("array nesting"):
            identity._read_metadata_value(
                nested_reader,
                9,
                budget=identity._ParseBudget(),
                depth=identity._MAX_ARRAY_DEPTH,
            )
    finally:
        os.close(nested_fd)

    monkeypatch.setattr(identity, "_MAX_TOTAL_ARRAY_ITEMS", 0)
    items_reader, items_fd = _reader(tmp_path, struct.pack("<IQ", 0, 1) + b"\x00")
    try:
        with _error("total metadata array items"):
            identity._read_metadata_value(
                items_reader, 9, budget=identity._ParseBudget()
            )
    finally:
        os.close(items_fd)


def test_tensor_extent_overflow_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(identity, "_MAX_FILE_BYTES", 1)
    with _error("byte extent"):
        identity._tensor_byte_length(dimensions=[1], element_count=1, tensor_type=0)


def test_stream_hash_detects_truncation_and_growth(
    tmp_path: Path,
) -> None:
    short = tmp_path / "short"
    short.write_bytes(b"a")
    descriptor = os.open(short, os.O_RDONLY)
    try:
        with _error("truncated while hashing"):
            identity._stream_file_sha256(descriptor, 2)
    finally:
        os.close(descriptor)

    long = tmp_path / "long"
    long.write_bytes(b"ab")
    descriptor = os.open(long, os.O_RDONLY)
    try:
        with _error("grew while hashing"):
            identity._stream_file_sha256(descriptor, 1)
    finally:
        os.close(descriptor)
