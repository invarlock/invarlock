from __future__ import annotations

import hashlib
import os
import socket
import struct
from pathlib import Path

import pytest

from invarlock.runtime_providers import gguf_identity


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _metadata(key: str, value_type: int, value: bytes) -> bytes:
    return _string(key) + struct.pack("<I", value_type) + value


def _string_value(value: str) -> bytes:
    return _string(value)


def _string_array(values: list[str]) -> bytes:
    return struct.pack("<IQ", 8, len(values)) + b"".join(
        _string(value) for value in values
    )


def _tensor(
    name: str,
    dimensions: tuple[int, ...] = (2, 2),
    tensor_type: int = 0,
    offset: int = 0,
) -> bytes:
    return (
        _string(name)
        + struct.pack("<I", len(dimensions))
        + struct.pack("<" + "Q" * len(dimensions), *dimensions)
        + struct.pack("<IQ", tensor_type, offset)
    )


def _fixture(
    *,
    version: int = 3,
    architecture: str = "llama",
    alignment: int = 32,
    tokens: list[str] | None = None,
    tensors: list[bytes] | None = None,
    extra_metadata: list[bytes] | None = None,
    tensor_data: bytes | None = None,
) -> bytes:
    metadata = [
        _metadata("general.architecture", 8, _string_value(architecture)),
        _metadata("general.alignment", 4, struct.pack("<I", alignment)),
        _metadata("tokenizer.ggml.model", 8, _string_value("llama")),
        _metadata(
            "tokenizer.ggml.tokens",
            9,
            _string_array(tokens if tokens is not None else ["a", "b"]),
        ),
    ]
    metadata.extend(extra_metadata or [])
    tensor_infos = tensors if tensors is not None else [_tensor("token_embd.weight")]
    header = (
        b"GGUF"
        + struct.pack("<IQQ", version, len(tensor_infos), len(metadata))
        + b"".join(metadata)
        + b"".join(tensor_infos)
    )
    padding_alignment = alignment if alignment >= 1 else 1
    padding = b"\x00" * (
        (padding_alignment - len(header) % padding_alignment) % padding_alignment
    )
    return header + padding + (tensor_data if tensor_data is not None else b"\x00" * 16)


@pytest.mark.parametrize("version", [2, 3])
def test_read_gguf_artifact_identity_accepts_bounded_v2_v3(
    tmp_path: Path, version: int
) -> None:
    payload = _fixture(version=version)
    first = tmp_path / "private-host-model-name.gguf"
    second_dir = tmp_path / "other"
    second_dir.mkdir()
    second = second_dir / "renamed.gguf"
    first.write_bytes(payload)
    second.write_bytes(payload)

    identity = gguf_identity.read_gguf_artifact_identity(first)
    renamed = gguf_identity.read_gguf_artifact_identity(second)
    digest = hashlib.sha256(payload).hexdigest()

    assert identity == renamed
    assert identity.sha256 == digest
    assert identity.byte_length == len(payload)
    assert identity.artifact_name == f"gguf-sha256-{digest}.gguf"
    assert "private-host-model-name" not in identity.artifact_name
    expected_derived = (
        "e85c484be8bfc8a8b8a4d4c288e65fd091b8d991166845ccf9a7be0d3113ce69",
        "7a1d4d2e37ac12ae80a4015007ffa8429b128ef72ac8e8615d148a395ba130ca",
        "44025af5d76058199ecb6ca148b0148b7bf9874686088677d0998652532f8b1b",
    )
    assert (
        identity.gguf_metadata_sha256,
        identity.tensor_inventory_sha256,
        identity.tokenizer_metadata_sha256,
    ) == expected_derived
    for derived in expected_derived:
        assert len(derived) == 64
        assert set(derived) <= set("0123456789abcdef")


def test_gguf_identity_digests_partition_metadata_tokenizer_and_tensors(
    tmp_path: Path,
) -> None:
    payloads = {
        "base": _fixture(),
        "metadata": _fixture(architecture="mistral"),
        "tokenizer": _fixture(tokens=["a", "changed"]),
        "tensor": _fixture(tensors=[_tensor("output.weight")]),
    }
    identities = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.gguf"
        path.write_bytes(payload)
        identities[name] = gguf_identity.read_gguf_artifact_identity(path)

    base = identities["base"]
    assert identities["metadata"].gguf_metadata_sha256 != base.gguf_metadata_sha256
    assert identities["metadata"].tokenizer_metadata_sha256 == (
        base.tokenizer_metadata_sha256
    )
    assert identities["metadata"].tensor_inventory_sha256 == (
        base.tensor_inventory_sha256
    )
    assert identities["tokenizer"].gguf_metadata_sha256 != base.gguf_metadata_sha256
    assert identities["tokenizer"].tokenizer_metadata_sha256 != (
        base.tokenizer_metadata_sha256
    )
    assert identities["tokenizer"].tensor_inventory_sha256 == (
        base.tensor_inventory_sha256
    )
    assert identities["tensor"].gguf_metadata_sha256 == base.gguf_metadata_sha256
    assert identities["tensor"].tokenizer_metadata_sha256 == (
        base.tokenizer_metadata_sha256
    )
    assert identities["tensor"].tensor_inventory_sha256 != (
        base.tensor_inventory_sha256
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"BAD!" + _fixture()[4:], "magic"),
        (b"GGUF" + struct.pack("<IQQ", 1, 0, 0), "version"),
        (b"GGUF" + struct.pack(">IQQ", 3, 0, 0), "version or endian"),
        (_fixture()[:-1], "tensor data"),
        (b"GGUF" + struct.pack("<IQQ", 3, 262_145, 0), "tensor count"),
        (b"GGUF" + struct.pack("<IQQ", 3, 0, 16_385), "metadata count"),
    ],
)
def test_gguf_identity_rejects_malformed_header(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    path = tmp_path / "bad.gguf"
    path.write_bytes(payload)

    with pytest.raises(gguf_identity.GGUFIdentityError, match=message):
        gguf_identity.read_gguf_artifact_identity(path)


def test_gguf_identity_rejects_truncated_and_unsupported_metadata(
    tmp_path: Path,
) -> None:
    cases = {
        "truncated": b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + struct.pack("<Q", 4)
        + b"key",
        "unsupported": b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + _metadata("general.bad", 99, b""),
        "oversized-string": b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + struct.pack("<Q", 65_536),
        "oversized-array": b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + _string("tokenizer.ggml.tokens")
        + struct.pack("<IIQ", 9, 8, 2_000_001),
        "oversized-value-string": b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + _string("general.description")
        + struct.pack("<IQ", 8, 16 * 1024**2 + 1),
        "invalid-bool": b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + _metadata("general.flag", 7, b"\x02"),
    }
    for name, payload in cases.items():
        path = tmp_path / f"{name}.gguf"
        path.write_bytes(payload)
        with pytest.raises(gguf_identity.GGUFIdentityError):
            gguf_identity.read_gguf_artifact_identity(path)


def test_gguf_identity_rejects_duplicate_metadata_and_tensor_names(
    tmp_path: Path,
) -> None:
    duplicate_metadata = _fixture(
        extra_metadata=[_metadata("general.architecture", 8, _string_value("llama"))]
    )
    duplicate_tensors = _fixture(
        tensors=[_tensor("same.weight", offset=0), _tensor("same.weight", offset=32)]
    )

    for name, payload, message in (
        ("metadata", duplicate_metadata, "duplicate metadata"),
        ("tensor", duplicate_tensors, "duplicate tensor"),
    ):
        path = tmp_path / f"duplicate-{name}.gguf"
        path.write_bytes(payload)
        with pytest.raises(gguf_identity.GGUFIdentityError, match=message):
            gguf_identity.read_gguf_artifact_identity(path)


def test_gguf_identity_rejects_invalid_alignment_and_tensor_descriptor(
    tmp_path: Path,
) -> None:
    bad_alignment = _fixture(alignment=7)
    non_power_of_two_alignment = _fixture(alignment=24)
    bad_dimension = _fixture(tensors=[_tensor("weight", dimensions=())])
    bad_offset = _fixture(tensors=[_tensor("weight", offset=1)])
    for name, payload, message in (
        ("alignment", bad_alignment, "general.alignment"),
        (
            "non-power-of-two-alignment",
            non_power_of_two_alignment,
            "general.alignment",
        ),
        ("dimension", bad_dimension, "dimension count"),
        ("offset", bad_offset, "not aligned"),
    ):
        path = tmp_path / f"bad-{name}.gguf"
        path.write_bytes(payload)
        with pytest.raises(gguf_identity.GGUFIdentityError, match=message):
            gguf_identity.read_gguf_artifact_identity(path)


def test_gguf_identity_rejects_unknown_and_removed_tensor_types(
    tmp_path: Path,
) -> None:
    for tensor_type in (4, 5, 31, 33, 36, 38, 43, 999):
        path = tmp_path / f"tensor-type-{tensor_type}.gguf"
        path.write_bytes(
            _fixture(
                tensors=[_tensor("weight", dimensions=(1,), tensor_type=tensor_type)],
                tensor_data=b"\x00",
            )
        )
        with pytest.raises(gguf_identity.GGUFIdentityError, match="tensor type"):
            gguf_identity.read_gguf_artifact_identity(path)


def test_gguf_identity_validates_quantized_block_shape_and_exact_extent(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid-q4-0.gguf"
    valid.write_bytes(
        _fixture(
            tensors=[_tensor("weight", dimensions=(32, 2), tensor_type=2)],
            tensor_data=b"\x00" * 36,
        )
    )
    identity = gguf_identity.read_gguf_artifact_identity(valid)
    assert identity.byte_length == len(valid.read_bytes())

    invalid_row = tmp_path / "invalid-q4-0-row.gguf"
    invalid_row.write_bytes(
        _fixture(
            tensors=[_tensor("weight", dimensions=(31, 2), tensor_type=2)],
            tensor_data=b"\x00" * 36,
        )
    )
    with pytest.raises(gguf_identity.GGUFIdentityError, match="block size"):
        gguf_identity.read_gguf_artifact_identity(invalid_row)

    truncated = tmp_path / "truncated-q4-0.gguf"
    truncated.write_bytes(
        _fixture(
            tensors=[_tensor("weight", dimensions=(32, 2), tensor_type=2)],
            tensor_data=b"\x00" * 35,
        )
    )
    with pytest.raises(gguf_identity.GGUFIdentityError, match="truncated"):
        gguf_identity.read_gguf_artifact_identity(truncated)


@pytest.mark.parametrize(
    ("tensor_type", "block_size", "encoded_block_bytes"),
    [
        (0, 1, 4),
        (1, 1, 2),
        (2, 32, 18),
        (3, 32, 20),
        (6, 32, 22),
        (7, 32, 24),
        (8, 32, 34),
        (9, 32, 40),
        (10, 256, 84),
        (11, 256, 110),
        (12, 256, 144),
        (13, 256, 176),
        (14, 256, 210),
        (15, 256, 292),
        (16, 256, 66),
        (17, 256, 74),
        (18, 256, 98),
        (19, 256, 50),
        (20, 32, 18),
        (21, 256, 110),
        (22, 256, 82),
        (23, 256, 136),
        (24, 1, 1),
        (25, 1, 2),
        (26, 1, 4),
        (27, 1, 8),
        (28, 1, 8),
        (29, 256, 56),
        (30, 1, 2),
        (34, 256, 54),
        (35, 256, 66),
        (39, 32, 17),
        (40, 64, 36),
        (41, 128, 18),
        (42, 64, 18),
    ],
)
def test_gguf_identity_enforces_each_supported_tensor_type_extent(
    tmp_path: Path,
    tensor_type: int,
    block_size: int,
    encoded_block_bytes: int,
) -> None:
    valid = tmp_path / f"valid-{tensor_type}.gguf"
    valid.write_bytes(
        _fixture(
            tensors=[
                _tensor(
                    "weight",
                    dimensions=(block_size,),
                    tensor_type=tensor_type,
                )
            ],
            tensor_data=b"\x00" * encoded_block_bytes,
        )
    )
    gguf_identity.read_gguf_artifact_identity(valid)

    truncated = tmp_path / f"truncated-{tensor_type}.gguf"
    truncated.write_bytes(
        _fixture(
            tensors=[
                _tensor(
                    "weight",
                    dimensions=(block_size,),
                    tensor_type=tensor_type,
                )
            ],
            tensor_data=b"\x00" * (encoded_block_bytes - 1),
        )
    )
    with pytest.raises(gguf_identity.GGUFIdentityError, match="truncated"):
        gguf_identity.read_gguf_artifact_identity(truncated)


def test_gguf_identity_rejects_overlapping_tensor_ranges(tmp_path: Path) -> None:
    overlapping = tmp_path / "overlapping.gguf"
    overlapping.write_bytes(
        _fixture(
            tensors=[
                _tensor("large.weight", dimensions=(8, 2), offset=0),
                _tensor("overlap.weight", dimensions=(2, 2), offset=32),
            ],
            tensor_data=b"\x00" * 64,
        )
    )
    with pytest.raises(gguf_identity.GGUFIdentityError, match="overlap"):
        gguf_identity.read_gguf_artifact_identity(overlapping)

    non_overlapping = tmp_path / "non-overlapping.gguf"
    non_overlapping.write_bytes(
        _fixture(
            tensors=[
                _tensor("first.weight", dimensions=(2, 2), offset=0),
                _tensor("second.weight", dimensions=(2, 2), offset=32),
            ],
            tensor_data=b"\x00" * 48,
        )
    )
    gguf_identity.read_gguf_artifact_identity(non_overlapping)


def test_gguf_identity_rejects_non_regular_and_symlink_inputs(tmp_path: Path) -> None:
    regular = tmp_path / "model.gguf"
    regular.write_bytes(_fixture())
    symlink = tmp_path / "link.gguf"
    symlink.symlink_to(regular)
    directory = tmp_path / "directory.gguf"
    directory.mkdir()
    fifo = tmp_path / "fifo.gguf"
    os.mkfifo(fifo)
    for path in (symlink, directory, fifo, Path("/dev/null")):
        with pytest.raises(gguf_identity.GGUFIdentityError):
            gguf_identity.read_gguf_artifact_identity(path)


def test_gguf_identity_rejects_socket_input() -> None:
    socket_path = Path.cwd() / f".invarlock-gguf-{os.getpid()}.sock"
    socket_path.unlink(missing_ok=True)
    unix_socket = socket.socket(socket.AF_UNIX)
    try:
        try:
            unix_socket.bind(str(socket_path))
        except PermissionError:
            pytest.skip("sandbox does not permit creating Unix-domain sockets")
        with pytest.raises(gguf_identity.GGUFIdentityError):
            gguf_identity.read_gguf_artifact_identity(socket_path)
    finally:
        unix_socket.close()
        socket_path.unlink(missing_ok=True)


def test_gguf_identity_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "model.gguf").write_bytes(_fixture())
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)

    with pytest.raises(gguf_identity.GGUFIdentityError, match="symlink"):
        gguf_identity.read_gguf_artifact_identity(linked / "model.gguf")


def test_gguf_identity_rejects_file_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(_fixture())
    original = gguf_identity._stream_file_sha256

    def mutate_after_hash(descriptor: int, expected_size: int) -> str:
        digest = original(descriptor, expected_size)
        with path.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            original_byte = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([original_byte[0] ^ 0xFF]))
            handle.flush()
            os.fsync(handle.fileno())
        return digest

    monkeypatch.setattr(gguf_identity, "_stream_file_sha256", mutate_after_hash)

    with pytest.raises(gguf_identity.GGUFIdentityError, match="changed"):
        gguf_identity.read_gguf_artifact_identity(path)


def test_gguf_identity_rejects_oversized_tensor_name(tmp_path: Path) -> None:
    payload = b"GGUF" + struct.pack("<IQQQ", 3, 1, 0, 65)
    path = tmp_path / "oversized-tensor-name.gguf"
    path.write_bytes(payload)

    with pytest.raises(gguf_identity.GGUFIdentityError, match="tensor name length"):
        gguf_identity.read_gguf_artifact_identity(path)


def test_gguf_identity_rejects_path_replacement_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.gguf"
    replacement = tmp_path / "replacement.gguf"
    path.write_bytes(_fixture())
    replacement.write_bytes(_fixture(tokens=["replacement"]))
    original = gguf_identity._stream_file_sha256

    def replace_after_hash(descriptor: int, expected_size: int) -> str:
        digest = original(descriptor, expected_size)
        replacement.replace(path)
        return digest

    monkeypatch.setattr(gguf_identity, "_stream_file_sha256", replace_after_hash)

    with pytest.raises(gguf_identity.GGUFIdentityError, match="changed|replaced"):
        gguf_identity.read_gguf_artifact_identity(path)


def test_gguf_identity_rejects_bounded_file_and_header_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(_fixture())
    monkeypatch.setattr(gguf_identity, "_MAX_FILE_BYTES", path.stat().st_size - 1)
    with pytest.raises(gguf_identity.GGUFIdentityError, match="file size"):
        gguf_identity.read_gguf_artifact_identity(path)

    monkeypatch.setattr(gguf_identity, "_MAX_FILE_BYTES", 1024 * 1024)
    monkeypatch.setattr(gguf_identity, "_MAX_HEADER_BYTES", 32)
    with pytest.raises(gguf_identity.GGUFIdentityError, match="header size"):
        gguf_identity.read_gguf_artifact_identity(path)
