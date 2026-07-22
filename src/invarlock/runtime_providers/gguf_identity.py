"""Bounded, dependency-free GGUF v2/v3 artifact identity reader."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path

from invarlock.core.runtime_provider.types import GGUFArtifactIdentity

_MAX_FILE_BYTES = 2 * 1024**4
_MAX_HEADER_BYTES = 512 * 1024**2
_MAX_METADATA_COUNT = 16_384
_MAX_TENSOR_COUNT = 262_144
_MAX_METADATA_KEY_BYTES = 65_535
_MAX_TENSOR_NAME_BYTES = 64
_MAX_STRING_BYTES = 16 * 1024**2
_MAX_ARRAY_ITEMS = 2_000_000
_MAX_TOTAL_ARRAY_ITEMS = 4_000_000
_MAX_ARRAY_DEPTH = 4
_MAX_ALIGNMENT = 1024 * 1024
_MAX_TENSOR_ELEMENTS = (1 << 63) - 1
_HASH_CHUNK_BYTES = 1024 * 1024

_METADATA_KEY = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*(?:\.[a-z0-9]+(?:_[a-z0-9]+)*)*$")
_FIXED_VALUE_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "B",
    10: "Q",
    11: "q",
    12: "d",
}
_STRING_VALUE_TYPE = 8
_ARRAY_VALUE_TYPE = 9
_GGML_TENSOR_BLOCK_SIZES = {
    # GGMLQuantizationType -> (elements per block, encoded bytes per block).
    # Keep this fail-closed table synchronized with llama.cpp's canonical
    # GGML_QUANT_SIZES. Removed or future types are deliberately rejected until
    # their layout is inspected here.
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 40),  # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),  # I8
    25: (1, 2),  # I16
    26: (1, 4),  # I32
    27: (1, 8),  # I64
    28: (1, 8),  # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),  # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),  # MXFP4
    40: (64, 36),  # NVFP4
    41: (128, 18),  # Q1_0
    42: (64, 18),  # Q2_0
}


class GGUFIdentityError(ValueError):
    """Raised when a GGUF file cannot support a secure artifact identity."""


@dataclass(frozen=True)
class _OpenArtifact:
    absolute_path: Path
    descriptor: int
    parent_descriptor: int
    basename: str
    initial_stat: os.stat_result


@dataclass
class _ParseBudget:
    array_items: int = 0


class _HeaderReader:
    def __init__(self, descriptor: int, *, file_size: int) -> None:
        self.descriptor = descriptor
        self.file_size = file_size
        self.position = 0

    def read_exact(self, size: int, *, label: str) -> bytes:
        if size < 0 or self.position + size > min(self.file_size, _MAX_HEADER_BYTES):
            if self.position + size > _MAX_HEADER_BYTES:
                raise GGUFIdentityError("GGUF header size exceeds the configured bound")
            raise GGUFIdentityError(f"truncated GGUF {label}")
        remaining = size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(self.descriptor, remaining)
            if not chunk:
                raise GGUFIdentityError(f"truncated GGUF {label}")
            chunks.append(chunk)
            remaining -= len(chunk)
            self.position += len(chunk)
        return b"".join(chunks)

    def uint32(self, *, label: str) -> int:
        return int(struct.unpack("<I", self.read_exact(4, label=label))[0])

    def uint64(self, *, label: str) -> int:
        return int(struct.unpack("<Q", self.read_exact(8, label=label))[0])


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_regular_without_symlinks(path: str | os.PathLike[str]) -> _OpenArtifact:
    try:
        absolute = Path(os.path.abspath(os.fspath(path)))
    except (TypeError, ValueError, OSError) as exc:
        raise GGUFIdentityError("GGUF artifact path is invalid") from exc
    if absolute.name in {"", ".", ".."}:
        raise GGUFIdentityError("GGUF artifact path must identify a regular file")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise GGUFIdentityError(
            "secure nofollow artifact opening is unavailable on this platform"
        )

    directory_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        directory_descriptor = os.open(absolute.anchor, directory_flags)
    except OSError as exc:
        raise GGUFIdentityError("GGUF artifact root cannot be opened safely") from exc
    try:
        for component in absolute.parts[1:-1]:
            try:
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise GGUFIdentityError(
                    "GGUF artifact path contains a symlink or inaccessible directory"
                ) from exc
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor

        try:
            before = os.stat(
                absolute.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise GGUFIdentityError("GGUF artifact is unavailable") from exc
        if stat.S_ISLNK(before.st_mode):
            raise GGUFIdentityError("GGUF artifact must not be a symlink")
        if not stat.S_ISREG(before.st_mode):
            raise GGUFIdentityError("GGUF artifact must be a stable regular file")

        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        try:
            descriptor = os.open(
                absolute.name,
                file_flags,
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            raise GGUFIdentityError("GGUF artifact cannot be opened safely") from exc
        try:
            opened = os.fstat(descriptor)
        except OSError as exc:
            os.close(descriptor)
            raise GGUFIdentityError("GGUF artifact cannot be inspected safely") from exc
        if not stat.S_ISREG(opened.st_mode) or _stat_identity(before) != _stat_identity(
            opened
        ):
            os.close(descriptor)
            raise GGUFIdentityError("GGUF artifact changed while being opened")
        return _OpenArtifact(
            absolute_path=absolute,
            descriptor=descriptor,
            parent_descriptor=directory_descriptor,
            basename=absolute.name,
            initial_stat=opened,
        )
    except Exception:
        os.close(directory_descriptor)
        raise


def _stream_file_sha256(descriptor: int, expected_size: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = expected_size
    digest = hashlib.sha256()
    while remaining:
        chunk = os.read(descriptor, min(remaining, _HASH_CHUNK_BYTES))
        if not chunk:
            raise GGUFIdentityError("GGUF artifact changed or truncated while hashing")
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise GGUFIdentityError("GGUF artifact changed or grew while hashing")
    return digest.hexdigest()


def _read_length_prefixed_bytes(
    reader: _HeaderReader,
    *,
    maximum: int,
    label: str,
) -> bytes:
    length = reader.uint64(label=f"{label} length")
    if length > maximum:
        raise GGUFIdentityError(f"GGUF {label} length exceeds the configured bound")
    return reader.read_exact(length, label=label)


def _decode_metadata_key(reader: _HeaderReader) -> str:
    encoded = _read_length_prefixed_bytes(
        reader,
        maximum=_MAX_METADATA_KEY_BYTES,
        label="metadata key",
    )
    try:
        key = encoded.decode("ascii")
    except UnicodeDecodeError as exc:
        raise GGUFIdentityError("GGUF metadata key must be ASCII") from exc
    if _METADATA_KEY.fullmatch(key) is None:
        raise GGUFIdentityError("GGUF metadata key is not canonical")
    return key


def _decode_tensor_name(reader: _HeaderReader) -> str:
    encoded = _read_length_prefixed_bytes(
        reader,
        maximum=_MAX_TENSOR_NAME_BYTES,
        label="tensor name",
    )
    try:
        name = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFIdentityError("GGUF tensor name must be UTF-8") from exc
    if not name or any(ord(character) < 32 for character in name):
        raise GGUFIdentityError("GGUF tensor name must be non-empty and printable")
    return name


def _read_metadata_value(
    reader: _HeaderReader,
    value_type: int,
    *,
    budget: _ParseBudget,
    depth: int = 0,
) -> tuple[str, object | None]:
    if depth > _MAX_ARRAY_DEPTH:
        raise GGUFIdentityError("GGUF metadata array nesting exceeds the bound")
    digest = hashlib.sha256()
    digest.update(b"invarlock/gguf-value-v1\x00")
    digest.update(struct.pack("<I", value_type))

    if value_type in _FIXED_VALUE_FORMATS:
        value_format = _FIXED_VALUE_FORMATS[value_type]
        size = struct.calcsize("<" + value_format)
        encoded = reader.read_exact(size, label="metadata scalar")
        value = struct.unpack("<" + value_format, encoded)[0]
        if value_type == 7 and value not in {0, 1}:
            raise GGUFIdentityError("GGUF boolean metadata must be zero or one")
        if value_type in {6, 12} and not math.isfinite(float(value)):
            raise GGUFIdentityError("GGUF floating metadata must be finite")
        digest.update(encoded)
        return digest.hexdigest(), value

    if value_type == _STRING_VALUE_TYPE:
        encoded = _read_length_prefixed_bytes(
            reader,
            maximum=_MAX_STRING_BYTES,
            label="metadata string",
        )
        try:
            encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFIdentityError("GGUF metadata string must be UTF-8") from exc
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        return digest.hexdigest(), None

    if value_type == _ARRAY_VALUE_TYPE:
        if depth >= _MAX_ARRAY_DEPTH:
            raise GGUFIdentityError("GGUF metadata array nesting exceeds the bound")
        element_type = reader.uint32(label="metadata array element type")
        if element_type not in {*_FIXED_VALUE_FORMATS, 8, 9}:
            raise GGUFIdentityError("GGUF metadata array has unsupported value type")
        count = reader.uint64(label="metadata array length")
        if count > _MAX_ARRAY_ITEMS:
            raise GGUFIdentityError("GGUF metadata array length exceeds the bound")
        budget.array_items += count
        if budget.array_items > _MAX_TOTAL_ARRAY_ITEMS:
            raise GGUFIdentityError("GGUF total metadata array items exceed the bound")
        digest.update(struct.pack("<IQ", element_type, count))
        for _ in range(count):
            item_digest, _scalar = _read_metadata_value(
                reader,
                element_type,
                budget=budget,
                depth=depth + 1,
            )
            digest.update(bytes.fromhex(item_digest))
        return digest.hexdigest(), None

    raise GGUFIdentityError(f"GGUF metadata uses unsupported value type {value_type}")


def _canonical_records_sha256(records: list[dict[str, object]]) -> str:
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_byte_length(
    *, dimensions: list[int], element_count: int, tensor_type: int
) -> int:
    layout = _GGML_TENSOR_BLOCK_SIZES.get(tensor_type)
    if layout is None:
        raise GGUFIdentityError(f"GGUF tensor type {tensor_type} is unsupported")
    block_size, encoded_block_bytes = layout
    if dimensions[0] % block_size != 0:
        raise GGUFIdentityError(
            "GGUF quantized tensor row is not a multiple of its block size"
        )
    block_count = element_count // block_size
    if block_count > _MAX_FILE_BYTES // encoded_block_bytes:
        raise GGUFIdentityError("GGUF tensor byte extent exceeds the file bound")
    return block_count * encoded_block_bytes


def _parse_gguf(
    descriptor: int,
    *,
    file_size: int,
) -> tuple[str, str, str]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    reader = _HeaderReader(descriptor, file_size=file_size)
    if reader.read_exact(4, label="magic") != b"GGUF":
        raise GGUFIdentityError("GGUF magic is invalid")
    version = reader.uint32(label="version")
    if version not in {2, 3}:
        raise GGUFIdentityError(
            f"unsupported GGUF version or endian encoding: {version}"
        )
    tensor_count = reader.uint64(label="tensor count")
    metadata_count = reader.uint64(label="metadata count")
    if tensor_count > _MAX_TENSOR_COUNT:
        raise GGUFIdentityError("GGUF tensor count exceeds the configured bound")
    if metadata_count > _MAX_METADATA_COUNT:
        raise GGUFIdentityError("GGUF metadata count exceeds the configured bound")

    budget = _ParseBudget()
    metadata_records: list[dict[str, object]] = []
    tokenizer_records: list[dict[str, object]] = []
    metadata_keys: set[str] = set()
    alignment = 32
    for _ in range(metadata_count):
        key = _decode_metadata_key(reader)
        if key in metadata_keys:
            raise GGUFIdentityError(f"duplicate metadata key {key!r}")
        metadata_keys.add(key)
        value_type = reader.uint32(label="metadata value type")
        value_sha256, scalar = _read_metadata_value(
            reader,
            value_type,
            budget=budget,
        )
        record: dict[str, object] = {
            "key": key,
            "value_type": value_type,
            "value_sha256": value_sha256,
        }
        metadata_records.append(record)
        if key.startswith("tokenizer."):
            tokenizer_records.append(record)
        if key == "general.alignment":
            if value_type != 4 or not isinstance(scalar, int):
                raise GGUFIdentityError("GGUF general.alignment must be uint32")
            alignment = scalar

    if alignment < 8 or alignment > _MAX_ALIGNMENT or alignment & (alignment - 1) != 0:
        raise GGUFIdentityError("GGUF general.alignment is invalid")

    tensor_records: list[dict[str, object]] = []
    tensor_names: set[str] = set()
    tensor_extents: list[tuple[int, int, str]] = []
    for _ in range(tensor_count):
        name = _decode_tensor_name(reader)
        if name in tensor_names:
            raise GGUFIdentityError(f"duplicate tensor name {name!r}")
        tensor_names.add(name)
        dimension_count = reader.uint32(label="tensor dimension count")
        if dimension_count < 1 or dimension_count > 4:
            raise GGUFIdentityError("GGUF tensor dimension count is invalid")
        dimensions: list[int] = []
        element_count = 1
        for _dimension in range(dimension_count):
            size = reader.uint64(label="tensor dimension")
            if size < 1 or element_count > _MAX_TENSOR_ELEMENTS // size:
                raise GGUFIdentityError("GGUF tensor dimensions exceed the bound")
            element_count *= size
            dimensions.append(size)
        tensor_type = reader.uint32(label="tensor type")
        offset = reader.uint64(label="tensor offset")
        byte_length = _tensor_byte_length(
            dimensions=dimensions,
            element_count=element_count,
            tensor_type=tensor_type,
        )
        tensor_extents.append((offset, byte_length, name))
        tensor_records.append(
            {
                "byte_length": byte_length,
                "dimensions": dimensions,
                "name": name,
                "offset": offset,
                "tensor_type": tensor_type,
            }
        )

    tensor_data_start = reader.position + (
        (alignment - reader.position % alignment) % alignment
    )
    padding_size = tensor_data_start - reader.position
    padding = reader.read_exact(padding_size, label="header padding")
    if any(padding):
        raise GGUFIdentityError("GGUF header padding must be zero-filled")
    tensor_data_size = file_size - tensor_data_start
    if tensor_count and tensor_data_size <= 0:
        raise GGUFIdentityError("GGUF tensor data is missing or truncated")
    for offset, byte_length, _name in tensor_extents:
        if offset % alignment != 0:
            raise GGUFIdentityError("GGUF tensor offset is not aligned")
        if offset >= tensor_data_size:
            raise GGUFIdentityError("GGUF tensor offset exceeds tensor data")
        if byte_length > tensor_data_size - offset:
            raise GGUFIdentityError("GGUF tensor data is missing or truncated")

    ordered_extents = sorted(tensor_extents)
    for previous, current in zip(ordered_extents, ordered_extents[1:], strict=False):
        previous_offset, previous_length, _previous_name = previous
        current_offset, _current_length, _current_name = current
        if current_offset < previous_offset + previous_length:
            raise GGUFIdentityError("GGUF tensor data ranges overlap")

    metadata_records.sort(key=lambda record: str(record["key"]))
    tokenizer_records.sort(key=lambda record: str(record["key"]))
    tensor_records.sort(key=lambda record: str(record["name"]))
    return (
        _canonical_records_sha256(metadata_records),
        _canonical_records_sha256(tensor_records),
        _canonical_records_sha256(tokenizer_records),
    )


def _confirm_unchanged(artifact: _OpenArtifact) -> None:
    expected = _stat_identity(artifact.initial_stat)
    if _stat_identity(os.fstat(artifact.descriptor)) != expected:
        raise GGUFIdentityError("GGUF artifact changed while being read")
    try:
        named = os.stat(
            artifact.basename,
            dir_fd=artifact.parent_descriptor,
            follow_symlinks=False,
        )
        current = artifact.absolute_path.lstat()
    except OSError as exc:
        raise GGUFIdentityError("GGUF artifact was replaced while being read") from exc
    if _stat_identity(named) != expected or _stat_identity(current) != expected:
        raise GGUFIdentityError("GGUF artifact was replaced while being read")


def read_gguf_artifact_identity(
    path: str | os.PathLike[str],
) -> GGUFArtifactIdentity:
    """Read a stable local GGUF file and return privacy-safe content identity."""

    artifact = _open_regular_without_symlinks(path)
    try:
        file_size = artifact.initial_stat.st_size
        if file_size < 24 or file_size > _MAX_FILE_BYTES:
            raise GGUFIdentityError("GGUF file size is outside the configured bounds")
        file_sha256 = _stream_file_sha256(artifact.descriptor, file_size)
        metadata_sha256, tensor_sha256, tokenizer_sha256 = _parse_gguf(
            artifact.descriptor,
            file_size=file_size,
        )
        _confirm_unchanged(artifact)
        return GGUFArtifactIdentity(
            artifact_name=f"gguf-sha256-{file_sha256}.gguf",
            sha256=file_sha256,
            byte_length=file_size,
            gguf_metadata_sha256=metadata_sha256,
            tensor_inventory_sha256=tensor_sha256,
            tokenizer_metadata_sha256=tokenizer_sha256,
        )
    finally:
        os.close(artifact.descriptor)
        os.close(artifact.parent_descriptor)


__all__ = [
    "GGUFIdentityError",
    "read_gguf_artifact_identity",
]
