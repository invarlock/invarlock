"""Native GGUF inspection against complete source identity and rejection semantics."""

import base64
import json
import math
import os
import struct
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from invarlock.runtime_providers import gguf_identity as reference
from tests.runtime_providers.test_gguf_identity import (
    _fixture,
    _metadata,
    _string,
    _tensor,
)


@pytest.fixture(scope="module")
def binary():
    value = os.environ.get("INVARLOCK_NATIVE_VERIFY")
    assert value and Path(value).is_file(), (
        "set INVARLOCK_NATIVE_VERIFY to the rebuilt verifier"
    )
    return value


def native(binary, path):
    config = {"path_base64": base64.b64encode(os.fsencode(path)).decode("ascii")}
    run = subprocess.run(
        [binary, "gguf-identity", "--request", "-"],
        input=json.dumps(config).encode(),
        capture_output=True,
        check=False,
        timeout=60,
    )
    return run.returncode, json.loads(run.stdout)


def check(binary, path):
    try:
        expected = asdict(reference.read_gguf_artifact_identity(path))
    except reference.GGUFIdentityError as error:
        status, result = native(binary, path)
        assert status == 4
        assert result["error"] == str(error)
    else:
        expected.pop("format_version")
        expected.pop("artifact_format")
        status, result = native(binary, path)
        assert status == 0, result
        assert result == expected


@pytest.mark.parametrize("version", [2, 3])
def test_versions_and_inventory_partitions(binary, tmp_path, version):
    for index, payload in enumerate(
        [
            _fixture(version=version),
            _fixture(version=version, architecture="mistral"),
            _fixture(version=version, tokens=["💛", "\\", '"', "\u0000"]),
            _fixture(version=version, tensors=[_tensor("日本語.weight")]),
            _fixture(version=version, tensors=[], tensor_data=b""),
            _fixture(version=version, alignment=1024, tensor_data=bytes(64)),
        ]
    ):
        path = tmp_path / f"{index}.gguf"
        path.write_bytes(payload)
        check(binary, path)


@pytest.mark.parametrize(
    "kind,layout", sorted(reference._GGML_TENSOR_BLOCK_SIZES.items())
)
def test_all_tensor_layouts_and_truncation(binary, tmp_path, kind, layout):
    block, size = layout
    for retained in [size, size - 1]:
        path = tmp_path / f"{kind}-{retained}.gguf"
        path.write_bytes(
            _fixture(
                tensors=[_tensor("weight", dimensions=(block,), tensor_type=kind)],
                tensor_data=bytes(retained),
            )
        )
        check(binary, path)


def test_scalar_array_types_and_nesting(binary, tmp_path):
    extra = []
    for kind, fmt in reference._FIXED_VALUE_FORMATS.items():
        value = 1.25 if kind in [6, 12] else 1
        extra.append(_metadata(f"scalar.v{kind}", kind, struct.pack("<" + fmt, value)))
        extra.append(
            _metadata(
                f"array.v{kind}",
                9,
                struct.pack("<IQ", kind, 3)
                + struct.pack("<" + fmt * 3, *([value] * 3)),
            )
        )
    nested = struct.pack("<IQ", 0, 1) + b"\x01"
    for _ in range(3):
        nested = struct.pack("<IQ", 9, 1) + nested
    extra.append(_metadata("array.nested", 9, nested))
    path = tmp_path / "values.gguf"
    path.write_bytes(_fixture(extra_metadata=extra))
    check(binary, path)
    nested = struct.pack("<IQ", 9, 1) + nested
    path = tmp_path / "depth.gguf"
    path.write_bytes(_fixture(extra_metadata=[_metadata("array.nested", 9, nested)]))
    check(binary, path)


def test_malformed_headers_and_metadata(binary, tmp_path):
    def header(tensors, metadata):
        return b"GGUF" + struct.pack("<IQQ", 3, tensors, metadata)

    cases = [
        b"",
        b"BAD!" + _fixture()[4:],
        b"GGUF" + struct.pack("<IQQ", 1, 0, 0),
        b"GGUF" + struct.pack(">IQQ", 3, 0, 0),
        header(262145, 0),
        header(0, 16385),
        header(0, 1) + struct.pack("<Q", 65536),
        header(0, 1) + _metadata("general.bad", 99, b""),
        header(0, 1) + _metadata("general.bool", 7, b"\x02"),
        header(0, 1) + _metadata("general.float", 6, struct.pack("<f", math.nan)),
        header(0, 1) + _metadata("general.float", 12, struct.pack("<d", math.inf)),
        header(0, 1) + _metadata("general.bad", 8, struct.pack("<Q", 1) + b"\xff"),
        header(0, 1)
        + _metadata("general.big", 8, struct.pack("<Q", 16 * 1024 * 1024 + 1)),
        header(0, 1) + _metadata("general.array", 9, struct.pack("<I", 99)),
        header(0, 1) + _metadata("general.array", 9, struct.pack("<IQ", 0, 2000001)),
        header(0, 1) + _metadata("Bad.key", 0, b"\x00"),
        header(0, 1) + struct.pack("<Q", 1) + b"\xff",
        _fixture(extra_metadata=[_metadata("general.architecture", 0, b"\x01")]),
        _fixture(alignment=7),
        _fixture(alignment=24),
        header(0, 1) + _metadata("general.alignment", 8, _string("32")),
        _fixture(tensors=[_tensor("x", dimensions=())]),
        _fixture(tensors=[_tensor("x", dimensions=(0,))]),
        _fixture(tensors=[_tensor("x", dimensions=(1 << 63,))]),
        _fixture(tensors=[_tensor("x", tensor_type=999)]),
        _fixture(tensors=[_tensor("x", offset=1)]),
        _fixture(tensors=[_tensor("x", offset=32)]),
        _fixture(tensors=[_tensor("x"), _tensor("x")]),
        _fixture(
            tensors=[_tensor("x", dimensions=(32,), tensor_type=2)],
            tensor_data=bytes(17),
        ),
        _fixture(tensors=[_tensor("x", dimensions=(31,), tensor_type=2)]),
        _fixture(
            tensors=[_tensor("x", dimensions=(8, 2)), _tensor("y", offset=32)],
            tensor_data=bytes(64),
        ),
        _fixture(tensors=[_tensor("")]),
        _fixture(tensors=[_tensor("\n")]),
        header(1, 0) + struct.pack("<Q", 65),
    ]
    for i, payload in enumerate(cases):
        path = tmp_path / f"invalid-{i}.gguf"
        path.write_bytes(payload)
        check(binary, path)
    payload = bytearray(_fixture())
    payload[-17] = 1
    path = tmp_path / "padding.gguf"
    path.write_bytes(payload)
    check(binary, path)


def test_streamed_large_metadata_and_path_guards(binary, tmp_path):
    path = tmp_path / "large.gguf"
    path.write_bytes(_fixture(tokens=["token"] * 100000))
    check(binary, path)
    path = tmp_path / "max-string.gguf"
    path.write_bytes(
        _fixture(
            extra_metadata=[
                _metadata("general.large", 8, _string("x" * (16 * 1024 * 1024)))
            ]
        )
    )
    check(binary, path)
    linked = tmp_path / "link.gguf"
    linked.symlink_to(path)
    check(binary, linked)
    directory = tmp_path / "directory"
    directory.mkdir()
    check(binary, directory)
    parent = tmp_path / "linked-directory"
    parent.symlink_to(directory, target_is_directory=True)
    check(binary, parent / "missing")
    check(binary, tmp_path / "missing")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    check(binary, fifo)
    check(binary, Path("/"))


def test_non_utf8_filesystem_path(binary, tmp_path):
    import errno

    invalid_name = os.fsdecode(os.fsencode(tmp_path) + b"/name-\xff.gguf")
    try:
        Path(invalid_name).write_bytes(_fixture())
    except OSError as exc:
        if exc.errno == errno.EILSEQ:
            pytest.skip("filesystem rejects non-UTF-8 filenames")
        raise
    check(binary, invalid_name)


def test_relative_paths_are_normalized_without_following_symlinks(
    binary, tmp_path, monkeypatch
):
    (tmp_path / "model.gguf").write_bytes(_fixture())
    monkeypatch.chdir(tmp_path)
    check(binary, "model.gguf")
    check(binary, "missing/../model.gguf")


@pytest.mark.skipif(
    not os.environ.get("INVARLOCK_NATIVE_GGUF_BINDINGS"),
    reason="requires rebuilt native GGUF extension",
)
def test_selected_rust_adapter_uses_native_parser(tmp_path, monkeypatch):
    from invarlock.runtime_providers.native_execution import using_execution_backend

    path = tmp_path / "model.gguf"
    path.write_bytes(_fixture())
    expected = reference.read_gguf_artifact_identity(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Rust-selected GGUF inspection called Python parser")

    monkeypatch.setattr(reference, "_parse_gguf", forbidden)
    monkeypatch.setattr(reference, "_stream_file_sha256", forbidden)
    with using_execution_backend("rust"):
        assert reference.read_gguf_artifact_identity(path) == expected
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"BAD!" + _fixture()[4:])
        with pytest.raises(reference.GGUFIdentityError, match="GGUF magic is invalid"):
            reference.read_gguf_artifact_identity(bad)
    monkeypatch.setenv("INVARLOCK_EXECUTION_BACKEND", "rust")
    assert reference.read_gguf_artifact_identity(path) == expected
