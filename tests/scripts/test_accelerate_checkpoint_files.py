"""Exercise filesystem boundaries without importing a model runtime."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.security import accelerate_checkpoint_files as checkpoint

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor contract")


@pytest.mark.parametrize(
    "name",
    [
        None,
        42,
        "",
        "/tmp/a",
        "C:a",
        "a\\b",
        "a\0b",
        ".",
        "..",
        "a/../b",
        "a/./b",
        "a//b",
        "a/",
    ],
)
def test_rejects_untrusted_shard_names(name):
    with pytest.raises(ValueError, match="checkpoint shard"):
        checkpoint._checkpoint_parts(name)


def test_directory_and_index_preserve_nested_member_order(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.safetensors").write_bytes(b"nested")
    (tmp_path / "a.bin").write_bytes(b"flat")
    index = tmp_path / "model.index.json"
    index.write_text(
        json.dumps({"weight_map": {"x": "a/b.safetensors", "y": "a.bin", "z": "a.bin"}})
    )
    for selected in (tmp_path, index):
        with checkpoint._checkpoint_files(selected) as members:
            assert ["/".join(member.parts) for member in members] == [
                "a.bin",
                "a/b.safetensors",
            ]
            for member, expected in zip(members, (b"flat", b"nested"), strict=True):
                with checkpoint._checkpoint_file(member) as (alias, safe):
                    assert Path(alias).read_bytes() == expected
                    assert safe == (expected == b"nested")
        with pytest.raises(OSError):
            os.fstat(members[0].directory)


@pytest.mark.parametrize("name", ["pytorch_model.bin", "model.safetensors"])
def test_materialized_single_file_and_directory(tmp_path, name):
    selected = tmp_path / name
    selected.write_bytes(b"checkpoint")
    for value in (tmp_path, selected):
        with checkpoint._checkpoint_files(value) as members:
            assert len(members) == 1
            with checkpoint._checkpoint_file(members[0]) as (alias, _):
                assert Path(alias).read_bytes() == b"checkpoint"
    with checkpoint._checkpoint_file(selected) as (alias, _):
        assert Path(alias).read_bytes() == b"checkpoint"


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        "{}",
        '{"weight_map":{}}',
        '{"weight_map":[]}',
        '{"x":"a","x":"b"}',
        '{"weight_map":{"x":"a","x":"b"}}',
        '{"weight_map":{"x":"a","y":"../b"}}',
        "{",
    ],
)
def test_rejects_invalid_index_before_loading_any_shard(tmp_path, payload):
    selected = tmp_path / "model.index.json"
    selected.write_text(payload)
    with pytest.raises(ValueError), checkpoint._checkpoint_files(selected):
        pytest.fail("invalid index admitted")


def test_flat_index_supported_and_ambiguous_directory_rejected(tmp_path):
    selected = tmp_path / "one.index.json"
    selected.write_text('{"x":"weights.bin"}')
    with checkpoint._checkpoint_files(selected) as members:
        assert members[0].parts == ("weights.bin",)
    for count in (0, 2):
        if count == 0:
            selected.unlink()
        else:
            selected.write_text("{}")
            (tmp_path / "two.index.json").write_text("{}")
        with (
            pytest.raises(ValueError, match="exactly one"),
            checkpoint._checkpoint_files(tmp_path),
        ):
            pytest.fail("ambiguous directory admitted")


@pytest.mark.parametrize(
    "kind", ["symlink", "fifo", "directory", "nested_link", "nested_file"]
)
def test_member_rejects_nonregular_and_redirected_paths(tmp_path, kind):
    regular = tmp_path / "regular"
    regular.write_bytes(b"outside")
    member = tmp_path / "member"
    suffix = ""
    if kind == "symlink":
        member.symlink_to(regular)
    elif kind == "fifo":
        os.mkfifo(member)
    elif kind == "directory":
        member.mkdir()
    elif kind == "nested_link":
        member.symlink_to(tmp_path, target_is_directory=True)
        suffix = "/regular"
    else:
        member.write_bytes(b"not a directory")
        suffix = "/regular"
    index = tmp_path / "model.index.json"
    index.write_text(json.dumps({"x": "member" + suffix}))
    with checkpoint._checkpoint_files(index) as members:
        with pytest.raises(ValueError), checkpoint._checkpoint_file(members[0]):
            pytest.fail("nonregular member admitted")


@pytest.mark.parametrize("value", ["", b"binary", "bad\0path"])
def test_invalid_caller_path_rejected(value):
    with pytest.raises(ValueError), checkpoint._checkpoint_parent(value):
        pytest.fail("invalid path admitted")


def test_explicit_root_and_caller_ancestor_link(tmp_path):
    with checkpoint._checkpoint_parent("/") as (fd, name):
        assert name == "."
        assert os.fstat(fd).st_ino == os.stat("/").st_ino
    target = tmp_path / "target"
    target.mkdir()
    (target / "weights.bin").write_bytes(b"data")
    link = tmp_path / "parent"
    link.symlink_to(target, target_is_directory=True)
    with checkpoint._checkpoint_file(link / "weights.bin") as (alias, _):
        assert Path(alias).read_bytes() == b"data"


def test_platform_requirements_fail_closed(monkeypatch):
    monkeypatch.setattr(checkpoint._checkpoint_os, "supports_dir_fd", set())
    with pytest.raises(ValueError, match="POSIX"):
        checkpoint._checkpoint_flags()


def test_descriptor_alias_unsupported_platform_and_wrong_file(tmp_path, monkeypatch):
    selected = tmp_path / "weights"
    selected.write_bytes(b"data")
    fd = os.open(selected, os.O_RDONLY)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(checkpoint._checkpoint_sys, "platform", "unsupported")
            with pytest.raises(ValueError, match="Linux and macOS"):
                checkpoint._checkpoint_alias(fd)
        original = os.fstat
        with monkeypatch.context() as patch:
            patch.setattr(
                checkpoint._checkpoint_os,
                "fstat",
                lambda value: (
                    original(value)
                    if value == fd
                    else SimpleNamespace(
                        st_dev=-1, st_ino=-1, st_mode=original(value).st_mode
                    )
                ),
            )
            with pytest.raises(ValueError, match="alias"):
                checkpoint._checkpoint_alias(fd)
    finally:
        os.close(fd)


def test_replacement_during_open_rejected_and_descriptor_closed(tmp_path, monkeypatch):
    selected = tmp_path / "weights"
    selected.write_bytes(b"old")
    other = tmp_path / "other"
    other.write_bytes(b"new")
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    original = os.open
    opened = []

    def replace(name, flags, *, dir_fd=None):
        other.replace(selected)
        fd = original(name, flags, dir_fd=dir_fd)
        opened.append(fd)
        return fd

    try:
        # Keep the feature probe bound to the fault-injected open function.
        monkeypatch.setattr(checkpoint._checkpoint_os, "open", replace)
        monkeypatch.setattr(checkpoint._checkpoint_os, "supports_dir_fd", {replace})
        with pytest.raises(ValueError, match="changed"):
            checkpoint._checkpoint_open_at(parent, "weights")
        with pytest.raises(OSError) as closed:
            os.fstat(opened[0])
        assert closed.value.errno == errno.EBADF
    finally:
        os.close(parent)


def test_failed_index_stream_closes_duplicate(tmp_path, monkeypatch):
    selected = tmp_path / "index.json"
    selected.write_text("{}")
    fd = os.open(selected, os.O_RDONLY)
    duplicate = []
    original = os.dup

    def track(value):
        result = original(value)
        duplicate.append(result)
        return result

    def fail(*args, **kwargs):
        raise OSError("stream failure")

    try:
        monkeypatch.setattr(checkpoint._checkpoint_os, "dup", track)
        monkeypatch.setattr(checkpoint._checkpoint_os, "fdopen", fail)
        with pytest.raises(OSError, match="stream failure"):
            checkpoint._checkpoint_read_index(fd)
        with pytest.raises(OSError):
            os.fstat(duplicate[0])
        assert os.fstat(fd).st_size == 2
    finally:
        os.close(fd)
