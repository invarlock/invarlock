"""Descriptor-backed checkpoint input for the maintained Accelerate distribution.

Shard names are relative to the selected checkpoint directory. Indexes and
shards must be regular, materialized files; symlink-based cache layouts must be
materialized before calling these checkpoint APIs. Pinning prevents pathname
replacement, not concurrent modification of the same inode. The owner must keep
checkpoint contents immutable for the lifetime of memory-mapped tensors.

This module is embedded verbatim into the authenticated upstream modeling module.
It deliberately depends only on the Python standard library.
"""

import contextlib as _checkpoint_contextlib
import json as _checkpoint_json
import os as _checkpoint_os
import pathlib as _checkpoint_pathlib
import stat as _checkpoint_stat
import sys as _checkpoint_sys
from typing import NamedTuple as _CheckpointNamedTuple


class _CheckpointMember(_CheckpointNamedTuple):
    directory: int
    parts: tuple[str, ...]


def _checkpoint_parts(name):
    if (
        not isinstance(name, str)
        or not name
        or "\x00" in name
        or "\\" in name
        or _checkpoint_pathlib.PureWindowsPath(name).drive
        or name.startswith("/")
    ):
        raise ValueError("checkpoint shard must be a relative file name")
    parts = tuple(name.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("checkpoint shard contains a noncanonical path component")
    return parts


def _checkpoint_flags():
    if (
        _checkpoint_os.name != "posix"
        or not hasattr(_checkpoint_os, "O_NOFOLLOW")
        or not hasattr(_checkpoint_os, "O_NONBLOCK")
        or not hasattr(_checkpoint_os, "O_DIRECTORY")
        or _checkpoint_os.open not in _checkpoint_os.supports_dir_fd
    ):
        raise ValueError(
            "maintained checkpoint loading requires POSIX descriptor support"
        )
    return (
        _checkpoint_os.O_RDONLY
        | _checkpoint_os.O_NOFOLLOW
        | _checkpoint_os.O_NONBLOCK
        | getattr(_checkpoint_os, "O_CLOEXEC", 0)
    )


def _checkpoint_open_at(directory, name, *, allow_directory=False):
    flags = _checkpoint_flags()
    before = _checkpoint_os.stat(name, dir_fd=directory, follow_symlinks=False)
    directory_entry = _checkpoint_stat.S_ISDIR(before.st_mode)
    if not (
        _checkpoint_stat.S_ISREG(before.st_mode)
        or (allow_directory and directory_entry)
    ):
        raise ValueError(
            "checkpoint input must be a regular file or permitted directory"
        )
    if directory_entry:
        flags |= _checkpoint_os.O_DIRECTORY
    descriptor = _checkpoint_os.open(name, flags, dir_fd=directory)
    try:
        opened = _checkpoint_os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_mode) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
        ):
            raise ValueError("checkpoint input changed while being opened")
        if not (
            _checkpoint_stat.S_ISREG(opened.st_mode)
            or (allow_directory and _checkpoint_stat.S_ISDIR(opened.st_mode))
        ):
            raise ValueError("checkpoint input is not a regular file")
    except BaseException:
        _checkpoint_os.close(descriptor)
        raise
    return descriptor


@_checkpoint_contextlib.contextmanager
def _checkpoint_entry(directory, name, *, allow_directory=False):
    # Keep all pre-open and post-open checks in the existing validator. This
    # context only owns the descriptor after that validator has admitted it.
    descriptor = _checkpoint_open_at(directory, name, allow_directory=allow_directory)
    try:
        yield descriptor
    finally:
        _checkpoint_os.close(descriptor)


@_checkpoint_contextlib.contextmanager
def _checkpoint_parent(path):
    value = _checkpoint_os.fspath(path)
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("checkpoint path must be a nonempty string")
    flags = _checkpoint_flags()
    # The caller selects this root. Only index-derived names are untrusted
    # descendants; every such component is opened relative to its pinned fd.
    absolute = _checkpoint_os.path.abspath(value)
    parent, name = _checkpoint_os.path.split(absolute)
    if not name:
        parent, name = absolute, "."
    descriptor = _checkpoint_os.open(
        parent, (flags & ~_checkpoint_os.O_NOFOLLOW) | _checkpoint_os.O_DIRECTORY
    )
    try:
        yield descriptor, name
    finally:
        _checkpoint_os.close(descriptor)


@_checkpoint_contextlib.contextmanager
def _checkpoint_member_descriptor(member):
    with _checkpoint_contextlib.ExitStack() as stack:
        directory = member.directory
        for component in member.parts[:-1]:
            child = stack.enter_context(
                _checkpoint_entry(directory, component, allow_directory=True)
            )
            if not _checkpoint_stat.S_ISDIR(_checkpoint_os.fstat(child).st_mode):
                raise ValueError("checkpoint path component must be a directory")
            directory = child
        descriptor = stack.enter_context(_checkpoint_entry(directory, member.parts[-1]))
        yield descriptor


def _checkpoint_alias(descriptor):
    if _checkpoint_sys.platform.startswith("linux"):
        alias = f"/proc/self/fd/{descriptor}"
    elif _checkpoint_sys.platform == "darwin":
        alias = f"/dev/fd/{descriptor}"
    else:
        raise ValueError("maintained checkpoint loading supports Linux and macOS")
    original = _checkpoint_os.fstat(descriptor)
    # macOS stat on /dev/fd reports the descriptor filesystem's device. Opening
    # the alias and inspecting that descriptor identifies the actual file on
    # both supported platforms, as the downstream serializers will do.
    probe = _checkpoint_os.open(
        alias, _checkpoint_os.O_RDONLY | _checkpoint_os.O_NONBLOCK
    )
    try:
        observed = _checkpoint_os.fstat(probe)
    finally:
        _checkpoint_os.close(probe)
    if not _checkpoint_stat.S_ISREG(observed.st_mode) or (
        original.st_dev,
        original.st_ino,
    ) != (observed.st_dev, observed.st_ino):
        raise ValueError(
            "checkpoint descriptor alias does not identify the opened file"
        )
    return alias


@_checkpoint_contextlib.contextmanager
def _checkpoint_file(path):
    with _checkpoint_contextlib.ExitStack() as stack:
        if isinstance(path, _CheckpointMember):
            member = path
        else:
            parent, name = stack.enter_context(_checkpoint_parent(path))
            member = _CheckpointMember(parent, (name,))
        descriptor = stack.enter_context(_checkpoint_member_descriptor(member))
        yield _checkpoint_alias(descriptor), member.parts[-1].endswith(".safetensors")


def _checkpoint_read_index(descriptor):
    duplicate = _checkpoint_os.dup(descriptor)
    try:
        stream = _checkpoint_os.fdopen(duplicate, "r", encoding="utf-8")
    except BaseException:
        _checkpoint_os.close(duplicate)
        raise
    with stream:

        def unique_members(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("checkpoint index contains duplicate keys")
                result[key] = value
            return result

        index = _checkpoint_json.load(stream, object_pairs_hook=unique_members)
    if not isinstance(index, dict):
        raise ValueError("checkpoint index must be an object")
    weights = index.get("weight_map", index)
    if not isinstance(weights, dict) or not weights:
        raise ValueError("checkpoint index must contain a nonempty weight map")
    # Validate every name before opening any index-selected shard.
    return sorted(
        {_checkpoint_parts(value) for value in weights.values()}, key="/".join
    )


@_checkpoint_contextlib.contextmanager
def _checkpoint_files(checkpoint):
    with _checkpoint_contextlib.ExitStack() as stack:
        parent, name = stack.enter_context(_checkpoint_parent(checkpoint))
        selected = stack.enter_context(
            _checkpoint_entry(parent, name, allow_directory=True)
        )
        if _checkpoint_stat.S_ISDIR(_checkpoint_os.fstat(selected).st_mode):
            directory = selected
            names = _checkpoint_os.listdir(directory)
            if "pytorch_model.bin" in names:
                members = [("pytorch_model.bin",)]
            elif "model.safetensors" in names:
                members = [("model.safetensors",)]
            else:
                indexes = [item for item in names if item.endswith(".index.json")]
                if len(indexes) != 1:
                    raise ValueError(
                        "checkpoint directory must contain exactly one index"
                    )
                index = stack.enter_context(_checkpoint_entry(directory, indexes[0]))
                members = _checkpoint_read_index(index)
        else:
            directory = parent
            if name.endswith(".json"):
                members = _checkpoint_read_index(selected)
            else:
                members = [(name,)]
        yield [_CheckpointMember(directory, parts) for parts in members]
