from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.runtime_providers import tensorrt_llm_identity as identity


def _error(message: str):
    return pytest.raises(identity.TensorRTLLMIdentityError, match=message)


def test_logical_names_and_root_paths_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _error("entry name is invalid"):
        identity._logical_name(())
    with _error("valid UTF-8"):
        identity._logical_name(("\ud800",))
    monkeypatch.setattr(identity, "_MAX_LOGICAL_PATH_BYTES", 2)
    with _error("configured bound"):
        identity._logical_name(("long",))

    class InvalidPath:
        def __fspath__(self) -> str:
            raise OSError("invalid")

    with _error("path is invalid"):
        identity._open_root_without_symlinks(InvalidPath())  # type: ignore[arg-type]
    with _error("identify a directory"):
        identity._open_root_without_symlinks(Path("/"))

    monkeypatch.delattr(identity.os, "O_NOFOLLOW")
    with _error("nofollow bundle opening"):
        identity._open_root_without_symlinks(tmp_path)


def test_root_open_and_directory_identity_errors_close_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        identity.os, "open", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError())
    )
    with _error("root cannot be opened"):
        identity._open_root_without_symlinks(tmp_path)


def test_json_budget_and_config_validation_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_budget = identity._JsonBudget()
    identity._validate_json_budget([], budget=empty_budget)
    assert empty_budget.items == 0

    with _error("object keys must be strings"):
        identity._validate_json_budget({1: "value"}, budget=identity._JsonBudget())
    with _error("invalid Unicode key"):
        identity._validate_json_budget({"\ud800": 1}, budget=identity._JsonBudget())
    with _error("invalid Unicode string"):
        identity._validate_json_budget("\ud800", budget=identity._JsonBudget())
    with _error("non-finite JSON number"):
        identity._validate_json_budget(math.inf, budget=identity._JsonBudget())
    monkeypatch.setattr(identity, "_MAX_JSON_ITEMS", 1)
    with _error("item count"):
        identity._validate_json_budget([1, 2], budget=identity._JsonBudget())

    with _error("finite canonical JSON"):
        identity._canonical_json({"value": object()})
    assert identity._finite_float("1.5") == 1.5
    with _error("must be printable"):
        identity._require_nonempty_text("bad\x00value", field="version")

    valid = {
        "version": "1",
        "pretrained_config": {
            "architecture": "LlamaForCausalLM",
            "dtype": "float16",
            "mapping": {"world_size": 1},
        },
        "build_config": {"max_batch_size": 1},
    }
    cases = [
        ({**valid, "extra": True}, "contain exactly"),
        ({**valid, "version": " bad"}, "bounded non-empty text"),
        ({**valid, "version": "bad\n"}, "bounded non-empty text"),
        ({**valid, "pretrained_config": []}, "pretrained_config"),
        ({**valid, "build_config": []}, "build_config"),
        (
            {
                **valid,
                "pretrained_config": {
                    "architecture": "LlamaForCausalLM",
                    "dtype": "float16",
                    "mapping": [],
                },
            },
            "mapping must be an object",
        ),
    ]
    for candidate, message in cases:
        with _error(message):
            identity._validated_engine_config(candidate)  # type: ignore[arg-type]


def test_tree_budget_listing_and_entry_lookup_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    descriptor = os.open(bundle, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(identity, "_MAX_DIRECTORY_COUNT", 0)
        with _error("directory count"):
            identity._collect_files(descriptor)
        monkeypatch.setattr(identity, "_MAX_DIRECTORY_COUNT", 32)
        monkeypatch.setattr(
            identity.os, "listdir", lambda _fd: (_ for _ in ()).throw(OSError())
        )
        with _error("cannot be listed"):
            identity._collect_files(descriptor)
    finally:
        os.close(descriptor)


def test_tree_collection_reports_entry_disappearance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    descriptor = os.open(bundle, os.O_RDONLY | os.O_DIRECTORY)
    monkeypatch.setattr(identity.os, "listdir", lambda _fd: ["missing.engine"])
    monkeypatch.setattr(
        identity.os,
        "stat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("gone")),
    )
    try:
        with _error("entry 'missing.engine' is unavailable"):
            identity._collect_files(descriptor)
    finally:
        os.close(descriptor)


def test_tree_depth_limit_rejects_nested_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "nested").mkdir(parents=True)
    descriptor = os.open(bundle, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(identity, "_MAX_TREE_DEPTH", 0)
        with _error("tree depth"):
            identity._collect_files(descriptor)
    finally:
        os.close(descriptor)


def test_nested_component_and_file_open_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    original_open = identity.os.open
    try:

        def fail_directory(path, flags, *args, **kwargs):  # noqa: ANN001, ANN202
            if path == "nested":
                raise OSError("changed")
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(identity.os, "open", fail_directory)
        with _error("directory for 'nested/value.engine' changed"):
            identity._open_file_by_components(descriptor, "nested/value.engine")

        def fail_file(path, flags, *args, **kwargs):  # noqa: ANN001, ANN202
            if path == "value.engine":
                raise OSError("changed")
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(identity.os, "open", fail_file)
        with _error("cannot be opened safely"):
            identity._open_file_by_components(descriptor, "value.engine")
    finally:
        os.close(descriptor)


def test_bounded_reader_detects_size_and_stat_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    value = root / "config.json"
    value.write_bytes(b"{}")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    record = identity._FileRecord("config.json", 2, value.stat())
    try:
        with _error("JSON-size bound"):
            identity._read_bounded_file(descriptor, record, 1)

        changed = identity._FileRecord(
            "config.json",
            2,
            SimpleNamespace(
                st_dev=0,
                st_ino=0,
                st_mode=0,
                st_size=2,
                st_mtime_ns=0,
                st_ctime_ns=0,
            ),  # type: ignore[arg-type]
        )
        with _error("changed before parsing"):
            identity._read_bounded_file(descriptor, changed, 10)

        with _error("changed while parsing"):
            identity._read_bounded_file(
                descriptor,
                identity._FileRecord("config.json", 1, value.stat()),
                10,
            )
    finally:
        os.close(descriptor)


def test_file_hashing_rejects_nonregular_truncated_growing_and_replaced_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    value = root / "value.engine"
    value.write_bytes(b"engine")
    root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    initial = value.stat()
    try:
        monkeypatch.setattr(identity.os, "fstat", lambda _fd: root.stat())
        with _error("not a stable regular file"):
            identity._hash_file(
                root_descriptor,
                identity._FileRecord("value.engine", len(b"engine"), initial),
            )
        monkeypatch.undo()

        with _error("truncated while hashing"):
            identity._hash_file(
                root_descriptor,
                identity._FileRecord("value.engine", len(b"engine") + 1, initial),
            )
        with _error("grew while hashing"):
            identity._hash_file(
                root_descriptor,
                identity._FileRecord("value.engine", len(b"engine") - 1, initial),
            )

        real_stat = identity.os.stat

        def missing_after_hash(path, *args, **kwargs):  # noqa: ANN001, ANN202
            if path == "value.engine" and kwargs.get("dir_fd") is not None:
                raise OSError("replaced")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(identity.os, "stat", missing_after_hash)
        with _error("changed after hashing"):
            identity._hash_file(
                root_descriptor,
                identity._FileRecord("value.engine", len(b"engine"), initial),
            )
    finally:
        os.close(root_descriptor)


def test_bounded_config_reader_rejects_truncation_and_invalid_json(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    value = root / "config.json"
    value.write_bytes(b"{}")
    root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with _error("truncated while parsing"):
            identity._read_bounded_file(
                root_descriptor,
                identity._FileRecord("config.json", 3, value.stat()),
                10,
            )
    finally:
        os.close(root_descriptor)

    with _error("not strict JSON"):
        identity._parse_config(b"{")


def test_empty_authenticated_bundle_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "empty"
    bundle.mkdir()
    with _error("engine bundle is empty"):
        identity.read_tensorrt_llm_engine_tree_sha256(bundle)
