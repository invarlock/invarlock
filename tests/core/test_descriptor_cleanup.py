"""Descriptor ownership survives errors while retiring earlier handles."""

from __future__ import annotations

import os

import pytest

from invarlock import evaluation_transaction, evidence_reporting, trust_inputs
from invarlock.core import checkpoint_identity, evaluation_request
from invarlock.core.runtime_provider import types as provider_types
from invarlock.filesystem import atomic_directory
from invarlock.runtime_providers import gguf_identity, tensorrt_llm_identity


@pytest.mark.parametrize(
    "operation, failure_point",
    [
        (operation, "first")
        for operation in [
            "trust_parent",
            "trust_input",
            "checkpoint_root",
            "checkpoint_file",
            "checkpoint_return",
            "atomic_parent",
            "gguf_parent",
            "tensorrt_parent",
            "tensorrt_file",
            "request_input",
            "request_output",
            "transaction_input",
            "transaction_output",
            "transaction_anchor",
            "provider_resource",
            "html_output",
        ]
    ]
    + [
        ("request_input", "leaf"),
        ("request_output", "parent"),
        ("transaction_input", "leaf"),
        ("transaction_output", "parent"),
        ("transaction_anchor", "root"),
        ("provider_resource", "leaf"),
        ("provider_resource", "parent"),
        ("html_output", "parent"),
    ],
)
def test_parent_close_failure_releases_newly_owned_descriptors(
    tmp_path, monkeypatch, operation, failure_point
):
    directory = tmp_path / "a" / "b"
    directory.mkdir(parents=True)
    source = directory / "input"
    source.write_bytes(b"payload")
    original_open, original_dup, original_close = os.open, os.dup, os.close
    external_root = original_open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    active = set()
    duplicated = False
    opened = {}
    root_descriptor = None
    failed = False

    def track_open(*args, **kwargs):
        nonlocal root_descriptor
        descriptor = original_open(*args, **kwargs)
        active.add(descriptor)
        opened[str(args[0])] = descriptor
        if root_descriptor is None:
            root_descriptor = descriptor
        return descriptor

    def track_dup(descriptor):
        nonlocal duplicated
        result = original_dup(descriptor)
        active.add(result)
        duplicated = True
        return result

    def fail_close(descriptor):
        nonlocal failed
        original_close(descriptor)
        active.discard(descriptor)
        selected = {
            "first": True,
            "leaf": descriptor == opened.get("input"),
            "parent": descriptor == opened.get("b"),
            "root": duplicated and descriptor == root_descriptor,
        }[failure_point]
        if (
            not failed
            and selected
            and (operation != "transaction_anchor" or duplicated)
        ):
            failed = True
            raise OSError("parent close failed")

    operations = {
        "trust_parent": lambda: trust_inputs._open_directory_without_links(
            directory, label="test"
        ),
        "trust_input": lambda: trust_inputs._read_relative_regular_file(
            external_root, ("a", "b", "input"), label="test", max_bytes=20
        ),
        "checkpoint_root": lambda: checkpoint_identity._open_checkpoint_root(directory),
        "checkpoint_file": lambda: checkpoint_identity._open_checkpoint_file(
            external_root, "a/b/input"
        ),
        "checkpoint_return": lambda: checkpoint_identity._open_checkpoint_file(
            external_root, "a"
        ),
        "atomic_parent": lambda: atomic_directory._open_directory(
            directory, label="test"
        ),
        "gguf_parent": lambda: gguf_identity._open_regular_without_symlinks(source),
        "tensorrt_parent": lambda: tensorrt_llm_identity._open_root_without_symlinks(
            directory
        ),
        "tensorrt_file": lambda: tensorrt_llm_identity._open_file_by_components(
            external_root, "a/b/input"
        ),
        "request_input": lambda: evaluation_request._resolve_existing_reference(
            tmp_path, "a/b/input", label="test", expected="file"
        ),
        "request_output": lambda: evaluation_request._resolve_output_reference(
            tmp_path, "a/b/output", label="test"
        ),
        "transaction_input": lambda: evaluation_transaction._read_request_file(
            tmp_path, source, label="test"
        ),
        "transaction_output": lambda: evaluation_transaction._prepare_output_parent(
            tmp_path, directory / "output"
        ),
        "transaction_anchor": lambda: evaluation_transaction._prepare_output_parent(
            tmp_path, directory / "output"
        ),
        "provider_resource": lambda: provider_types._validate_resource_path(
            tmp_path, "a/b/input", label="test"
        ),
        "html_output": lambda: evidence_reporting._write_html_no_clobber(
            directory / "report.html", "report"
        ),
    }
    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "open", track_open)
            patch.setattr(os, "dup", track_dup)
            patch.setattr(os, "close", fail_close)
            if operation == "html_output":
                # Cleanup cannot turn a synchronized publication into failure.
                operations[operation]()
                assert (directory / "report.html").read_text() == "report"
            else:
                with pytest.raises((OSError, ValueError)):
                    operations[operation]()
        assert failed
        # fdopen-owned handles are closed by their file object, bypassing os.close.
        for descriptor in tuple(active):
            try:
                os.fstat(descriptor)
            except OSError:
                active.remove(descriptor)
        assert not active, f"{operation} leaked descriptors {active}"
    finally:
        for descriptor in active:
            original_close(descriptor)
        original_close(external_root)
