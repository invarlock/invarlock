"""Filesystem owners release resources when acquisition or cleanup fails."""

from __future__ import annotations

import ast
import errno
import hashlib
import importlib.util
import io
import os
import stat
from functools import partial
from pathlib import Path

import pytest

from examples import generate_keys
from examples.integrations import modelkit_handoff, trust_material
from examples.qualification import (
    k2_campaign,
    k2_runtime_apt,
    k2_runtime_build,
    k2_runtime_finalize,
    k2_runtime_source,
)


def _assert_closed(descriptors):
    leaked = []
    for descriptor in descriptors:
        try:
            os.fstat(descriptor)
        except OSError as exc:
            assert exc.errno == errno.EBADF
        else:
            leaked.append(descriptor)
            os.close(descriptor)
    assert not leaked, "file descriptors were leaked"


@pytest.mark.parametrize(
    "reader",
    [
        "native",
        "source",
        "build",
        "apt",
        "snapshot",
        "blob",
        "inventory",
        "key",
        "probe",
    ],
)
def test_stream_creation_failure_releases_owned_descriptor(
    tmp_path, monkeypatch, reader
):
    source = tmp_path / "source"
    source.write_bytes(b"input")
    opened = []
    error = OSError(errno.EMFILE, "cannot create stream")

    def fail_fdopen(descriptor, *args, **kwargs):
        opened.append(descriptor)
        raise error

    if reader == "native":
        spec = importlib.util.spec_from_file_location(
            "native_cleanup",
            Path(__file__).resolve().parents[2]
            / "examples/captured-results/native_handoff.py",
        )
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        action = partial(native.read, source)
    elif reader == "source":
        action = partial(k2_runtime_source.prepare, source, tmp_path / "output")
    elif reader == "build":
        action = partial(k2_runtime_build._read, source, 100)
    elif reader == "apt":
        action = partial(k2_runtime_apt.read_input, source, 100)
    elif reader == "snapshot":
        action = partial(
            k2_campaign.measure_snapshot,
            tmp_path,
            [{"path": "source", "size_bytes": 5, "sha256": "", "git_blob": ""}],
        )
    elif reader == "blob":
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        source.rename(tmp_path / digest)
        action = partial(
            modelkit_handoff._blob,
            tmp_path,
            "sha256:" + digest,
            io.BytesIO(),
            maximum=100,
        )
    elif reader == "inventory":
        action = partial(
            modelkit_handoff._inventory, tmp_path, modelkit_handoff.Limits()
        )
    elif reader == "key":
        action = partial(generate_keys._write_key, tmp_path / "key.pem")
    else:
        function = next(
            node
            for node in ast.parse(k2_runtime_finalize.INVENTORY).body
            if isinstance(node, ast.FunctionDef) and node.name == "campaign_hash"
        )
        namespace = {"os": os, "stat": stat, "hashlib": hashlib}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "inventory", "exec"),
            namespace,
        )
        original_open = os.open
        monkeypatch.setattr(
            os, "open", lambda path, flags: original_open(source, flags)
        )
        action = partial(namespace["campaign_hash"], "source")
    monkeypatch.setattr(os, "fdopen", fail_fdopen)
    with pytest.raises(OSError) as failure:
        action()
    assert failure.value is error
    assert opened
    _assert_closed(opened)
    assert not (tmp_path / "key.pem").exists()


@pytest.mark.parametrize("reader", ["trust", "finalize"])
def test_traversal_closes_new_child_when_previous_parent_close_fails(
    tmp_path, monkeypatch, reader
):
    source = tmp_path / "source"
    source.write_bytes(b"input")
    original_open, original_close = os.open, os.close
    live = set()
    error = OSError(errno.EIO, "parent close failed")
    failed = False

    def track_open(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        live.add(descriptor)
        return descriptor

    def fail_first_close(descriptor):
        nonlocal failed
        original_close(descriptor)
        live.discard(descriptor)
        if not failed:
            failed = True
            raise error

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", track_open)
        patch.setattr(os, "close", fail_first_close)
        with pytest.raises((OSError, ValueError)):
            if reader == "trust":
                trust_material.read_external_file(source, label="source")
            else:
                k2_runtime_finalize.read(source)
    assert failed
    _assert_closed(live)


def test_trust_root_cleanup_continues_after_policy_directory_close_error(
    tmp_path, monkeypatch
):
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    original_open, original_close = os.open, os.close
    live = set()
    policy_descriptor = None
    error = OSError(errno.EIO, "policy directory close failed")

    def track_open(name, *args, **kwargs):
        nonlocal policy_descriptor
        descriptor = original_open(name, *args, **kwargs)
        live.add(descriptor)
        if name == "policy":
            policy_descriptor = descriptor
        return descriptor

    def fail_policy_close(descriptor):
        original_close(descriptor)
        live.discard(descriptor)
        if descriptor == policy_descriptor:
            raise error

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", track_open)
        patch.setattr(os, "close", fail_policy_close)
        with pytest.raises(OSError) as failure:
            trust_material.create_trust_material(
                transaction_root=transaction,
                evidence_key=tmp_path / "signer.pem",
                verifier_key_bytes=b"key",
                evidence_fingerprint="evidence",
                verifier_fingerprint="verifier",
                trust_root=tmp_path / "trust",
                policy_bytes=b"{}\n",
                verifier_identity="recipient",
                anchors={},
            )
    assert failure.value is error
    _assert_closed(live)
