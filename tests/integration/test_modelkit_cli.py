"""Opt-in real KitOps serialization; these fixture bytes do not run a model."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from examples.integrations.modelkit_handoff import ModelKitError, verify_package_content
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256

pytestmark = pytest.mark.integration


def test_real_kit_pack_repack_and_independent_recipient(tmp_path):
    configured = os.environ.get("INVARLOCK_KIT_BINARY")
    expected_binary = os.environ.get("INVARLOCK_KIT_BINARY_SHA256")
    if not configured or not expected_binary:
        pytest.skip("requires a KitOps1.15.0 binary and independently verified SHA-256")
    binary = Path(configured)
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == expected_binary
    producer = tmp_path / "producer"
    context = tmp_path / "context"
    context.mkdir()
    model = context / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"serialization-fixture"}\n')
    (model / "model.safetensors").write_bytes(
        b"Synthetic package fixture; no model inference\n"
    )
    tag = "example.invalid/serialization/model:v1"

    def kit(actor, *args):
        result = subprocess.run(
            [str(binary), "--config", str(actor), "--progress", "none", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout

    version = kit(producer, "version")
    assert "1.15.0" in version
    assert "6b8162ae5da4d46f1d2af2beb43e7fb077f052f4" in version

    def pack(description, compression):
        (context / "Kitfile").write_text(
            "manifestVersion: 1.0.0\npackage:\n  name: serialization-fixture\n"
            f"  description: {description}\nmodel:\n  path: model\n"
        )
        kit(producer, "pack", str(context), "--tag", tag, "--compression", compression)
        return json.loads(kit(producer, "inspect", tag))["digest"]

    original = pack("Synthetic original; no inference", "none")
    repacked = pack("Same model in changed package", "gzip")
    assert original != repacked
    recipient = tmp_path / "recipient"
    shutil.copytree(producer / "storage", recipient / "storage")
    # The mutable tag now selects the repack. The recipient independently uses
    # the original manifest digest and an entirely separate local store.
    destination = tmp_path / "recipient-model"
    kit(
        recipient,
        "unpack",
        tag.split(":")[0] + "@" + original,
        "--filter",
        "model",
        "--dir",
        str(destination),
    )
    candidate = destination / "model"
    content = checkpoint_tree_sha256(model)
    assert checkpoint_tree_sha256(candidate) == content
    blobs = recipient / "storage/blobs/sha256"
    first = verify_package_content(
        blobs=blobs,
        expected_package_digest=original,
        candidate=candidate,
        expected_content_digest=content,
    )
    second = verify_package_content(
        blobs=blobs,
        expected_package_digest=repacked,
        candidate=candidate,
        expected_content_digest=content,
    )
    assert first["artifact_content_digest"] == second["artifact_content_digest"]
    (candidate / "model.safetensors").write_bytes(b"replaced after receipt")
    with pytest.raises(ModelKitError, match="content"):
        verify_package_content(
            blobs=blobs,
            expected_package_digest=original,
            candidate=candidate,
            expected_content_digest=content,
        )
