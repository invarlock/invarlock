"""Replay the retained container reference and reject altered transport bytes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "examples/qualification/container-engines"
spec = importlib.util.spec_from_file_location(
    "container_reference", REFERENCE / "replay.py"
)
assert spec and spec.loader
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def test_reference_replays_all_signed_packs(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["replay.py"])
    assert reference.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["pack_count"] == 14
    assert all(
        row["evidence_accepted"] and row["receipt_ok"] for row in result["results"]
    )


def test_reference_inventory_is_bounded_and_public():
    archive = REFERENCE / "reference.zip"
    provenance = json.loads((REFERENCE / "provenance.json").read_bytes())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == reference.ARCHIVE_SHA256
    assert (REFERENCE / "reference.sha256").read_text().split()[
        0
    ] == reference.ARCHIVE_SHA256
    assert provenance["archive"]["sha256"] == reference.ARCHIVE_SHA256
    assert archive.stat().st_size == provenance["archive"]["bytes"] < 400_000
    with zipfile.ZipFile(archive) as stream:
        manifest = json.loads(stream.read("manifest.json"))
        names = stream.namelist()
        assert names == sorted(set(names))
        assert set(names) == set(manifest["files"]) | {"manifest.json"}
        assert len(names) == provenance["archive"]["file_count"]
        assert len(manifest["rows"]) == provenance["archive"]["pack_count"] == 14
        assert (
            sum(info.file_size for info in stream.infolist())
            == provenance["archive"]["expanded_bytes"]
            < 800_000
        )
        for info in stream.infolist():
            assert stat.S_ISREG(info.external_attr >> 16)
            assert not Path(info.filename).is_absolute()
            assert ".." not in Path(info.filename).parts
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            payload = stream.read(info)
            for marker in (b"PRIVATE KEY", b"/opt/", b"/Users/", b"/root/"):
                assert marker not in payload
            if info.filename != "manifest.json":
                assert manifest["files"][info.filename] == {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }


def test_modified_archive_is_rejected_before_extraction(tmp_path, monkeypatch):
    archive = tmp_path / "modified.zip"
    archive.write_bytes((REFERENCE / "reference.zip").read_bytes() + b"changed")

    def no_extraction(*args, **kwargs):
        pytest.fail("archive was opened before its expected digest was checked")

    monkeypatch.setattr(zipfile, "ZipFile", no_extraction)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        reference.replay(archive)


def test_verified_snapshot_is_used_after_path_replacement(tmp_path, monkeypatch):
    archive = tmp_path / "reference.zip"
    archive.write_bytes((REFERENCE / "reference.zip").read_bytes())
    original_reader = reference.read_regular_file_bytes

    def replace_after_read(path, **kwargs):
        snapshot = original_reader(path, **kwargs)
        path.write_bytes(b"replacement is not a ZIP archive")
        return snapshot

    monkeypatch.setattr(reference, "read_regular_file_bytes", replace_after_read)
    assert reference.replay(archive)["ok"] is True


@pytest.mark.parametrize("kind", ["oversized", "symlink"])
def test_nonregular_or_oversized_archive_is_rejected(tmp_path, kind):
    archive = tmp_path / "reference.zip"
    if kind == "oversized":
        archive.write_bytes(b"x" * 400_001)
        message = "size limit"
    else:
        archive.symlink_to(REFERENCE / "reference.zip")
        message = "symlink"
    with pytest.raises(ValueError, match=message):
        reference.replay(archive)


def test_judge_receipt_must_match_fresh_pack_replay(tmp_path):
    from dataclasses import replace

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from invarlock.judge_measurements.acceptance import (
        verify_judge_evidence_with_policy,
        verify_signed_judge_verification_receipt,
        write_signed_judge_verification_receipt,
    )
    from invarlock.judge_measurements.evidence import object_sha256

    root = tmp_path.resolve()
    with zipfile.ZipFile(REFERENCE / "reference.zip") as stream:
        stream.extractall(root)
    manifest = json.loads((root / "manifest.json").read_bytes())
    row = next(row for row in manifest["rows"] if row["name"] == "native-judge-docker")
    pack, policy = root / row["pack"], root / row["policy"]
    actual = verify_judge_evidence_with_policy(pack, policy)
    assert actual.accepted
    # A valid signature under the same recipient policy can still describe a
    # different envelope. Require the receipt's entire result to match replay.
    different = replace(actual, envelope_sha256="0" * 64)
    key = Ed25519PrivateKey.generate()
    key_path = root / "test-key.pem"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    (root / row["verifier_public_key"]).write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    receipt_path = root / "different-result.receipt.json"
    fingerprint = write_signed_judge_verification_receipt(
        pack,
        different,
        receipt_path,
        recipient_policy_path=policy,
        verifier_identity="native-judge-container-recipient",
        verifier_signing_key_path=key_path,
    )
    assert verify_signed_judge_verification_receipt(
        receipt_path,
        expected_verifier_identity="native-judge-container-recipient",
        expected_verifier_fingerprint=fingerprint,
        expected_recipient_policy_sha256=object_sha256(json.loads(policy.read_bytes())),
    ).ok
    result = reference.verify_row(root, {**row, "receipt": receipt_path.name})
    assert result["evidence_accepted"] is True
    assert result["receipt_ok"] is False
    assert "differs from fresh recipient verification" in result["receipt_errors"][0]
