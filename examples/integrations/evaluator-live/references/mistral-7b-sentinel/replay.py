#!/usr/bin/env python3
"""Recheck the retained evaluator sentinel without models or evaluator SDKs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from sysconfig import get_path

from invarlock.evidence_pack_json import read_regular_file_bytes

ARCHIVES = {
    "captures.zip": "ad3573797f9fe6d9ff69f2016e4b1c70d96b9388cc82ceb22e39eda1d2b90e27",
    "exact-match.zip": "f86c66bfee349eda13a4e04267ae0976d935a55afc568f56fd676cc527dd724b",
    "normalized-nll.zip": "0afee050178e58eccea1db31cac0c37cb8c4571e79c3bbf1e4aed2d8571539b5",
}
ARCHIVE_LIMIT = 10 * 1024**2
EXPANDED_LIMIT = 128 * 1024**2


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_archive(
    path,
    *,
    expected_sha256,
    maximum_bytes,
    maximum_expanded_bytes,
    maximum_members=6000,
):
    """Authenticate one bounded snapshot before inspecting its ZIP members."""
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("an independent archive SHA-256 is required")
    if any(
        type(value) is not int or value <= 0
        for value in (maximum_bytes, maximum_expanded_bytes, maximum_members)
    ):
        raise ValueError("archive limits must be positive integers")
    raw = read_regular_file_bytes(
        Path(path), label="evaluator reference archive", max_bytes=maximum_bytes
    )
    if sha(raw) != expected_sha256:
        raise ValueError("archive SHA-256 mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if (
            len(entries) > maximum_members
            or sum(item.file_size for item in entries) > maximum_expanded_bytes
        ):
            raise ValueError("archive exceeds its retained member or expanded limit")
        names = set()
        for item in entries:
            name = PurePosixPath(item.filename)
            if (
                item.filename in names
                or name.is_absolute()
                or ".." in name.parts
                or str(name) != item.filename
                or "\\" in item.filename
                or ":" in item.filename
                or not stat.S_ISREG(item.external_attr >> 16)
            ):
                raise ValueError("archive contains an unsafe or duplicate member")
            names.add(item.filename)
        return {item.filename: archive.read(item) for item in entries}


def read_reference(directory):
    files = {}
    for name, pin in ARCHIVES.items():
        incoming = read_archive(
            Path(directory) / name,
            expected_sha256=pin,
            maximum_bytes=ARCHIVE_LIMIT,
            maximum_expanded_bytes=EXPANDED_LIMIT,
        )
        if files.keys() & incoming.keys():
            raise ValueError("reference archives contain overlapping members")
        files.update(incoming)
    manifest = json.loads(files["archive-manifest.json"])
    if set(manifest["files"]) != set(files) - {"archive-manifest.json"}:
        raise ValueError("reference member inventory differs")
    for name, expected in manifest["files"].items():
        if expected["sha256"] != "sha256:" + sha(files[name]) or expected[
            "bytes"
        ] != len(files[name]):
            raise ValueError(f"reference member hash differs: {name}")
    return files


def verify_pack(root, row, receipt_directory, key_bytes):
    from invarlock.captured_verification import verify_captured_evidence
    from invarlock.evidence_receipt import verify_signed_verification_receipt
    from invarlock.trust_inputs import load_trust_inputs

    directory = root / row["directory"]
    anchors = row["anchors"]
    profile = load_trust_inputs(
        directory / "trust.json",
        verifier_key_bytes_override=row["receipt_public_key"].encode(),
    )
    if profile.profile_digest != row["verifier"]["trust_profile_digest"]:
        raise ValueError("historical trust profile differs")
    original = verify_signed_verification_receipt(
        directory / "verification.receipt.json",
        directory / "evidence",
        policy_path=directory / "policy.json",
        expected_run_digests={
            "baseline": anchors["baseline_run_digest"],
            "subject": anchors["subject_run_digest"],
        },
        expected_request_digest=anchors["request_digest"],
        expected_pack_signer_fingerprint=anchors["evidence_signer_fingerprint"],
        expected_verifier_identity=row["verifier"]["identity"],
        expected_verifier_fingerprint=row["verifier"]["signing_key_fingerprint"],
        expected_trust_profile_digest=profile.profile_digest,
    )
    if not original.ok:
        raise ValueError(f"original receipt failed: {original.errors}")
    # A fresh ephemeral verifier signs the new replay; no historical private
    # key is supplied. Its receipt exists only in the temporary replay directory.
    result = verify_captured_evidence(
        directory / "evidence",
        policy_path=directory / "policy.json",
        expected_baseline_run=anchors["baseline_run_digest"],
        expected_subject_run=anchors["subject_run_digest"],
        expected_request_digest=anchors["request_digest"],
        expected_signer=anchors["evidence_signer_fingerprint"],
        receipt_path=receipt_directory / (sha(row["id"].encode()) + ".json"),
        verifier_signing_key_path=None,
        verifier_signing_key_bytes=key_bytes,
        verifier_identity="offline-reference-replay",
    )
    if any(result[name] != value for name, value in row["expected"].items()):
        raise ValueError("replayed result differs from retained expectations")
    if any(
        original.statement["verdict"][name] != result[name]
        for name in ("decision", "integrity_ok", "policy_verdict", "ok")
    ):
        raise ValueError("original receipt verdict differs from replay")
    if original.statement["verdict"]["verification_status"] != row["expected_status"]:
        raise ValueError("historical verification status differs")
    return {
        "id": row["id"],
        "integrity_ok": result["integrity_ok"],
        "receipt_authenticated": original.ok,
        "decision": result["decision"],
        "policy_verdict": result["policy_verdict"],
        "verification_status": row["expected_status"],
    }


def replay(directory):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    files = read_reference(directory)
    reference = json.loads(files["reference.json"])
    key_bytes = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    with tempfile.TemporaryDirectory(prefix="evaluator-sentinel-") as temporary:
        root = Path(temporary).resolve()
        for name, raw in files.items():
            path = root.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        receipt_directory = root / "fresh-receipts"
        receipt_directory.mkdir()
        results = [
            verify_pack(root, row, receipt_directory, key_bytes)
            for row in reference["packs"]
        ]
    return {
        "ok": True,
        "evaluator_count": len(reference["captures"]),
        "signed_packs_replayed": len(results),
        "new_model_or_judge_calls": 0,
        "historical_policy_acceptance": False,
        "results": results,
    }


def block_network(event, _args):
    if event in {
        "socket.connect",
        "socket.getaddrinfo",
        "socket.sendto",
        "socket.sendmsg",
    }:
        raise RuntimeError("reference replay forbids outbound network")


def require_installed():
    import invarlock

    if (
        not Path(invarlock.__file__)
        .resolve()
        .is_relative_to(Path(get_path("purelib")).resolve())
    ):
        raise ValueError("replay requires an installed recipient outside the checkout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    sys.addaudithook(block_network)
    require_installed()
    result = replay(args.directory)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
