#!/usr/bin/env python3
"""Verify the pinned container reference with an installed InvarLock recipient."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives.serialization import load_pem_public_key

from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_json import read_regular_file_bytes
from invarlock.evidence_pack_verification import verify_comparison_evidence
from invarlock.evidence_receipt import verify_signed_verification_receipt
from invarlock.judge_measurements.acceptance import (
    replay_signed_judge_verification_receipt,
    verify_judge_evidence_with_policy,
)
from invarlock.trust_inputs import _canonical_json_bytes

ARCHIVE_SHA256 = "6e8fafb37a5e5c95a76351c3a35e4f01b6e471caa9b09091d61c447a590dd6ab"


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def verify_row(root: Path, row: dict) -> dict:
    pack, policy = root / row["pack"], root / row["policy"]
    receipt_path = root / row["receipt"]
    fingerprint = public_key_fingerprint(
        load_pem_public_key((root / row["verifier_public_key"]).read_bytes())
    )
    if row["kind"] == "judge":
        result = verify_judge_evidence_with_policy(pack, policy).to_dict()
        receipt = replay_signed_judge_verification_receipt(
            receipt_path,
            evidence_path=pack,
            recipient_policy_path=policy,
            expected_verifier_identity="native-judge-container-recipient",
            expected_verifier_fingerprint=fingerprint,
        )
        accepted = all(
            result[k] is True for k in ("accepted", "authenticated", "replayed")
        )
        return {
            "name": row["name"],
            "evidence_accepted": accepted,
            "receipt_ok": receipt.verified and receipt.accepted,
            "receipt_errors": list(receipt.errors),
        }
    else:
        profile = read(root / row["profile"])
        anchors = profile["anchors"]
        expected = {
            "policy_path": policy,
            "expected_artifact_digests": {
                role: anchors[f"{role}_artifact_digest"]
                for role in ("baseline", "subject")
            },
            "expected_schedule_digest": anchors["schedule_digest"],
            "expected_runtime_digests": {
                role: anchors[f"{role}_runtime_digest"]
                for role in ("baseline", "subject")
            },
            "expected_request_digest": anchors.get("request_digest"),
        }
        result = verify_comparison_evidence(
            pack,
            **expected,
            expected_signer_fingerprint=anchors["evidence_signer_fingerprint"],
        )
        accepted = result.status == 0 and result.payload.get("ok") is True
        receipt = verify_signed_verification_receipt(
            receipt_path,
            pack,
            **expected,
            expected_pack_signer_fingerprint=anchors["evidence_signer_fingerprint"],
            expected_verifier_identity=profile["verifier"]["identity"],
            expected_verifier_fingerprint=fingerprint,
            expected_trust_profile_digest="sha256:"
            + hashlib.sha256(_canonical_json_bytes(profile)).hexdigest(),
        )
    return {
        "name": row["name"],
        "evidence_accepted": accepted,
        "receipt_ok": receipt.ok,
        "receipt_errors": list(receipt.errors),
    }


def replay(archive: Path) -> dict:
    # Authenticate the complete, fixed archive before any extraction. This is a
    # replay of one reviewed reference, not a general archive import interface.
    archive_bytes = read_regular_file_bytes(
        archive, label="container reference archive", max_bytes=400_000
    )
    if hashlib.sha256(archive_bytes).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("container reference archive SHA-256 mismatch")
    with tempfile.TemporaryDirectory(prefix="invarlock-container-reference-") as temp:
        root = Path(temp).resolve()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as stream:
            stream.extractall(root)
        manifest = read(root / "manifest.json")
        results = [verify_row(root, row) for row in manifest["rows"]]
    return {
        "pack_count": len(results),
        "ok": len(results) == 14
        and all(row["evidence_accepted"] and row["receipt_ok"] for row in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path(__file__).with_name("reference.zip")
    )
    args = parser.parse_args()
    result = replay(args.archive)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
