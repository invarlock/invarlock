#!/usr/bin/env python3
"""Replay retained ModelKit evidence without models, containers, or private keys."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from invarlock.acceptance_attestation import verify_acceptance_attestation
from invarlock.evidence_pack_json import read_regular_file_bytes
from invarlock.evidence_pack_verification import verify_comparison_evidence
from invarlock.evidence_receipt import verify_signed_verification_receipt

ARCHIVE_SHA256 = "bd8d73d8ad01ede51f36939ecf956a605893d763e074fd3a2a82d3d802cc9348"
ARCHIVE_LIMIT = 2 * 1024 * 1024


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def verify_snapshot(root: Path) -> dict:
    reference = read(root / "reference.json")
    anchors = reference["technical_anchors"]
    options = {
        "policy_path": root / "technical-policy.json",
        "expected_artifact_digests": anchors["artifact_digests"],
        "expected_schedule_digest": anchors["schedule_digest"],
        "expected_runtime_digests": anchors["runtime_digests"],
        "expected_request_digest": anchors["request_digest"],
    }
    technical = verify_comparison_evidence(
        root / "evidence",
        **options,
        expected_signer_fingerprint=anchors["evidence_signer_fingerprint"],
    )
    verifier = reference["verifier"]
    receipt = verify_signed_verification_receipt(
        root / "verification.receipt.json",
        root / "evidence",
        **options,
        expected_pack_signer_fingerprint=anchors["evidence_signer_fingerprint"],
        expected_verifier_identity=verifier["identity"],
        expected_verifier_fingerprint=verifier["signing_key_fingerprint"],
        expected_trust_profile_digest=verifier["trust_profile_digest"],
    )
    envelope = verify_acceptance_attestation(
        root / "acceptance.dsse.json",
        trusted_public_keys={
            reference["envelope_signer_fingerprint"]: root / "envelope-public.pem"
        },
        recipient_policy=root / "recipient-policy.json",
        expected_subject_digest=reference["model_content_digests"]["subject"],
        now=datetime.fromisoformat(reference["historical_replay_at"]),
    )
    statement = envelope.statement or {}
    predicate = statement.get("predicate", {})
    bound = predicate.get("receipt", {}).get("content") == read(
        root / "verification.receipt.json"
    ) and all(
        predicate.get(role, {}).get("artifact_digest")
        == reference["model_content_digests"][role]
        and predicate.get(role, {}).get("artifact_identity_digest")
        == anchors["artifact_digests"][role]
        for role in ("baseline", "subject")
    )
    expected = reference["expected"]
    result = {
        "integrity_ok": technical.payload.get("integrity_ok") is True,
        "verification_status": technical.status,
        "policy_verdict": technical.payload.get("policy_verdict"),
        "receipt_ok": receipt.ok,
        "envelope_authenticated": envelope.envelope_authenticated,
        "embedded_receipt_authenticated": envelope.receipt_authenticated,
        "subject_digest_bound": envelope.subject_bound,
        "envelope_evidence_bound": bound,
        "historical_recipient_accepted": envelope.accepted,
        "historical_technical_verdict": envelope.historical_technical_verdict,
        "envelope_errors": list(envelope.errors),
        "receipt_errors": list(receipt.errors),
        "scope": "Historical evidence replay; package bytes and inference are not rerun",
    }
    result["ok"] = (
        all(
            result[key]
            for key in (
                "integrity_ok",
                "receipt_ok",
                "envelope_authenticated",
                "embedded_receipt_authenticated",
                "subject_digest_bound",
                "envelope_evidence_bound",
            )
        )
        and result["verification_status"] == expected["verification_status"]
        and result["policy_verdict"] == expected["policy_verdict"]
        and envelope.historical_technical_verdict == expected["policy_verdict"]
        and envelope.accepted == expected["recipient_accepted"]
        and list(envelope.errors) == expected["envelope_errors"]
    )
    return result


def replay(archive: Path) -> dict:
    snapshot = read_regular_file_bytes(
        archive, label="ModelKit reference archive", max_bytes=ARCHIVE_LIMIT
    )
    if hashlib.sha256(snapshot).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("ModelKit reference archive SHA-256 mismatch")
    # This fixed archive is authenticated in full before extraction; it is not
    # a general-purpose archive import or a source of current acceptance policy.
    with tempfile.TemporaryDirectory(prefix="invarlock-modelkit-reference-") as temp:
        root = Path(temp).resolve()
        with zipfile.ZipFile(io.BytesIO(snapshot)) as stream:
            stream.extractall(root)
        return verify_snapshot(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path(__file__).with_name("reference.zip")
    )
    result = replay(parser.parse_args().archive)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
