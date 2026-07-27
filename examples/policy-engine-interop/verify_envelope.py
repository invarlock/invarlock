#!/usr/bin/env python3
"""Verify the generic DSSE/receipt cryptography and emit policy-engine input.

This reference verifier is intentionally standalone: it does not import or call
InvarLock. OPA and CUE consume its authenticated projection through stdin or a
JSON file.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

INPUT_FORMAT = "invarlock/acceptance-policy-input-v1"
PAYLOAD_TYPE = "application/vnd.in-toto+json"


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def decode_base64(value: object, *, label: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise ValueError(f"{label} is invalid base64") from exc


def fingerprint(key: ed25519.Ed25519PublicKey) -> str:
    raw = key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def load_ed25519_public_key(payload: bytes) -> ed25519.Ed25519PublicKey:
    key = serialization.load_pem_public_key(payload)
    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise ValueError("public key must be Ed25519")
    return key


def pae(payload_type: str, payload: bytes) -> bytes:
    type_bytes = payload_type.encode()
    return (
        b"DSSEv1 "
        + str(len(type_bytes)).encode()
        + b" "
        + type_bytes
        + b" "
        + str(len(payload)).encode()
        + b" "
        + payload
    )


def unix_seconds(value: str) -> int:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return int(parsed.astimezone(UTC).timestamp())


def verify_envelope(
    *,
    envelope_path: Path,
    envelope_key_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope = strict_json_bytes(envelope_path.read_bytes(), label="DSSE envelope")
    if set(envelope) != {"payload", "payloadType", "signatures"}:
        raise ValueError("DSSE envelope fields are invalid")
    if envelope["payloadType"] != PAYLOAD_TYPE:
        raise ValueError("DSSE payload type is unsupported")
    signatures = envelope["signatures"]
    if not isinstance(signatures, list) or len(signatures) != 1:
        raise ValueError("DSSE envelope must contain exactly one signature")
    signature = signatures[0]
    if not isinstance(signature, dict) or set(signature) != {"keyid", "sig"}:
        raise ValueError("DSSE signature fields are invalid")
    payload = decode_base64(envelope["payload"], label="DSSE payload")
    statement = strict_json_bytes(payload, label="in-toto Statement")
    if payload != canonical_bytes(statement):
        raise ValueError("in-toto Statement must use canonical JSON")
    key = load_ed25519_public_key(envelope_key_path.read_bytes())
    observed_fingerprint = fingerprint(key)
    if signature["keyid"] != observed_fingerprint:
        raise ValueError("DSSE key ID does not match the supplied public key")
    key.verify(
        decode_base64(signature["sig"], label="DSSE signature"),
        pae(PAYLOAD_TYPE, payload),
    )
    return statement, {
        "envelope_signer_fingerprint": observed_fingerprint,
        "envelope_signature": True,
    }


def verify_receipt(statement: dict[str, Any]) -> dict[str, Any]:
    predicate = statement["predicate"]
    receipt = predicate["receipt"]
    raw = decode_base64(receipt["raw_base64"], label="embedded receipt")
    if receipt["digest"] != f"sha256:{hashlib.sha256(raw).hexdigest()}":
        raise ValueError("embedded receipt digest is invalid")
    content = strict_json_bytes(raw, label="embedded receipt")
    if raw != canonical_bytes(content) or content != receipt["content"]:
        raise ValueError("embedded receipt representation is inconsistent")
    signature = content["signature"]
    receipt_statement = content["statement"]
    public_block = signature["public_key"]
    if public_block.get("encoding") != "pem":
        raise ValueError("receipt public key encoding is unsupported")
    key = load_ed25519_public_key(public_block["value"].encode())
    observed_fingerprint = fingerprint(key)
    key.verify(
        decode_base64(signature["value"], label="receipt signature"),
        canonical_bytes(receipt_statement),
    )
    verifier = receipt_statement["verifier"]
    signers = predicate["signers"]
    technical = predicate["technical_verdict"]
    if (
        verifier["signing_key_fingerprint"] != observed_fingerprint
        or signers["receipt"]["fingerprint"] != observed_fingerprint
        or signers["receipt"]["identity"] != verifier["identity"]
        or receipt_statement["verdict"] != technical
    ):
        raise ValueError("signed receipt projection is inconsistent")
    return {
        "receipt_signature": True,
        "receipt_verifier_fingerprint": observed_fingerprint,
        "receipt_verifier_identity": verifier["identity"],
    }


def build_policy_input(
    *,
    envelope_path: Path,
    envelope_key_path: Path,
    recipient_policy_path: Path,
    expected_subject_name: str,
    expected_subject_sha256: str,
    now: str,
) -> dict[str, Any]:
    statement, envelope_verification = verify_envelope(
        envelope_path=envelope_path,
        envelope_key_path=envelope_key_path,
    )
    receipt_verification = verify_receipt(statement)
    policy = strict_json_bytes(
        recipient_policy_path.read_bytes(),
        label="recipient policy",
    )
    predicate = statement["predicate"]
    signers = predicate["signers"]
    if (
        signers["envelope"]["fingerprint"]
        != envelope_verification["envelope_signer_fingerprint"]
    ):
        raise ValueError("signed envelope-signer projection is inconsistent")
    issued_at = predicate["timestamps"]["attestation_issued_at"]
    return {
        "authentication": {
            "envelope_signature": True,
            "projection_consistent": True,
            "receipt_signature": True,
        },
        "format": INPUT_FORMAT,
        "recipient": {
            "allowed_contract_versions": policy["allowed_contract_versions"],
            "expected_predicate_type": policy["expected_predicate_type"],
            "expected_subject": {
                "name": expected_subject_name,
                "sha256": expected_subject_sha256,
            },
            "max_attestation_age_seconds": policy["freshness"][
                "max_envelope_age_seconds"
            ],
            "required_technical_verdict": policy["required_technical_verdict"],
            "trusted_receipt_verifiers": policy["trusted_receipt_verifiers"],
            "trusted_signers": policy["trusted_signers"],
        },
        "statement": statement,
        "verified": {
            "attestation_issued_at_unix": unix_seconds(issued_at),
            "envelope_signer_fingerprint": envelope_verification[
                "envelope_signer_fingerprint"
            ],
            "envelope_signer_identity": signers["envelope"]["identity"],
            "now_unix": unix_seconds(now),
            **receipt_verification,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--envelope-key", type=Path, required=True)
    parser.add_argument("--recipient-policy", type=Path, required=True)
    parser.add_argument("--expected-subject-name", required=True)
    parser.add_argument("--expected-subject-sha256", required=True)
    parser.add_argument("--now", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_policy_input(
        envelope_path=args.envelope,
        envelope_key_path=args.envelope_key,
        recipient_policy_path=args.recipient_policy,
        expected_subject_name=args.expected_subject_name,
        expected_subject_sha256=args.expected_subject_sha256,
        now=args.now,
    )
    print(canonical_bytes(result).decode(), end="")


if __name__ == "__main__":
    main()
