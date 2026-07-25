from __future__ import annotations

import base64
import hashlib
import json
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.acceptance_attestation import (
    ACCEPTANCE_PREDICATE_FORMAT,
    ACCEPTANCE_PREDICATE_TYPE,
    DSSE_PAYLOAD_TYPE,
    IN_TOTO_STATEMENT_TYPE,
    RECIPIENT_POLICY_FORMAT,
    verify_acceptance_attestation,
    write_acceptance_attestation,
)
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.public_contracts import (
    load_acceptance_predicate_schema,
    load_recipient_acceptance_policy_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    REPO_ROOT / "examples/artifacts/trust-boundary-demo/evaluation/artifacts/evidence"
)
RECEIPT = (
    REPO_ROOT
    / "examples/artifacts/trust-boundary-demo/verifier/receipts/accepted.receipt.json"
)
ISSUED_AT = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
SUBJECT_DIGEST = "sha256:" + "a" * 64


def _key(tmp_path: Path, name: str, seed: int = 7) -> tuple[Path, Path, str]:
    key = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes((seed + offset) % 256 for offset in range(32))
    )
    private = tmp_path / f"{name}.private.pem"
    public = tmp_path / f"{name}.public.pem"
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private, public, public_key_fingerprint(key.public_key())


def _envelope(
    tmp_path: Path,
    *,
    signer_identity: str = "producer.example/release-assurance",
    seed: int = 7,
) -> tuple[Path, Path, str]:
    private, public, fingerprint = _key(tmp_path, "producer", seed)
    envelope = tmp_path / "acceptance.dsse.json"
    write_acceptance_attestation(
        RECEIPT,
        EVIDENCE,
        envelope,
        signing_key_path=private,
        signer_identity=signer_identity,
        policy_identity="producer.example/policies/release-regression-v3",
        issued_at=ISSUED_AT,
        evaluation_completed_at=datetime(2026, 7, 25, 11, 55, tzinfo=UTC),
    )
    return envelope, public, fingerprint


def _policy(
    fingerprint: str,
    *,
    identity: str = "producer.example/release-assurance",
    status: str = "active",
    max_age_seconds: int = 3600,
    versions: list[str] | None = None,
    allow_countersigned: bool = True,
) -> dict[str, Any]:
    return {
        "format": RECIPIENT_POLICY_FORMAT,
        "expected_predicate_type": ACCEPTANCE_PREDICATE_TYPE,
        "trusted_signers": [
            {
                "identity": identity,
                "fingerprint": fingerprint,
                "status": status,
            }
        ],
        "freshness": {
            "max_age_seconds": max_age_seconds,
            "clock_skew_seconds": 0,
        },
        "allowed_contract_versions": versions or ["0.13.0"],
        "required_technical_verdict": "pass",
        "allow_countersigned_receipts": allow_countersigned,
    }


def _payload(envelope: Path) -> dict[str, Any]:
    outer = json.loads(envelope.read_bytes())
    return json.loads(base64.b64decode(outer["payload"], validate=True))


def _resign(envelope: Path, private: Path, statement: dict[str, Any]) -> None:
    outer = json.loads(envelope.read_bytes())
    payload = (
        json.dumps(
            statement,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    payload_type = outer["payloadType"].encode()
    pae = (
        b"DSSEv1 "
        + str(len(payload_type)).encode()
        + b" "
        + payload_type
        + b" "
        + str(len(payload)).encode()
        + b" "
        + payload
    )
    key = serialization.load_pem_private_key(private.read_bytes(), password=None)
    assert isinstance(key, ed25519.Ed25519PrivateKey)
    outer["payload"] = base64.b64encode(payload).decode("ascii")
    outer["signatures"][0]["sig"] = base64.b64encode(key.sign(pae)).decode("ascii")
    envelope.chmod(stat.S_IMODE(envelope.stat().st_mode) | stat.S_IWUSR)
    envelope.write_text(json.dumps(outer), encoding="utf-8")


def test_predicate_and_policy_schemas_are_packaged_and_closed(
    tmp_path: Path,
) -> None:
    envelope, _public, fingerprint = _envelope(tmp_path)
    statement = _payload(envelope)
    predicate = statement["predicate"]
    policy = _policy(fingerprint)

    jsonschema.Draft202012Validator(load_acceptance_predicate_schema()).validate(
        predicate
    )
    jsonschema.Draft202012Validator(load_recipient_acceptance_policy_schema()).validate(
        policy
    )
    assert load_acceptance_predicate_schema()["additionalProperties"] is False
    assert load_recipient_acceptance_policy_schema()["additionalProperties"] is False


def test_v013_receipt_wraps_without_relabelling_and_binds_exact_subject(
    tmp_path: Path,
) -> None:
    envelope, _public, fingerprint = _envelope(tmp_path)
    outer = json.loads(envelope.read_bytes())
    statement = _payload(envelope)
    predicate = statement["predicate"]

    assert outer["payloadType"] == DSSE_PAYLOAD_TYPE
    assert outer["signatures"][0]["keyid"] == fingerprint
    assert statement["_type"] == IN_TOTO_STATEMENT_TYPE
    assert statement["predicateType"] == ACCEPTANCE_PREDICATE_TYPE
    assert statement["subject"] == [
        {"name": "org/subject", "digest": {"sha256": "a" * 64}}
    ]
    assert predicate["format"] == ACCEPTANCE_PREDICATE_FORMAT
    assert predicate["subject"]["artifact_digest"] == SUBJECT_DIGEST
    assert predicate["contracts"] == {
        "invarlock_release": "0.13.0",
        "evidence_pack": "invarlock/evidence-pack-v1",
        "comparison_report": "invarlock/comparison-report-v2",
        "receipt": "invarlock/evidence-verification-receipt-v1",
    }
    assert (
        predicate["receipt"]["content"]["statement"]["format"]
        == "invarlock/evidence-verification-receipt-v1"
    )
    assert predicate["receipt"]["representation"] == "embedded"
    assert predicate["signers"]["relationship"] == "countersigned"


def test_canonical_statement_bytes_are_independently_reproducible(
    tmp_path: Path,
) -> None:
    envelope, _public, _fingerprint = _envelope(tmp_path)
    outer = json.loads(envelope.read_bytes())
    statement = _payload(envelope)
    expected = (
        json.dumps(
            statement,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()

    assert base64.b64decode(outer["payload"], validate=True) == expected
    receipt = statement["predicate"]["receipt"]
    expected_receipt = (
        json.dumps(
            receipt["content"],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    assert receipt["digest"] == "sha256:" + hashlib.sha256(expected_receipt).hexdigest()


def test_recipient_authenticates_and_accepts_with_its_current_policy(
    tmp_path: Path,
) -> None:
    envelope, public, fingerprint = _envelope(tmp_path)

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT + timedelta(minutes=10),
    )

    assert decision.envelope_authenticated is True
    assert decision.receipt_authenticated is True
    assert decision.subject_bound is True
    assert decision.historical_technical_verdict == "pass"
    assert decision.accepted is True
    assert decision.errors == ()


def test_valid_technical_receipt_can_fail_stricter_current_policy(
    tmp_path: Path,
) -> None:
    envelope, public, fingerprint = _envelope(tmp_path)

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint, versions=["0.14.0"]),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT + timedelta(minutes=10),
    )

    assert decision.envelope_authenticated is True
    assert decision.receipt_authenticated is True
    assert decision.historical_technical_verdict == "pass"
    assert decision.accepted is False
    assert "contract version is not allowed" in " ".join(decision.errors)


def test_wrong_artifact_is_rejected_even_with_valid_receipt(
    tmp_path: Path,
) -> None:
    envelope, public, fingerprint = _envelope(tmp_path)

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint),
        expected_subject_digest="sha256:" + "9" * 64,
        now=ISSUED_AT,
    )

    assert decision.envelope_authenticated is True
    assert decision.subject_bound is False
    assert decision.accepted is False
    assert "subject digest does not match" in " ".join(decision.errors)


def test_tampered_envelope_is_rejected(tmp_path: Path) -> None:
    envelope, public, fingerprint = _envelope(tmp_path)
    outer = json.loads(envelope.read_bytes())
    outer["payload"] = outer["payload"][:-2] + "AA"
    envelope.chmod(stat.S_IMODE(envelope.stat().st_mode) | stat.S_IWUSR)
    envelope.write_text(json.dumps(outer), encoding="utf-8")

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT,
    )

    assert decision.envelope_authenticated is False
    assert decision.accepted is False


@pytest.mark.parametrize(
    ("policy_fingerprint", "status", "message"),
    [
        ("sha256:" + "9" * 64, "active", "not trusted"),
        (None, "revoked", "revoked"),
    ],
)
def test_unknown_or_revoked_envelope_signer_is_rejected(
    tmp_path: Path,
    policy_fingerprint: str | None,
    status: str,
    message: str,
) -> None:
    envelope, public, fingerprint = _envelope(tmp_path)
    selected = policy_fingerprint or fingerprint

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(selected, status=status),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT,
    )

    assert decision.accepted is False
    assert message in " ".join(decision.errors)


def test_stale_attestation_is_rejected_by_configured_freshness(
    tmp_path: Path,
) -> None:
    envelope, public, fingerprint = _envelope(tmp_path)

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint, max_age_seconds=60),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT + timedelta(seconds=61),
    )

    assert decision.envelope_authenticated is True
    assert decision.accepted is False
    assert "stale" in " ".join(decision.errors)


def test_contradictory_receipt_and_predicate_is_rejected_even_when_resigned(
    tmp_path: Path,
) -> None:
    private, public, fingerprint = _key(tmp_path, "producer")
    envelope = tmp_path / "acceptance.dsse.json"
    write_acceptance_attestation(
        RECEIPT,
        EVIDENCE,
        envelope,
        signing_key_path=private,
        signer_identity="producer.example/release-assurance",
        policy_identity="producer.example/policies/release-regression-v3",
        issued_at=ISSUED_AT,
    )
    statement = _payload(envelope)
    statement["predicate"]["technical_verdict"]["policy_verdict"] = "fail"
    _resign(envelope, private, statement)

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT,
    )

    assert decision.envelope_authenticated is True
    assert decision.receipt_authenticated is True
    assert decision.accepted is False
    assert "technical verdict disagrees" in " ".join(decision.errors)


def test_inner_receipt_tampering_is_rejected_even_when_outer_envelope_is_resigned(
    tmp_path: Path,
) -> None:
    private, public, fingerprint = _key(tmp_path, "producer")
    envelope = tmp_path / "acceptance.dsse.json"
    write_acceptance_attestation(
        RECEIPT,
        EVIDENCE,
        envelope,
        signing_key_path=private,
        signer_identity="producer.example/release-assurance",
        policy_identity="producer.example/policies/release-regression-v3",
        issued_at=ISSUED_AT,
    )
    statement = _payload(envelope)
    statement["predicate"]["receipt"]["content"]["statement"]["verdict"]["ok"] = False
    receipt = statement["predicate"]["receipt"]
    canonical_receipt = (
        json.dumps(
            receipt["content"],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    receipt["digest"] = "sha256:" + hashlib.sha256(canonical_receipt).hexdigest()
    _resign(envelope, private, statement)

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT,
    )

    assert decision.envelope_authenticated is True
    assert decision.receipt_authenticated is False
    assert decision.accepted is False
    assert "receipt signature verification failed" in " ".join(decision.errors)


def test_countersigned_receipt_relationship_is_current_policy_control(
    tmp_path: Path,
) -> None:
    envelope, public, fingerprint = _envelope(tmp_path)

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint, allow_countersigned=False),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT,
    )

    assert decision.envelope_authenticated is True
    assert decision.receipt_authenticated is True
    assert decision.accepted is False
    assert "countersigned receipts are not allowed" in " ".join(decision.errors)
