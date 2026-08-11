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
GOLDEN = REPO_ROOT / "examples/acceptance-handoff/golden"
EVIDENCE = GOLDEN / "evidence"
RECEIPT = GOLDEN / "verification.receipt.json"
ISSUED_AT = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
SUBJECT_DIGEST = (
    "sha256:a9fcf5a7cb042b0f4db67dead3d64fad8c3775d7ea25c91ee6759b019b5603cb"
)
RECEIPT_VERIFIER_IDENTITY = "verifier.example/release-qualification"
RECEIPT_VERIFIER_FINGERPRINT = (
    "sha256:74a97c1d8fe8d7d58faac074d3a3a9267d8db501d9e4aed77eaeb9ad4efb32ff"
)


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
    signer_identity: str = "envelope-signer.example/release-assurance",
    seed: int = 7,
) -> tuple[Path, Path, str]:
    private, public, fingerprint = _key(tmp_path, "envelope-signer", seed)
    envelope = tmp_path / "acceptance.dsse.json"
    write_acceptance_attestation(
        RECEIPT,
        EVIDENCE,
        envelope,
        signing_key_path=private,
        signer_identity=signer_identity,
        policy_identity="evaluation.example/policies/release-regression-v3",
        issued_at=ISSUED_AT,
        evaluation_completed_at=datetime(2026, 7, 25, 11, 55, tzinfo=UTC),
    )
    return envelope, public, fingerprint


def _policy(
    fingerprint: str,
    *,
    identity: str = "envelope-signer.example/release-assurance",
    status: str = "active",
    max_age_seconds: int = 3600,
    max_evidence_age_seconds: int | None = None,
    versions: list[str] | None = None,
    allow_countersigned: bool = True,
    receipt_identity: str = RECEIPT_VERIFIER_IDENTITY,
    receipt_fingerprint: str = RECEIPT_VERIFIER_FINGERPRINT,
    receipt_status: str = "active",
    expected_receipt_trust_profile_digest: str | None = None,
) -> dict[str, Any]:
    policy = {
        "format": RECIPIENT_POLICY_FORMAT,
        "expected_predicate_type": ACCEPTANCE_PREDICATE_TYPE,
        "trusted_signers": [
            {
                "identity": identity,
                "fingerprint": fingerprint,
                "status": status,
            }
        ],
        "trusted_receipt_verifiers": [
            {
                "identity": receipt_identity,
                "fingerprint": receipt_fingerprint,
                "status": receipt_status,
            }
        ],
        "freshness": {
            "max_envelope_age_seconds": max_age_seconds,
            "max_evidence_age_seconds": max_evidence_age_seconds,
            "clock_skew_seconds": 0,
        },
        "allowed_contract_versions": versions or ["0.15.0"],
        "required_technical_verdict": "pass",
        "allow_countersigned_receipts": allow_countersigned,
    }
    if expected_receipt_trust_profile_digest is not None:
        policy["expected_receipt_trust_profile_digest"] = (
            expected_receipt_trust_profile_digest
        )
    return policy


def _payload(envelope: Path) -> dict[str, Any]:
    outer = json.loads(envelope.read_bytes())
    return json.loads(base64.b64decode(outer["payload"], validate=True))


def _canonical(value: object) -> bytes:
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


def _resign_embedded_receipt(
    receipt: dict[str, Any],
    key: ed25519.Ed25519PrivateKey,
    *,
    identity: str,
    trust_profile_digest: str | None = None,
) -> None:
    fingerprint = public_key_fingerprint(key.public_key())
    receipt["statement"]["verifier"] = {
        "identity": identity,
        "signing_key_fingerprint": fingerprint,
        "trust_profile_digest": trust_profile_digest,
    }
    receipt["signature"] = {
        "format": "invarlock/evidence-verification-receipt-signature-v1",
        "algorithm": "ed25519",
        "public_key": {
            "encoding": "pem",
            "value": key.public_key()
            .public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode("ascii"),
        },
        "value": base64.b64encode(key.sign(_canonical(receipt["statement"]))).decode(
            "ascii"
        ),
    }


def _replace_embedded_receipt(
    block: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    raw = _canonical(receipt)
    block["content"] = receipt
    block["digest"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    if "raw_base64" in block:
        block["raw_base64"] = base64.b64encode(raw).decode("ascii")


def _resign(envelope: Path, private: Path, statement: dict[str, Any]) -> None:
    outer = json.loads(envelope.read_bytes())
    payload = _canonical(statement)
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


@pytest.mark.parametrize(
    "registry",
    ["trusted_signers", "trusted_receipt_verifiers"],
)
def test_recipient_policy_schema_rejects_exact_duplicate_trust_records(
    tmp_path: Path,
    registry: str,
) -> None:
    _envelope_path, _public, fingerprint = _envelope(tmp_path)
    policy = _policy(fingerprint)
    policy[registry].append(dict(policy[registry][0]))

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(
            load_recipient_acceptance_policy_schema()
        ).validate(policy)


def test_current_receipt_wraps_without_relabelling_and_binds_exact_subject(
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
        {
            "name": "artifact.example/subject",
            "digest": {
                "sha256": (
                    "a9fcf5a7cb042b0f4db67dead3d64fad8c3775d7ea25c91ee6759b019b5603cb"
                )
            },
        }
    ]
    assert predicate["format"] == ACCEPTANCE_PREDICATE_FORMAT
    assert predicate["subject"]["artifact_digest"] == SUBJECT_DIGEST
    assert predicate["contracts"] == {
        "invarlock_release": "0.15.0",
        "evidence_pack": "invarlock/evidence-pack-v1",
        "comparison_report": "invarlock/comparison-report-v3",
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


@pytest.mark.parametrize(
    "registry",
    ["trusted_signers", "trusted_receipt_verifiers"],
)
@pytest.mark.parametrize(
    "statuses",
    [("active", "revoked"), ("revoked", "active")],
)
def test_duplicate_trust_pair_is_rejected_independent_of_record_order(
    tmp_path: Path,
    registry: str,
    statuses: tuple[str, str],
) -> None:
    envelope, public, fingerprint = _envelope(tmp_path)
    policy = _policy(fingerprint)
    original = policy[registry][0]
    original["status"] = statuses[0]
    policy[registry].append({**original, "status": statuses[1]})

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=policy,
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT,
    )

    assert decision.envelope_authenticated is True
    assert decision.accepted is False
    assert "duplicate identity/fingerprint pair" in " ".join(decision.errors)


def test_trusted_envelope_signer_cannot_introduce_unknown_receipt_verifier(
    tmp_path: Path,
) -> None:
    private, public, fingerprint = _key(tmp_path, "envelope-signer")
    envelope = tmp_path / "acceptance.dsse.json"
    write_acceptance_attestation(
        RECEIPT,
        EVIDENCE,
        envelope,
        signing_key_path=private,
        signer_identity="envelope-signer.example/release-assurance",
        policy_identity="evaluation.example/policies/release-regression-v3",
        issued_at=ISSUED_AT,
    )
    statement = _payload(envelope)
    receipt_block = statement["predicate"]["receipt"]
    receipt = receipt_block["content"]
    unknown_key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    _resign_embedded_receipt(
        receipt,
        unknown_key,
        identity="unknown.example/technical-verifier",
    )
    unknown_fingerprint = public_key_fingerprint(unknown_key.public_key())
    _replace_embedded_receipt(receipt_block, receipt)
    statement["predicate"]["signers"]["receipt"] = {
        "identity": "unknown.example/technical-verifier",
        "fingerprint": unknown_fingerprint,
    }
    statement["predicate"]["signers"]["relationship"] = "countersigned"
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
    assert "receipt verifier is not trusted" in " ".join(decision.errors)


def test_recipient_can_pin_receipt_trust_profile_digest(tmp_path: Path) -> None:
    envelope, public, fingerprint = _envelope(tmp_path)

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(
            fingerprint,
            expected_receipt_trust_profile_digest="sha256:" + "8" * 64,
        ),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT,
    )

    assert decision.envelope_authenticated is True
    assert decision.receipt_authenticated is True
    assert decision.accepted is False
    assert "trust-profile digest" in " ".join(decision.errors)


def test_recipient_rejects_revoked_receipt_verifier(tmp_path: Path) -> None:
    envelope, public, fingerprint = _envelope(tmp_path)

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint, receipt_status="revoked"),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT,
    )

    assert decision.envelope_authenticated is True
    assert decision.receipt_authenticated is True
    assert decision.accepted is False
    assert "receipt verifier is revoked" in " ".join(decision.errors)


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


def test_fresh_rewrap_cannot_renew_receipt_without_authoritative_evidence_time(
    tmp_path: Path,
) -> None:
    fresh_issue = ISSUED_AT + timedelta(days=365)
    private, public, fingerprint = _key(tmp_path, "envelope-signer")
    envelope = tmp_path / "rewrapped.dsse.json"
    write_acceptance_attestation(
        RECEIPT,
        EVIDENCE,
        envelope,
        signing_key_path=private,
        signer_identity="envelope-signer.example/release-assurance",
        policy_identity="evaluation.example/policies/release-regression-v3",
        issued_at=fresh_issue,
        evaluation_completed_at=ISSUED_AT,
    )
    policy = _policy(
        fingerprint,
        max_age_seconds=3600,
        max_evidence_age_seconds=86400,
    )

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=policy,
        expected_subject_digest=SUBJECT_DIGEST,
        now=fresh_issue + timedelta(minutes=5),
    )

    assert decision.envelope_authenticated is True
    assert decision.receipt_authenticated is True
    assert decision.accepted is False
    assert "authoritative evidence timestamp is unavailable" in " ".join(
        decision.errors
    )


@pytest.mark.parametrize(
    ("receipt_issued_at", "expected_error"),
    [
        (
            ISSUED_AT + timedelta(seconds=1),
            "authoritative evidence timestamp is in the future",
        ),
        (
            ISSUED_AT - timedelta(days=2),
            "technical evidence is stale under recipient policy",
        ),
    ],
)
def test_v013_wrapper_cannot_manufacture_an_evidence_timestamp(
    tmp_path: Path,
    receipt_issued_at: datetime,
    expected_error: str,
) -> None:
    envelope, public, fingerprint = _envelope(tmp_path)
    statement = _payload(envelope)
    statement["predicate"]["timestamps"]["receipt_issued_at"] = (
        receipt_issued_at.isoformat().replace("+00:00", "Z")
    )
    _resign(envelope, tmp_path / "envelope-signer.private.pem", statement)

    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(
            fingerprint,
            max_evidence_age_seconds=86400,
        ),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT,
    )

    assert decision.envelope_authenticated is True
    assert decision.receipt_authenticated is True
    assert decision.accepted is False
    errors = " ".join(decision.errors)
    assert "timestamp is not authenticated by the embedded v0.13 receipt" in errors
    assert expected_error in errors


def test_noncanonical_v013_receipt_bytes_are_preserved_in_wrapper(
    tmp_path: Path,
) -> None:
    receipt = json.loads(RECEIPT.read_bytes())
    noncanonical = json.dumps(receipt, indent=2, sort_keys=False).encode("utf-8")
    receipt_path = tmp_path / "noncanonical.receipt.json"
    receipt_path.write_bytes(noncanonical)
    envelope, public, fingerprint = _envelope(
        tmp_path,
        signer_identity="envelope-signer.example/release-assurance",
        seed=17,
    )
    envelope.unlink()
    private = tmp_path / "envelope-signer.private.pem"
    write_acceptance_attestation(
        receipt_path,
        EVIDENCE,
        envelope,
        signing_key_path=private,
        signer_identity="envelope-signer.example/release-assurance",
        policy_identity="evaluation.example/policies/release-regression-v3",
        issued_at=ISSUED_AT,
    )
    receipt_block = _payload(envelope)["predicate"]["receipt"]

    assert base64.b64decode(receipt_block["raw_base64"], validate=True) == noncanonical
    assert receipt_block["digest"] == (
        "sha256:" + hashlib.sha256(noncanonical).hexdigest()
    )
    decision = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT,
    )
    assert decision.accepted is True


def test_contradictory_receipt_and_predicate_is_rejected_even_when_resigned(
    tmp_path: Path,
) -> None:
    private, public, fingerprint = _key(tmp_path, "envelope-signer")
    envelope = tmp_path / "acceptance.dsse.json"
    write_acceptance_attestation(
        RECEIPT,
        EVIDENCE,
        envelope,
        signing_key_path=private,
        signer_identity="envelope-signer.example/release-assurance",
        policy_identity="evaluation.example/policies/release-regression-v3",
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
    private, public, fingerprint = _key(tmp_path, "envelope-signer")
    envelope = tmp_path / "acceptance.dsse.json"
    write_acceptance_attestation(
        RECEIPT,
        EVIDENCE,
        envelope,
        signing_key_path=private,
        signer_identity="envelope-signer.example/release-assurance",
        policy_identity="evaluation.example/policies/release-regression-v3",
        issued_at=ISSUED_AT,
    )
    statement = _payload(envelope)
    statement["predicate"]["receipt"]["content"]["statement"]["verdict"]["ok"] = False
    receipt = statement["predicate"]["receipt"]
    _replace_embedded_receipt(receipt, receipt["content"])
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
