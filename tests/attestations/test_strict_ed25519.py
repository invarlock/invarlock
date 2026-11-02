"""Reject degenerate signing keys even when a recipient explicitly pins one."""

import base64
import hashlib
import json
from datetime import timedelta

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from invarlock import engine
from invarlock.acceptance_attestation import verify_acceptance_attestation
from invarlock.captured_contracts import CapturedIntegrityError, authenticate_manifest
from invarlock.evidence_pack_integrity import verify_signature
from invarlock.judge_measurements.acceptance import (
    verify_judge_evidence_with_policy,
    verify_signed_judge_verification_receipt,
    write_signed_judge_verification_receipt,
)
from invarlock.judge_measurements.evidence import object_sha256
from tests.attestations.test_acceptance_attestation import (
    ISSUED_AT,
    SUBJECT_DIGEST,
    _canonical,
    _envelope,
    _payload,
    _policy,
    _replace_embedded_receipt,
    _resign,
)
from tests.evidence_packs.test_captured_transaction_receipts import (
    captured_receipt as captured_receipt,
)
from tests.evidence_packs.test_evidence_pack_integrity_edges import _signed_pack
from tests.evidence_packs.test_evidence_receipt import _write as _native_receipt
from tests.evidence_packs.test_evidence_receipt_edges import _verify
from tests.examples.test_policy_engine_interop import _verifier_module
from tests.judge_measurements.test_evidence_acceptance import (
    _publish,
    _verifier_key_path,
)


def weak_envelope(root):
    envelope, public, _ = _envelope(root)
    raw = b"\x01" + b"\0" * 31
    key = Ed25519PublicKey.from_public_bytes(raw)
    fingerprint = "sha256:" + hashlib.sha256(raw).hexdigest()
    public.write_bytes(
        key.public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    outer = json.loads(envelope.read_bytes())
    statement = _payload(envelope)
    statement["predicate"]["signers"]["envelope"]["fingerprint"] = fingerprint
    outer["payload"] = base64.b64encode(_canonical(statement)).decode()
    outer["signatures"] = [
        {"keyid": fingerprint, "sig": base64.b64encode(raw + b"\0" * 32).decode()}
    ]
    counterexample = root / "counterexample.dsse.json"
    counterexample.write_bytes(_canonical(outer))
    options = {
        "trusted_public_keys": {fingerprint: public},
        "recipient_policy": _policy(fingerprint),
        "expected_subject_digest": SUBJECT_DIGEST,
        "now": ISSUED_AT + timedelta(minutes=5),
    }
    return counterexample, options


def test_pinned_small_order_key_cannot_authenticate_an_attestation(tmp_path):
    path, options = weak_envelope(tmp_path)
    result = verify_acceptance_attestation(path, **options)
    assert not result.envelope_authenticated
    assert not result.accepted
    assert result.errors == ("DSSE signature verification failed",)


IDENTITY = bytes.fromhex("01" + "00" * 31)
FORGED_SIGNATURE = base64.b64encode(IDENTITY + bytes(32)).decode()
WEAK_FINGERPRINT = "sha256:" + hashlib.sha256(IDENTITY).hexdigest()
WEAK_PEM = (
    Ed25519PublicKey.from_public_bytes(IDENTITY)
    .public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    .decode()
)


def _write_json(path, value):
    path.chmod(0o644)
    path.write_bytes(_canonical(value))


def _forge_receipt(receipt):
    receipt["statement"]["verifier"]["signing_key_fingerprint"] = WEAK_FINGERPRINT
    receipt["signature"]["public_key"]["value"] = WEAK_PEM
    receipt["signature"]["value"] = FORGED_SIGNATURE


def test_pinned_small_order_embedded_receipt_is_rejected(tmp_path):
    envelope, public, fingerprint = _envelope(tmp_path)
    statement = _payload(envelope)
    block = statement["predicate"]["receipt"]
    receipt = block["content"]
    _forge_receipt(receipt)
    _replace_embedded_receipt(block, receipt)
    statement["predicate"]["signers"]["receipt"]["fingerprint"] = WEAK_FINGERPRINT
    _resign(envelope, tmp_path / "envelope-signer.private.pem", statement)
    result = verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint, receipt_fingerprint=WEAK_FINGERPRINT),
        expected_subject_digest=SUBJECT_DIGEST,
        now=ISSUED_AT + timedelta(minutes=5),
    )
    assert result.envelope_authenticated
    assert not result.receipt_authenticated
    assert not result.accepted
    assert "receipt signature verification failed" in result.errors


@pytest.mark.parametrize("family", ["native", "captured"])
def test_small_order_manifest_is_rejected(tmp_path, family):
    root, _ = _signed_pack(tmp_path)
    manifest = {"signing_key_fingerprint": WEAK_FINGERPRINT}
    raw = _canonical(manifest)
    signature = json.loads((root / "manifest.signature.json").read_bytes())
    signature["signing_key_fingerprint"] = WEAK_FINGERPRINT
    signature["public_key"]["value"] = WEAK_PEM
    signature["signature"]["value"] = FORGED_SIGNATURE
    (root / "manifest.json").write_bytes(raw)
    (root / "manifest.signature.json").write_bytes(_canonical(signature))
    if family == "captured":
        with pytest.raises(
            CapturedIntegrityError, match="manifest signature is invalid"
        ):
            authenticate_manifest(manifest, raw, _canonical(signature))
    else:
        errors, _, _ = verify_signature(
            root, strict=True, expected_fingerprints={WEAK_FINGERPRINT}
        )
        assert errors == ["manifest signature verification failed"]


def test_pinned_small_order_native_receipt_is_rejected(tmp_path):
    receipt, pack, policy, runtimes, pack_signer, _ = _native_receipt(tmp_path)
    payload = json.loads(receipt.read_bytes())
    _forge_receipt(payload)
    _write_json(receipt, payload)
    result = _verify(receipt, pack, policy, runtimes, pack_signer, WEAK_FINGERPRINT)
    assert not result.ok
    assert "signed receipt signature verification failed" in result.errors


def test_pinned_small_order_captured_receipt_is_rejected(captured_receipt):
    receipt, pack, kwargs = captured_receipt
    payload = json.loads(receipt.read_bytes())
    _forge_receipt(payload)
    _write_json(receipt, payload)
    result = engine.verify_signed_verification_receipt(
        receipt, pack, **{**kwargs, "expected_verifier_fingerprint": WEAK_FINGERPRINT}
    )
    assert not result.ok
    assert any("signature" in error for error in result.errors)


def test_pinned_small_order_judge_envelope_is_rejected(tmp_path):
    publication, policy_path = _publish(tmp_path)
    path = publication.path / "envelope.json"
    envelope = json.loads(path.read_bytes())
    envelope["signer"]["public_key"] = base64.b64encode(IDENTITY).decode()
    envelope["signer"]["public_key_sha256"] = WEAK_FINGERPRINT
    envelope["signature"] = FORGED_SIGNATURE
    _write_json(path, envelope)
    policy = json.loads(policy_path.read_bytes())
    policy["trusted_signer"]["public_key_sha256"] = WEAK_FINGERPRINT
    _write_json(policy_path, policy)
    result = verify_judge_evidence_with_policy(publication.path, policy_path)
    assert not result.authenticated
    assert not result.accepted
    assert any("signature" in error for error in result.errors)


def test_pinned_small_order_judge_receipt_is_rejected(tmp_path):
    publication, policy_path = _publish(tmp_path)
    local = verify_judge_evidence_with_policy(publication.path, policy_path)
    receipt_path = tmp_path / "signed-receipt.json"
    write_signed_judge_verification_receipt(
        publication.path,
        local,
        receipt_path,
        recipient_policy_path=policy_path,
        verifier_identity="example-verifier",
        verifier_signing_key_path=_verifier_key_path(tmp_path),
    )
    receipt = json.loads(receipt_path.read_bytes())
    _forge_receipt(receipt)
    _write_json(receipt_path, receipt)
    result = verify_signed_judge_verification_receipt(
        receipt_path,
        expected_verifier_identity="example-verifier",
        expected_verifier_fingerprint=WEAK_FINGERPRINT,
        expected_recipient_policy_sha256=object_sha256(
            json.loads(policy_path.read_bytes())
        ),
    )
    assert not result.ok
    assert any("signature" in error for error in result.errors)


def test_standalone_policy_recipient_rejects_small_order_envelope(tmp_path):
    path, options = weak_envelope(tmp_path)
    with pytest.raises(InvalidSignature):
        _verifier_module().verify_envelope(
            envelope_path=path,
            envelope_key_path=next(iter(options["trusted_public_keys"].values())),
        )


def test_standalone_policy_recipient_rejects_small_order_receipt(tmp_path):
    envelope, _, _ = _envelope(tmp_path)
    statement = _payload(envelope)
    receipt = statement["predicate"]["receipt"]["content"]
    _forge_receipt(receipt)
    _replace_embedded_receipt(statement["predicate"]["receipt"], receipt)
    statement["predicate"]["signers"]["receipt"]["fingerprint"] = WEAK_FINGERPRINT
    with pytest.raises(InvalidSignature):
        _verifier_module().verify_receipt(statement)
