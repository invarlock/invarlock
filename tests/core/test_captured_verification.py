"""Focused captured-pack verification and receipt coverage."""

import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.captured_verification import (
    CapturedVerificationError,
    CapturedVerificationIncomplete,
    verify_captured_evidence,
)
from invarlock.evaluation_record_contracts.contracts import digest
from invarlock.evaluation_records.templates import example_project
from invarlock.evidence_pack_integrity import public_key_fingerprint
from tests.core.test_captured_evaluation import _key, _request


def _private_key(tmp_path, name="verifier.pem"):
    path = tmp_path / name
    key = Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return path, public_key_fingerprint(key.public_key())


def _anchors(request):
    manifest = json.loads((request.evidence / "manifest.json").read_text())
    normalized = json.loads((request.evidence / "request.json").read_text())
    baseline = json.loads((request.evidence / "records/baseline.json").read_text())
    subject = json.loads((request.evidence / "records/subject.json").read_text())
    return {
        "baseline": digest(baseline),
        "subject": digest(subject),
        "request": digest(normalized),
        "signer": manifest["signing_key_fingerprint"],
    }


def test_signed_captured_pass_issues_scoped_receipt(tmp_path):
    baseline, subject, policy = example_project("classification")
    root = tmp_path / "pass"
    root.mkdir()
    request = _request(root, baseline, subject, policy)
    evaluate_key = _key(root)
    from invarlock.captured_evaluation import evaluate_captured_request

    evaluate_captured_request(request, signing_key_path=evaluate_key)
    anchors = _anchors(request)
    verifier, _ = _private_key(root)
    result = verify_captured_evidence(
        request.evidence,
        policy_path=root / "policy.json",
        expected_baseline_run=anchors["baseline"],
        expected_subject_run=anchors["subject"],
        expected_request_digest=anchors["request"],
        expected_signer=anchors["signer"],
        receipt_path=root / "verification.receipt.json",
        verifier_signing_key_path=verifier,
        verifier_identity="test-verifier",
    )
    assert result["ok"] is True
    assert result["kind"] == "captured"


def test_unsigned_captured_evidence_is_rejected(tmp_path):
    baseline, subject, policy = example_project("classification")
    root = tmp_path / "unsigned"
    root.mkdir()
    request = _request(root, baseline, subject, policy)
    from invarlock.captured_evaluation import evaluate_captured_request

    evaluate_captured_request(request, unsigned=True)
    verifier, _ = _private_key(root)
    with pytest.raises(CapturedVerificationError, match="requires signed"):
        verify_captured_evidence(
            request.evidence,
            policy_path=root / "policy.json",
            expected_baseline_run="sha256:" + "0" * 64,
            expected_subject_run="sha256:" + "0" * 64,
            expected_request_digest="sha256:" + "0" * 64,
            expected_signer="sha256:" + "0" * 64,
            receipt_path=root / "receipt.json",
            verifier_signing_key_path=verifier,
            verifier_identity="test-verifier",
        )


def test_captured_budget_refusal_has_no_receipt(tmp_path):
    baseline, subject, policy = example_project("classification")
    root = tmp_path / "budget"
    root.mkdir()
    request = _request(root, baseline, subject, policy)
    evaluate_key = _key(root)
    from invarlock.captured_evaluation import evaluate_captured_request

    evaluate_captured_request(request, signing_key_path=evaluate_key)
    anchors = _anchors(request)
    verifier, _ = _private_key(root)
    with pytest.raises(
        CapturedVerificationIncomplete, match="local_work_budget_exceeded"
    ):
        verify_captured_evidence(
            request.evidence,
            policy_path=root / "policy.json",
            expected_baseline_run=anchors["baseline"],
            expected_subject_run=anchors["subject"],
            expected_request_digest=anchors["request"],
            expected_signer=anchors["signer"],
            receipt_path=root / "budget.receipt.json",
            verifier_signing_key_path=verifier,
            verifier_identity="test-verifier",
            max_bootstrap_draws=0,
        )
    assert not (root / "budget.receipt.json").exists()
