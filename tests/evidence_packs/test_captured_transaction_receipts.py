"""Receipt dispatch preserves independent anchors and native wire bytes."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from invarlock import engine
from invarlock.evidence_pack_support import EvidencePackResult, EvidencePackStatus
from invarlock.evidence_receipt import write_signed_verification_receipt
from tests.core.test_captured_sdk_omissions import _bytes, _inputs, _key, _profile
from tests.evidence_packs.test_evidence_receipt import (
    _inputs as _native_inputs,
)
from tests.evidence_packs.test_evidence_receipt import (
    _write as _native_receipt,
)
from tests.evidence_packs.test_evidence_receipt_edges import _verify


@pytest.fixture
def captured_receipt(tmp_path):
    request, baseline, subject, policy = _inputs(tmp_path)
    signer = _key(tmp_path / "signer.pem")
    pack = engine.evaluate_request_file(
        tmp_path / "request.json", signing_key_path=tmp_path / "signer.pem"
    ).evidence_path
    _, verifier = _profile(tmp_path, request, baseline, subject, policy, signer)
    trust = engine.load_trust_inputs(tmp_path / "trust.json")
    assert isinstance(trust, engine.CapturedTrustInputs)
    receipt = tmp_path / "receipt.json"
    verified = engine.verify_evidence(
        pack,
        policy_path=trust.policy_path,
        expected_baseline_run=trust.expected_run_digests["baseline"],
        expected_subject_run=trust.expected_run_digests["subject"],
        expected_signer=signer,
        expected_request_digest=trust.expected_request_digest,
        receipt_path=receipt,
        verifier_signing_key_path=trust.verifier_signing_key_path,
        verifier_identity=trust.verifier_identity,
    )
    assert verified.payload["ok"] is True
    kwargs: dict[str, Any] = {
        "policy_path": trust.policy_path,
        "expected_run_digests": dict(trust.expected_run_digests),
        "expected_request_digest": trust.expected_request_digest,
        "expected_pack_signer_fingerprint": signer,
        "expected_verifier_identity": trust.verifier_identity,
        "expected_verifier_fingerprint": verifier,
    }
    assert engine.verify_signed_verification_receipt(receipt, pack, **kwargs).ok
    return receipt, pack, kwargs


@pytest.mark.parametrize(
    "anchor",
    [
        "expected_artifact_digests",
        "expected_schedule_digest",
        "expected_runtime_digests",
    ],
)
def test_captured_receipt_rejects_explicit_null_native_anchor(captured_receipt, anchor):
    receipt, pack, kwargs = captured_receipt
    before = receipt.read_bytes()
    result = engine.verify_signed_verification_receipt(
        receipt, pack, **{**kwargs, anchor: None}
    )
    assert result == engine.ReceiptVerification(
        False, True, None, None, ("native anchors are not valid for captured receipts",)
    )
    assert receipt.read_bytes() == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_run_digests": None},
        {"expected_run_digests": {}},
        {"expected_run_digests": {"baseline": "sha256:" + "a" * 64}},
        {"expected_request_digest": None},
    ],
)
def test_captured_receipt_requires_complete_independent_run_and_request_anchors(
    captured_receipt, overrides
):
    receipt, pack, kwargs = captured_receipt
    result = engine.verify_signed_verification_receipt(
        receipt, pack, **{**kwargs, **overrides}
    )
    assert result == engine.ReceiptVerification(
        False, True, None, None, ("captured receipt run/request anchors are required",)
    )


@pytest.mark.parametrize(
    "missing",
    [
        "expected_artifact_digests",
        "expected_schedule_digest",
        "expected_runtime_digests",
    ],
)
def test_native_receipt_keeps_required_keyword_error_convention(tmp_path, missing):
    receipt, pack, policy, runtimes, signer, verifier = _native_receipt(tmp_path)
    kwargs: dict[str, Any] = {
        "policy_path": policy,
        "expected_artifact_digests": {
            "baseline": "sha256:" + "d" * 64,
            "subject": "sha256:" + "e" * 64,
        },
        "expected_schedule_digest": "sha256:" + "f" * 64,
        "expected_runtime_digests": runtimes,
        "expected_pack_signer_fingerprint": signer,
        "expected_verifier_identity": "invarlock-verifier/release",
        "expected_verifier_fingerprint": verifier,
    }
    del kwargs[missing]
    with pytest.raises(TypeError, match="native receipt verification requires"):
        engine.verify_signed_verification_receipt(receipt, pack, **kwargs)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"expected_artifact_digests": {}}, "artifact anchors must contain exactly"),
        ({"expected_runtime_digests": {}}, "runtime anchors must contain exactly"),
        (
            {"expected_pack_signer_fingerprint": "invalid"},
            "pack signer anchor is invalid",
        ),
        (
            {"expected_trust_profile_digest": "invalid"},
            "expected trust profile digest must be",
        ),
        (
            {"expected_trust_profile_digest": "sha256:" + "a" * 64},
            "trust profile does not match",
        ),
        ({"expected_run_digests": {}}, "run anchors require a captured receipt"),
    ],
)
def test_native_receipt_rejects_malformed_independent_anchors(
    tmp_path, overrides, message
):
    receipt, pack, policy, runtimes, signer, verifier = _native_receipt(tmp_path)
    result = _verify(receipt, pack, policy, runtimes, signer, verifier, **overrides)
    assert not result.ok
    assert result.signed
    assert message in " ".join(result.errors)


@pytest.mark.parametrize("kind", ["invalid-pem", "ec"])
def test_native_receipt_rejects_wrong_public_key_type(tmp_path, kind):
    receipt, pack, policy, runtimes, signer, verifier = _native_receipt(tmp_path)
    public = (
        "not a PEM key"
        if kind == "invalid-pem"
        else ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        .decode("ascii")
    )
    value = json.loads(receipt.read_bytes())
    value["signature"]["public_key"]["value"] = public
    receipt.chmod(0o600)
    receipt.write_bytes(_bytes(value))
    result = _verify(receipt, pack, policy, runtimes, signer, verifier)
    assert not result.ok
    assert result.signed
    assert result.verifier_fingerprint is None
    assert "signed receipt public key is invalid" in " ".join(result.errors)


@pytest.mark.parametrize("request_bound", [False, True])
def test_native_receipt_bytes_match_independent_v1_v2_encoding(tmp_path, request_bound):
    pack, policy, runtimes, signer = _native_inputs(tmp_path)
    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    key_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public = key.public_key()
    fingerprint = (
        "sha256:"
        + hashlib.sha256(
            public.public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
        ).hexdigest()
    )
    manifest_digest = (
        "sha256:" + hashlib.sha256((pack / "manifest.json").read_bytes()).hexdigest()
    )
    request_digest = "sha256:" + "1" * 64 if request_bound else None
    anchors = {
        "policy_digest": "sha256:" + hashlib.sha256(policy.read_bytes()).hexdigest(),
        "artifact_digests": {
            "baseline": "sha256:" + "d" * 64,
            "subject": "sha256:" + "e" * 64,
        },
        "schedule_digest": "sha256:" + "f" * 64,
        "runtime_digests": runtimes,
        "pack_signer_fingerprint": signer,
        **({"request_digest": request_digest} if request_bound else {}),
    }
    statement = {
        "format": f"invarlock/evidence-verification-receipt-v{2 if request_bound else 1}",
        "pack_manifest_digest": manifest_digest,
        "anchors": anchors,
        "verifier": {
            "identity": "release/verifier",
            "signing_key_fingerprint": fingerprint,
            "trust_profile_digest": None,
        },
        "verdict": {
            "ok": True,
            "integrity_ok": True,
            "policy_verdict": "pass",
            "verification_status": 0,
        },
    }
    expected = _bytes(
        {
            "statement": statement,
            "signature": {
                "format": "invarlock/evidence-verification-receipt-signature-v1",
                "algorithm": "ed25519",
                "public_key": {
                    "encoding": "pem",
                    "value": public.public_bytes(
                        serialization.Encoding.PEM,
                        serialization.PublicFormat.SubjectPublicKeyInfo,
                    ).decode("ascii"),
                },
                "value": base64.b64encode(key.sign(_bytes(statement))).decode("ascii"),
            },
        }
    )
    # The result type permits verdict-only payloads; caller anchors still bind it.
    result = EvidencePackResult(
        payload={"ok": True, "integrity_ok": True, "policy_verdict": "pass"},
        status=EvidencePackStatus.OK,
        manifest_digest=manifest_digest,
    )
    kwargs: dict[str, Any] = {
        "policy_path": policy,
        "expected_artifact_digests": anchors["artifact_digests"],
        "expected_schedule_digest": anchors["schedule_digest"],
        "expected_runtime_digests": runtimes,
        "expected_pack_signer_fingerprint": signer,
        "expected_request_digest": request_digest,
        "verifier_identity": "release/verifier",
        "verifier_signing_key_path": tmp_path / "missing-key.pem",
    }
    missing_receipt = tmp_path / "missing-key.receipt.json"
    with pytest.raises(
        engine.EvidenceReceiptError, match="could not load receipt signing key"
    ):
        write_signed_verification_receipt(pack, result, missing_receipt, **kwargs)
    assert not missing_receipt.exists()
    receipt = tmp_path / "snapshot.receipt.json"
    assert (
        write_signed_verification_receipt(
            pack, result, receipt, **kwargs, verifier_signing_key_bytes=key_bytes
        )
        == fingerprint
    )
    assert receipt.read_bytes() == expected
    key_path = tmp_path / "key.pem"
    key_path.write_bytes(key_bytes)
    kwargs["verifier_signing_key_path"] = key_path
    path_receipt = tmp_path / "path.receipt.json"
    write_signed_verification_receipt(pack, result, path_receipt, **kwargs)
    assert path_receipt.read_bytes() == expected


@pytest.mark.parametrize("failure", ["missing-policy", "inside-policy", "inside-key"])
def test_captured_verification_trust_boundary_errors_keep_captured_json(
    tmp_path, failure
):
    _inputs(tmp_path)
    pack = engine.evaluate_request_file(
        tmp_path / "request.json", signing_key_path=None, unsigned=True
    ).evidence_path
    policy = (
        None
        if failure == "missing-policy"
        else pack / "policy.json"
        if failure == "inside-policy"
        else tmp_path / "policy.json"
    )
    with pytest.raises(engine.EvidenceVerificationError) as caught:
        engine.verify_evidence(
            pack,
            policy_path=policy,
            expected_baseline_run=None,
            expected_subject_run=None,
            expected_request_digest=None,
            expected_signer=None,
            verifier_signing_key_path=pack / "key.pem"
            if failure == "inside-key"
            else None,
        )
    assert caught.value.exit_code == 2
    payload = json.loads(caught.value.as_json())
    assert payload["kind"] == "captured"
    assert payload["integrity_ok"] is None
    assert payload["signed_receipt"] is None
    assert (
        "required" in str(caught.value)
        if failure == "missing-policy"
        else "outside" in str(caught.value)
    )


@pytest.mark.parametrize(
    "controls",
    [
        {"expected_baseline_run": "sha256:" + "a" * 64},
        {"expected_subject_run": "sha256:" + "a" * 64},
        {"max_bootstrap_draws": None},
        {"max_bootstrap_draws": True},
        {"max_bootstrap_draws": 0},
    ],
)
def test_native_verification_rejects_captured_controls(tmp_path, controls):
    kwargs: dict[str, Any] = {
        f"expected_{name}": None
        for name in (
            "baseline_artifact",
            "subject_artifact",
            "schedule",
            "baseline_runtime",
            "subject_runtime",
        )
    }
    with pytest.raises(
        engine.EvidenceVerificationError, match="captured anchors/work controls"
    ):
        engine.verify_evidence(
            tmp_path, policy_path=None, expected_signer=None, **kwargs, **controls
        )


def test_verification_rejects_malformed_manifest_before_anchor_validation(tmp_path):
    (tmp_path / "manifest.json").write_bytes(b"{invalid")
    with pytest.raises(engine.EvidenceVerificationError) as caught:
        engine.verify_evidence(
            tmp_path,
            policy_path=None,
            expected_baseline_run=None,
            expected_subject_run=None,
            expected_request_digest=None,
            expected_signer=None,
        )
    assert caught.value.exit_code == 4


def test_native_verification_wraps_independent_path_resolution_failure(
    tmp_path, monkeypatch
):
    pack, policy, _, _ = _native_inputs(tmp_path)
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == policy:
            raise OSError("mount unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(
        engine.EvidenceVerificationError, match="path could not be resolved safely"
    ):
        engine.verify_evidence(
            pack,
            policy_path=policy,
            expected_baseline_artifact=None,
            expected_subject_artifact=None,
            expected_schedule=None,
            expected_baseline_runtime=None,
            expected_subject_runtime=None,
            expected_signer=None,
        )


@pytest.mark.parametrize(
    "missing",
    [
        "baseline_artifact",
        "subject_artifact",
        "schedule",
        "baseline_runtime",
        "subject_runtime",
    ],
)
def test_native_verification_requires_all_keyword_anchors(tmp_path, missing):
    kwargs: dict[str, Any] = dict.fromkeys(
        (
            f"expected_{name}"
            for name in (
                "baseline_artifact",
                "subject_artifact",
                "schedule",
                "baseline_runtime",
                "subject_runtime",
            )
        ),
        None,
    )
    del kwargs[f"expected_{missing}"]
    with pytest.raises(TypeError, match="native verification requires"):
        engine.verify_evidence(
            tmp_path, policy_path=None, expected_signer=None, **kwargs
        )


def test_captured_verification_requires_key_after_external_policy_check(tmp_path):
    _inputs(tmp_path)
    pack = engine.evaluate_request_file(
        tmp_path / "request.json", signing_key_path=None, unsigned=True
    ).evidence_path
    with pytest.raises(engine.EvidenceVerificationError) as caught:
        engine.verify_evidence(
            pack,
            policy_path=tmp_path / "policy.json",
            expected_baseline_run=None,
            expected_subject_run=None,
            expected_request_digest=None,
            expected_signer=None,
        )
    assert caught.value.payload["kind"] == "captured"
    assert caught.value.payload["ok"] is False
    assert caught.value.payload["signed_receipt"] is None
