from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_support import EvidencePackResult, EvidencePackStatus
from invarlock.evidence_receipt import (
    SIGNED_RECEIPT_FORMAT_V1,
    SIGNED_RECEIPT_FORMAT_V2,
    EvidenceReceiptError,
    verify_signed_verification_receipt,
    write_signed_verification_receipt,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _artifact_digests() -> dict[str, str]:
    return {"baseline": _digest("d"), "subject": _digest("e")}


def _input_anchor_kwargs() -> dict[str, object]:
    return {
        "expected_artifact_digests": _artifact_digests(),
        "expected_schedule_digest": _digest("f"),
    }


def _key(tmp_path: Path, name: str) -> tuple[Path, str]:
    key = ed25519.Ed25519PrivateKey.generate()
    path = tmp_path / f"{name}.pem"
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return path, public_key_fingerprint(key.public_key())


def _inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, str], str]:
    pack = tmp_path / "evidence"
    pack.mkdir()
    (pack / "manifest.json").write_text(
        '{"format":"invarlock/evidence-pack-v1"}\n', encoding="utf-8"
    )
    policy = tmp_path / "policy.json"
    policy.write_text('{"policy":"test"}\n', encoding="utf-8")
    return (
        pack,
        policy,
        {"baseline": _digest("a"), "subject": _digest("b")},
        _digest("c"),
    )


def _result(
    pack: Path,
    policy: Path,
    runtimes: dict[str, str],
    pack_signer: str,
    request_digest: str | None = None,
) -> EvidencePackResult:
    anchors: dict[str, object] = {
        "policy_digest": "sha256:" + hashlib.sha256(policy.read_bytes()).hexdigest(),
        "artifact_digests": _artifact_digests(),
        "schedule_digest": _digest("f"),
        "runtime_digests": runtimes,
        "signer_fingerprint": pack_signer,
    }
    if request_digest is not None:
        anchors["request_digest"] = request_digest
    return EvidencePackResult(
        payload={
            "ok": True,
            "integrity_ok": True,
            "policy_verdict": "pass",
            "anchors": anchors,
        },
        status=EvidencePackStatus.OK,
        manifest_digest="sha256:"
        + hashlib.sha256(pack.joinpath("manifest.json").read_bytes()).hexdigest(),
    )


def _write(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str], str, str]:
    pack, policy, runtimes, pack_signer = _inputs(tmp_path)
    key, verifier_fingerprint = _key(tmp_path, "verifier")
    receipt = tmp_path / "verification.receipt.json"
    returned = write_signed_verification_receipt(
        pack,
        _result(pack, policy, runtimes, pack_signer),
        receipt,
        policy_path=policy,
        **_input_anchor_kwargs(),
        expected_runtime_digests=runtimes,
        expected_pack_signer_fingerprint=pack_signer,
        verifier_identity="invarlock-verifier/release",
        verifier_signing_key_path=key,
    )
    assert returned == verifier_fingerprint
    return receipt, pack, policy, runtimes, pack_signer, verifier_fingerprint


def test_signed_receipt_verifies_against_independent_roots(tmp_path: Path) -> None:
    receipt, pack, policy, runtimes, pack_signer, verifier = _write(tmp_path)

    result = verify_signed_verification_receipt(
        receipt,
        pack,
        policy_path=policy,
        **_input_anchor_kwargs(),
        expected_runtime_digests=runtimes,
        expected_pack_signer_fingerprint=pack_signer,
        expected_verifier_identity="invarlock-verifier/release",
        expected_verifier_fingerprint=verifier,
    )

    assert result.ok is True
    assert result.signed is True
    assert result.errors == ()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["statement"]["format"] == SIGNED_RECEIPT_FORMAT_V1


def test_request_anchor_uses_v2_and_is_required_for_gguf_receipts(
    tmp_path: Path,
) -> None:
    pack, policy, runtimes, pack_signer = _inputs(tmp_path)
    request = {
        "comparison": {
            "baseline": {"runtime": {"provider": "llama_cpp"}},
            "subject": {"runtime": {"provider": "llama_cpp"}},
        }
    }
    request_bytes = (
        json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    request_digest = "sha256:" + hashlib.sha256(request_bytes).hexdigest()
    (pack / "request.json").write_bytes(request_bytes)
    manifest = {
        "format": "invarlock/evidence-pack-v1",
        "evidence": {"request": {"path": "request.json", "digest": request_digest}},
    }
    (pack / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    key, verifier = _key(tmp_path, "v2-verifier")
    receipt = tmp_path / "v2.receipt.json"
    write_signed_verification_receipt(
        pack,
        _result(pack, policy, runtimes, pack_signer, request_digest),
        receipt,
        policy_path=policy,
        **_input_anchor_kwargs(),
        expected_runtime_digests=runtimes,
        expected_pack_signer_fingerprint=pack_signer,
        expected_request_digest=request_digest,
        verifier_identity="invarlock-verifier/release",
        verifier_signing_key_path=key,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["statement"]["format"] == SIGNED_RECEIPT_FORMAT_V2
    assert payload["statement"]["anchors"]["request_digest"] == request_digest

    verified = verify_signed_verification_receipt(
        receipt,
        pack,
        policy_path=policy,
        **_input_anchor_kwargs(),
        expected_runtime_digests=runtimes,
        expected_pack_signer_fingerprint=pack_signer,
        expected_request_digest=request_digest,
        expected_verifier_identity="invarlock-verifier/release",
        expected_verifier_fingerprint=verifier,
    )
    assert verified.ok is True

    rejected = verify_signed_verification_receipt(
        receipt,
        pack,
        policy_path=policy,
        **_input_anchor_kwargs(),
        expected_runtime_digests=runtimes,
        expected_pack_signer_fingerprint=pack_signer,
        expected_verifier_identity="invarlock-verifier/release",
        expected_verifier_fingerprint=verifier,
    )
    assert rejected.ok is False
    assert "independent request anchor" in " ".join(rejected.errors)


def test_statement_tampering_invalidates_signature(tmp_path: Path) -> None:
    receipt, pack, policy, runtimes, pack_signer, verifier = _write(tmp_path)
    receipt.chmod(0o644)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["statement"]["verdict"]["ok"] = False
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_signed_verification_receipt(
        receipt,
        pack,
        policy_path=policy,
        **_input_anchor_kwargs(),
        expected_runtime_digests=runtimes,
        expected_pack_signer_fingerprint=pack_signer,
        expected_verifier_identity="invarlock-verifier/release",
        expected_verifier_fingerprint=verifier,
    )

    assert result.ok is False
    assert "signature verification failed" in " ".join(result.errors)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("identity", "verifier identity"),
        ("fingerprint", "verifier key"),
        ("artifact", "receipt anchors"),
        ("schedule", "receipt anchors"),
        ("runtime", "receipt anchors"),
        ("pack_signer", "receipt anchors"),
    ],
)
def test_caller_roots_override_embedded_receipt_claims(
    tmp_path: Path, field: str, message: str
) -> None:
    receipt, pack, policy, runtimes, pack_signer, verifier = _write(tmp_path)
    expected_identity = "invarlock-verifier/release"
    input_anchors = _input_anchor_kwargs()
    if field == "identity":
        expected_identity = "invarlock-verifier/other"
    if field == "fingerprint":
        _other_key, verifier = _key(tmp_path, "other-verifier")
    if field == "runtime":
        runtimes = {**runtimes, "subject": _digest("d")}
    if field == "artifact":
        input_anchors["expected_artifact_digests"] = {
            **_artifact_digests(),
            "subject": _digest("0"),
        }
    if field == "schedule":
        input_anchors["expected_schedule_digest"] = _digest("0")
    if field == "pack_signer":
        pack_signer = _digest("e")

    result = verify_signed_verification_receipt(
        receipt,
        pack,
        policy_path=policy,
        **input_anchors,
        expected_runtime_digests=runtimes,
        expected_pack_signer_fingerprint=pack_signer,
        expected_verifier_identity=expected_identity,
        expected_verifier_fingerprint=verifier,
    )

    assert result.ok is False
    assert message in " ".join(result.errors)


def test_unsigned_diagnostic_json_is_rejected_in_strict_mode(tmp_path: Path) -> None:
    pack, policy, runtimes, pack_signer = _inputs(tmp_path)
    _key_path, verifier = _key(tmp_path, "verifier")
    receipt = tmp_path / "unsigned.json"
    receipt.write_text('{"ok":true}\n', encoding="utf-8")

    result = verify_signed_verification_receipt(
        receipt,
        pack,
        policy_path=policy,
        **_input_anchor_kwargs(),
        expected_runtime_digests=runtimes,
        expected_pack_signer_fingerprint=pack_signer,
        expected_verifier_identity="invarlock-verifier/release",
        expected_verifier_fingerprint=verifier,
    )

    assert result.ok is False
    assert result.signed is False
    assert result.errors == ("signed verification receipt is required",)


def test_writer_rejects_result_anchor_drift(tmp_path: Path) -> None:
    pack, policy, runtimes, pack_signer = _inputs(tmp_path)
    key, _verifier = _key(tmp_path, "verifier")
    drifted = {**runtimes, "subject": _digest("f")}

    with pytest.raises(EvidenceReceiptError, match="runtime anchors"):
        write_signed_verification_receipt(
            pack,
            _result(pack, policy, drifted, pack_signer),
            tmp_path / "receipt.json",
            policy_path=policy,
            **_input_anchor_kwargs(),
            expected_runtime_digests=runtimes,
            expected_pack_signer_fingerprint=pack_signer,
            verifier_identity="invarlock-verifier/release",
            verifier_signing_key_path=key,
        )


def test_receipt_is_external_and_no_clobber(tmp_path: Path) -> None:
    pack, policy, runtimes, pack_signer = _inputs(tmp_path)
    key, _verifier = _key(tmp_path, "verifier")
    result = _result(pack, policy, runtimes, pack_signer)

    with pytest.raises(EvidenceReceiptError, match="outside"):
        write_signed_verification_receipt(
            pack,
            result,
            pack / "receipt.json",
            policy_path=policy,
            **_input_anchor_kwargs(),
            expected_runtime_digests=runtimes,
            expected_pack_signer_fingerprint=pack_signer,
            verifier_identity="invarlock-verifier/release",
            verifier_signing_key_path=key,
        )
    receipt = tmp_path / "receipt.json"
    receipt.write_text("owned\n", encoding="utf-8")
    with pytest.raises(EvidenceReceiptError, match="already exists"):
        write_signed_verification_receipt(
            pack,
            result,
            receipt,
            policy_path=policy,
            **_input_anchor_kwargs(),
            expected_runtime_digests=runtimes,
            expected_pack_signer_fingerprint=pack_signer,
            verifier_identity="invarlock-verifier/release",
            verifier_signing_key_path=key,
        )
