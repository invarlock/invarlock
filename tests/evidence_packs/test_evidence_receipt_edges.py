from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from invarlock.evidence_pack_support import EvidencePackResult, EvidencePackStatus
from invarlock.evidence_receipt import (
    EvidenceReceiptError,
    verify_signed_verification_receipt,
    write_signed_verification_receipt,
)
from tests.evidence_packs.test_evidence_receipt import (
    _digest,
    _input_anchor_kwargs,
    _inputs,
    _key,
    _result,
    _write,
)


def _verify(
    receipt: Path,
    pack: Path,
    policy: Path,
    runtimes: dict[str, str],
    pack_signer: str,
    verifier: str,
    **overrides: object,
):
    kwargs: dict[str, object] = {
        "policy_path": policy,
        **_input_anchor_kwargs(),
        "expected_runtime_digests": runtimes,
        "expected_pack_signer_fingerprint": pack_signer,
        "expected_verifier_identity": "invarlock-verifier/release",
        "expected_verifier_fingerprint": verifier,
    }
    kwargs.update(overrides)
    return verify_signed_verification_receipt(
        receipt,
        pack,
        **kwargs,  # type: ignore[arg-type]
    )


def _mutate_receipt(receipt: Path, mutate: object) -> None:
    receipt.chmod(0o644)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(payload)
    receipt.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["statement"].update(extra=True), "statement fields"),
        (lambda value: value["statement"].update(format="other"), "receipt format"),
        (
            lambda value: value["statement"].update(pack_manifest_digest="bad"),
            "manifest digest is invalid",
        ),
        (lambda value: value["signature"].update(extra=True), "signature fields"),
        (lambda value: value["signature"].update(format="other"), "signature format"),
        (lambda value: value["signature"].update(algorithm="rsa"), "algorithm"),
        (lambda value: value["statement"].update(verifier=[]), "verifier fields"),
        (lambda value: value["statement"].update(verdict=[]), "verdict fields"),
        (
            lambda value: value["statement"]["verdict"].update(ok="yes"),
            "verdict booleans",
        ),
        (
            lambda value: value["statement"]["verdict"].update(
                policy_verdict="unknown"
            ),
            "policy verdict",
        ),
        (
            lambda value: value["statement"]["verdict"].update(
                verification_status=True
            ),
            "verification status",
        ),
        (
            lambda value: value["statement"]["verdict"].update(
                ok=True, integrity_ok=False
            ),
            "successful verdict is inconsistent",
        ),
        (
            lambda value: value["signature"].update(public_key=[]),
            "public key is invalid",
        ),
        (
            lambda value: value["signature"].update(value=None),
            "signature verification failed",
        ),
    ],
)
def test_receipt_verifier_rejects_every_open_or_inconsistent_signed_field(
    tmp_path: Path, mutate: object, message: str
) -> None:
    receipt, pack, policy, runtimes, pack_signer, verifier = _write(tmp_path)
    _mutate_receipt(receipt, mutate)

    result = _verify(receipt, pack, policy, runtimes, pack_signer, verifier)

    assert result.ok is False
    assert result.signed is True
    assert message in " ".join(result.errors)


def test_receipt_verifier_rejects_invalid_external_verifier_roots(
    tmp_path: Path,
) -> None:
    receipt, pack, policy, runtimes, pack_signer, verifier = _write(tmp_path)

    bad_fingerprint = _verify(
        receipt,
        pack,
        policy,
        runtimes,
        pack_signer,
        verifier,
        expected_verifier_fingerprint="invalid",
    )
    bad_identity = _verify(
        receipt,
        pack,
        policy,
        runtimes,
        pack_signer,
        verifier,
        expected_verifier_identity="identity with spaces",
    )

    assert "expected verifier fingerprint is invalid" in bad_fingerprint.errors
    assert "expected verifier identity is invalid" in " ".join(bad_identity.errors)


def test_receipt_binds_current_pack_and_policy_bytes(tmp_path: Path) -> None:
    receipt, pack, policy, runtimes, pack_signer, verifier = _write(tmp_path)
    (pack / "manifest.json").write_text(
        '{"format":"evidence-pack-v1","tampered":true}\n', encoding="utf-8"
    )
    policy.write_text('{"policy":"changed"}\n', encoding="utf-8")

    result = _verify(receipt, pack, policy, runtimes, pack_signer, verifier)

    joined = " ".join(result.errors)
    assert "does not bind the supplied pack manifest" in joined
    assert "anchors do not match" in joined


def test_receipt_inside_pack_and_unsigned_diagnostic_are_never_trusted(
    tmp_path: Path,
) -> None:
    receipt, pack, policy, runtimes, pack_signer, verifier = _write(tmp_path)
    inside = pack / "verification.receipt.json"
    inside.write_bytes(receipt.read_bytes())

    result = _verify(inside, pack, policy, runtimes, pack_signer, verifier)

    assert result.ok is False
    assert "inside the evidence pack" in " ".join(result.errors)

    unsigned = tmp_path / "diagnostic.json"
    unsigned.write_text('{"ok":true}\n', encoding="utf-8")
    diagnostic = _verify(
        unsigned,
        pack,
        policy,
        runtimes,
        pack_signer,
        verifier,
        require_signed=False,
    )
    assert diagnostic.ok is False
    assert diagnostic.signed is False
    assert diagnostic.errors == ()


def test_receipt_loader_rejects_missing_nonobject_and_duplicate_json(
    tmp_path: Path,
) -> None:
    pack, policy, runtimes, pack_signer = _inputs(tmp_path)
    _key_path, verifier = _key(tmp_path, "verifier")

    missing = _verify(
        tmp_path / "missing.json",
        pack,
        policy,
        runtimes,
        pack_signer,
        verifier,
    )
    assert "unavailable" in " ".join(missing.errors)

    for payload, message in (("[]", "JSON object"), ('{"x":1,"x":2}', "duplicate")):
        path = tmp_path / f"{len(payload)}.json"
        path.write_text(payload, encoding="utf-8")
        result = _verify(path, pack, policy, runtimes, pack_signer, verifier)
        assert message in " ".join(result.errors)


def test_receipt_writer_rejects_invalid_key_identity_anchors_and_verdict(
    tmp_path: Path,
) -> None:
    pack, policy, runtimes, pack_signer = _inputs(tmp_path)
    key, _verifier = _key(tmp_path, "verifier")
    base = _result(pack, policy, runtimes, pack_signer)

    invalid_cases = [
        (
            {"expected_artifact_digests": {"baseline": _digest("a")}},
            "exactly",
        ),
        ({"expected_schedule_digest": "invalid"}, "schedule anchor"),
        ({"expected_runtime_digests": {"baseline": _digest("a")}}, "exactly"),
        (
            {
                "expected_runtime_digests": {
                    "baseline": "invalid",
                    "subject": _digest("b"),
                }
            },
            "sha256",
        ),
        ({"expected_pack_signer_fingerprint": "invalid"}, "pack signer"),
        ({"verifier_identity": "identity with spaces"}, "identity is invalid"),
    ]
    for index, (overrides, message) in enumerate(invalid_cases):
        kwargs: dict[str, object] = {
            "policy_path": policy,
            **_input_anchor_kwargs(),
            "expected_runtime_digests": runtimes,
            "expected_pack_signer_fingerprint": pack_signer,
            "verifier_identity": "invarlock-verifier/release",
            "verifier_signing_key_path": key,
        }
        kwargs.update(overrides)
        with pytest.raises(EvidenceReceiptError, match=message):
            write_signed_verification_receipt(
                pack,
                base,
                tmp_path / f"invalid-{index}.json",
                **kwargs,  # type: ignore[arg-type]
            )

    invalid_verdict = EvidencePackResult(
        payload={**base.payload, "policy_verdict": "unknown"},
        status=EvidencePackStatus.OK,
        manifest_digest=base.manifest_digest,
    )
    with pytest.raises(EvidenceReceiptError, match="policy verdict"):
        write_signed_verification_receipt(
            pack,
            invalid_verdict,
            tmp_path / "bad-verdict.json",
            policy_path=policy,
            **_input_anchor_kwargs(),
            expected_runtime_digests=runtimes,
            expected_pack_signer_fingerprint=pack_signer,
            verifier_identity="invarlock-verifier/release",
            verifier_signing_key_path=key,
        )


def test_receipt_writer_rejects_non_ed25519_private_key(tmp_path: Path) -> None:
    pack, policy, runtimes, pack_signer = _inputs(tmp_path)
    private_key = ec.generate_private_key(ec.SECP256R1())
    key_path = tmp_path / "ec-key.pem"
    key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )

    with pytest.raises(EvidenceReceiptError, match="must be Ed25519"):
        write_signed_verification_receipt(
            pack,
            _result(pack, policy, runtimes, pack_signer),
            tmp_path / "receipt.json",
            policy_path=policy,
            **_input_anchor_kwargs(),
            expected_runtime_digests=runtimes,
            expected_pack_signer_fingerprint=pack_signer,
            verifier_identity="invarlock-verifier/release",
            verifier_signing_key_path=key_path,
        )
