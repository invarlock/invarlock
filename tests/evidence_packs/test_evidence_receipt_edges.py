from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from invarlock import evidence_receipt as receipt_module
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


def test_receipt_no_clobber_race_never_removes_the_other_writer_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"
    real_open = Path.open

    def raced_open(path: Path, mode: str = "r", *args, **kwargs):
        if path == receipt and mode == "xb":
            with real_open(receipt, "wb") as handle:
                handle.write(b"other writer")
            raise FileExistsError("raced")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", raced_open)

    with pytest.raises(receipt_module.EvidenceReceiptError, match="already exists"):
        receipt_module._write_no_clobber(receipt, b"ours")

    assert receipt.read_bytes() == b"other writer"


def test_receipt_writer_allows_caller_managed_signer_roles(
    tmp_path: Path,
) -> None:
    pack, policy, runtimes, _pack_signer = _inputs(tmp_path)
    verifier_key, verifier_fingerprint = _key(tmp_path, "same-signer")
    receipt = tmp_path / "receipt.json"
    result = _result(pack, policy, runtimes, verifier_fingerprint)

    written = write_signed_verification_receipt(
        pack,
        result,
        receipt,
        policy_path=policy,
        **_input_anchor_kwargs(),
        expected_runtime_digests=runtimes,
        expected_pack_signer_fingerprint=verifier_fingerprint,
        verifier_identity="invarlock-verifier/release",
        verifier_signing_key_path=verifier_key,
    )

    assert written == verifier_fingerprint
    assert receipt.is_file()


def test_receipt_writer_removes_partial_file_after_durable_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "receipt.json"
    monkeypatch.setattr(
        receipt_module.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("storage failure")),
    )

    with pytest.raises(EvidenceReceiptError, match="could not write"):
        receipt_module._write_no_clobber(destination, b"partial")

    assert not destination.exists()


def test_receipt_writer_maps_destination_open_failure_without_cleanup_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "receipt.json"
    real_open = Path.open

    def denied_open(path: Path, mode: str = "r", *args, **kwargs):
        if path == destination and mode == "xb":
            raise OSError("permission denied")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)
    with pytest.raises(EvidenceReceiptError, match="could not write"):
        receipt_module._write_no_clobber(destination, b"receipt")
    assert not destination.exists()


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
        '{"format":"invarlock/evidence-pack-v1","tampered":true}\n',
        encoding="utf-8",
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


@pytest.mark.parametrize(
    ("captured", "message"),
    [
        ({"policy_bytes": "{}"}, "policy bytes must be exact bytes"),
        (
            {"policy_bytes": b"x" * (4 * 1024 * 1024 + 1)},
            "policy anchor exceeds",
        ),
        (
            {"verifier_signing_key_bytes": "private-key"},
            "signing key bytes must be exact bytes",
        ),
        (
            {"verifier_signing_key_bytes": b"x" * (64 * 1024 + 1)},
            "signing key exceeds",
        ),
        ({"verifier_signing_key_bytes": b"not-a-key"}, "could not load"),
    ],
)
def test_receipt_writer_rejects_unsafe_captured_trust_material(
    tmp_path: Path, captured: dict[str, object], message: str
) -> None:
    pack, policy, runtimes, pack_signer = _inputs(tmp_path)
    key, _verifier = _key(tmp_path, "verifier")
    kwargs: dict[str, object] = {
        "policy_path": policy,
        **_input_anchor_kwargs(),
        "expected_runtime_digests": runtimes,
        "expected_pack_signer_fingerprint": pack_signer,
        "verifier_identity": "invarlock-verifier/release",
        "verifier_signing_key_path": key,
        **captured,
    }

    with pytest.raises(EvidenceReceiptError, match=message):
        write_signed_verification_receipt(
            pack,
            _result(pack, policy, runtimes, pack_signer),
            tmp_path / "receipt.json",
            **kwargs,  # type: ignore[arg-type]
        )


def test_receipt_writer_rejects_captured_non_ed25519_private_key(
    tmp_path: Path,
) -> None:
    pack, policy, runtimes, pack_signer = _inputs(tmp_path)
    key, _verifier = _key(tmp_path, "verifier")
    private_key = ec.generate_private_key(ec.SECP256R1())
    captured_key = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
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
            verifier_signing_key_path=key,
            verifier_signing_key_bytes=captured_key,
        )


@pytest.mark.parametrize(
    ("result_request", "expected_request", "message"),
    [
        (None, _digest("1"), "does not match caller request"),
        (_digest("1"), None, "unexpected request anchor"),
    ],
)
def test_receipt_writer_rejects_request_anchor_disagreement_with_result(
    tmp_path: Path,
    result_request: str | None,
    expected_request: str | None,
    message: str,
) -> None:
    pack, policy, runtimes, pack_signer = _inputs(tmp_path)
    key, _verifier = _key(tmp_path, "verifier")

    with pytest.raises(EvidenceReceiptError, match=message):
        write_signed_verification_receipt(
            pack,
            _result(pack, policy, runtimes, pack_signer, result_request),
            tmp_path / "receipt.json",
            policy_path=policy,
            **_input_anchor_kwargs(),
            expected_runtime_digests=runtimes,
            expected_pack_signer_fingerprint=pack_signer,
            expected_request_digest=expected_request,
            verifier_identity="invarlock-verifier/release",
            verifier_signing_key_path=key,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("policy_digest", _digest("0"), "policy anchor"),
        (
            "artifact_digests",
            {"baseline": _digest("0"), "subject": _digest("e")},
            "artifact anchors",
        ),
        ("schedule_digest", _digest("0"), "schedule anchor"),
        ("signer_fingerprint", _digest("0"), "signer anchor"),
        (
            "runtime_digests",
            {"baseline": _digest("0"), "subject": _digest("b")},
            "runtime anchors",
        ),
    ],
)
def test_receipt_writer_rejects_result_anchor_drift(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    pack, policy, runtimes, pack_signer = _inputs(tmp_path)
    key, _verifier = _key(tmp_path, "verifier")
    original = _result(pack, policy, runtimes, pack_signer)
    anchors = dict(original.payload["anchors"])
    anchors[field] = replacement
    drifted = EvidencePackResult(
        payload={**original.payload, "anchors": anchors},
        status=original.status,
        manifest_digest=original.manifest_digest,
    )

    with pytest.raises(EvidenceReceiptError, match=message):
        write_signed_verification_receipt(
            pack,
            drifted,
            tmp_path / "receipt.json",
            policy_path=policy,
            **_input_anchor_kwargs(),
            expected_runtime_digests=runtimes,
            expected_pack_signer_fingerprint=pack_signer,
            verifier_identity="invarlock-verifier/release",
            verifier_signing_key_path=key,
        )


def test_v1_receipt_rejects_unexpected_independent_request_anchor(
    tmp_path: Path,
) -> None:
    receipt, pack, policy, runtimes, pack_signer, verifier = _write(tmp_path)

    result = _verify(
        receipt,
        pack,
        policy,
        runtimes,
        pack_signer,
        verifier,
        expected_request_digest=_digest("1"),
    )

    assert result.ok is False
    assert "request anchors require signed receipt format v2" in result.errors
    assert "request anchor does not match the supplied pack request" in result.errors


@pytest.mark.parametrize(
    ("reference_path", "reference_digest", "request_payload", "message"),
    [
        ("other.json", _digest("0"), None, "reference path is invalid"),
        ("request.json", _digest("0"), {}, "digest does not match manifest"),
        ("request.json", None, {"not_comparison": {}}, "comparison is invalid"),
    ],
)
def test_receipt_verifier_rejects_malformed_pack_request_binding(
    tmp_path: Path,
    reference_path: str,
    reference_digest: str | None,
    request_payload: dict[str, object] | None,
    message: str,
) -> None:
    receipt, pack, policy, runtimes, pack_signer, verifier = _write(tmp_path)
    request_bytes = b""
    if request_payload is not None:
        request_bytes = (
            json.dumps(request_payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        (pack / "request.json").write_bytes(request_bytes)
    digest = reference_digest or ("sha256:" + hashlib.sha256(request_bytes).hexdigest())
    (pack / "manifest.json").write_text(
        json.dumps(
            {
                "format": "invarlock/evidence-pack-v1",
                "evidence": {"request": {"path": reference_path, "digest": digest}},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    result = _verify(receipt, pack, policy, runtimes, pack_signer, verifier)

    assert result.ok is False
    assert message in " ".join(result.errors)
