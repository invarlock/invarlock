from __future__ import annotations

import base64
import copy
import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

import invarlock.acceptance_attestation as target
from invarlock.acceptance_attestation import AcceptanceAttestationError
from invarlock.evidence_pack_integrity import public_key_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "examples/acceptance-handoff/golden"
EVIDENCE = GOLDEN / "evidence"
ISSUED_AT = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


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


def _private_key(seed: int = 31) -> ed25519.Ed25519PrivateKey:
    return ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes((seed + offset) % 256 for offset in range(32))
    )


def _write_private_key(path: Path, key: ed25519.Ed25519PrivateKey) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def _statement() -> dict[str, Any]:
    envelope = _object(GOLDEN / "acceptance.dsse.json")
    value = json.loads(base64.b64decode(envelope["payload"], validate=True))
    assert isinstance(value, dict)
    return value


def _signed_envelope(
    path: Path,
    statement: dict[str, Any],
    key: ed25519.Ed25519PrivateKey,
) -> tuple[str, ed25519.Ed25519PublicKey]:
    fingerprint = public_key_fingerprint(key.public_key())
    predicate = statement.get("predicate")
    if isinstance(predicate, dict):
        predicate["signers"]["envelope"] = {
            "identity": "recipient-tests/envelope",
            "fingerprint": fingerprint,
        }
        predicate["signers"]["relationship"] = "countersigned"
    payload = _canonical(statement)
    signature = key.sign(target._dsse_pae(target.DSSE_PAYLOAD_TYPE, payload))
    path.write_bytes(
        _canonical(
            {
                "payloadType": target.DSSE_PAYLOAD_TYPE,
                "payload": base64.b64encode(payload).decode("ascii"),
                "signatures": [
                    {
                        "keyid": fingerprint,
                        "sig": base64.b64encode(signature).decode("ascii"),
                    }
                ],
            }
        )
    )
    return fingerprint, key.public_key()


def _policy(
    fingerprint: str,
    *,
    envelope_identity: str = "recipient-tests/envelope",
) -> dict[str, Any]:
    return {
        "format": target.RECIPIENT_POLICY_FORMAT,
        "expected_predicate_type": target.ACCEPTANCE_PREDICATE_TYPE,
        "trusted_signers": [
            {
                "identity": envelope_identity,
                "fingerprint": fingerprint,
                "status": "active",
            }
        ],
        "trusted_receipt_verifiers": [
            {
                "identity": "verifier.example/release-qualification",
                "fingerprint": (
                    "sha256:"
                    "74a97c1d8fe8d7d58faac074d3a3a926"
                    "7d8db501d9e4aed77eaeb9ad4efb32ff"
                ),
                "status": "active",
            }
        ],
        "freshness": {
            "max_envelope_age_seconds": 3600,
            "max_evidence_age_seconds": None,
            "clock_skew_seconds": 0,
        },
        "allowed_contract_versions": ["0.15.0"],
        "required_technical_verdict": "pass",
        "allow_countersigned_receipts": True,
    }


def _resign_receipt(
    receipt: dict[str, Any],
    key: ed25519.Ed25519PrivateKey,
) -> None:
    fingerprint = public_key_fingerprint(key.public_key())
    receipt["statement"]["verifier"]["identity"] = "recipient-tests/verifier"
    receipt["statement"]["verifier"]["signing_key_fingerprint"] = fingerprint
    receipt["signature"] = {
        "format": target.SIGNED_RECEIPT_SIGNATURE_FORMAT,
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


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (
            lambda: target._canonical_json_bytes({"bad": float("nan")}),
            "canonical JSON",
        ),
        (lambda: target._normalized_digest(7, label="digest"), "sha256"),
        (lambda: target._safe_identity("../bad", label="identity"), "invalid"),
        (
            lambda: target._timestamp(datetime(2026, 1, 1), label="timestamp"),
            "timezone",
        ),
        (lambda: target._parse_timestamp(7, label="timestamp"), "invalid"),
        (lambda: target._parse_timestamp("not-a-time", label="timestamp"), "invalid"),
        (
            lambda: target._parse_timestamp("2026-01-01T00:00:00", label="timestamp"),
            "timezone",
        ),
        (
            lambda: target._contract_release(
                "invarlock/evidence-verification-receipt-v1",
                "invarlock/evidence-pack-v2",
                "invarlock/comparison-report-v2",
            ),
            "not a supported contract set",
        ),
        (
            lambda: target._contract_release(
                1,
                "invarlock/evidence-pack-v1",
                "invarlock/comparison-report-v1",
            ),
            "not a supported contract set",
        ),
    ],
)
def test_scalar_contract_helpers_fail_closed(action: Any, message: str) -> None:
    with pytest.raises(AcceptanceAttestationError, match=message):
        action()


@pytest.mark.parametrize(
    ("report_format", "expected_release"),
    [
        ("invarlock/comparison-report-v1", "0.13.0"),
        ("invarlock/comparison-report-v2", "0.13.0"),
        ("invarlock/comparison-report-v3", "0.15.0"),
    ],
)
def test_contract_release_tracks_the_report_semantics(
    report_format: str,
    expected_release: str,
) -> None:
    assert (
        target._contract_release(
            "invarlock/evidence-verification-receipt-v1",
            "invarlock/evidence-pack-v1",
            report_format,
        )
        == expected_release
    )


def test_object_and_key_loaders_reject_malformed_or_wrong_types(
    tmp_path: Path,
) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_bytes(b"{")
    with pytest.raises(AcceptanceAttestationError, match="valid JSON"):
        target._load_object(invalid_json, label="object")

    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(AcceptanceAttestationError, match="JSON object"):
        target._load_object(array, label="object")

    bad_key = tmp_path / "bad.pem"
    bad_key.write_bytes(b"not a key")
    with pytest.raises(AcceptanceAttestationError, match="signing key"):
        target._load_private_key(bad_key)
    with pytest.raises(AcceptanceAttestationError, match="public key"):
        target._load_public_key(b"not a key")

    p256 = ec.generate_private_key(ec.SECP256R1())
    private_path = tmp_path / "p256.private.pem"
    private_path.write_bytes(
        p256.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    with pytest.raises(AcceptanceAttestationError, match="must be Ed25519"):
        target._load_private_key(private_path)
    p256_public = p256.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    with pytest.raises(AcceptanceAttestationError, match="must be Ed25519"):
        target._load_public_key(p256_public)

    key = _private_key()
    public = key.public_key()
    assert target._load_public_key(public) is public


def test_no_clobber_and_invalid_evidence_are_rejected(tmp_path: Path) -> None:
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"already here")
    with pytest.raises(AcceptanceAttestationError, match="already exists"):
        target._write_no_clobber(existing, b"replacement")

    private = tmp_path / "envelope-signer.pem"
    _write_private_key(private, _private_key())
    with pytest.raises(AcceptanceAttestationError, match="real directory"):
        target.write_acceptance_attestation(
            GOLDEN / "verification.receipt.json",
            tmp_path / "missing",
            tmp_path / "envelope.json",
            signing_key_path=private,
            signer_identity="recipient-tests/envelope",
            policy_identity="recipient-tests/policy",
            issued_at=ISSUED_AT,
        )


def test_writer_rejects_receipt_bound_to_a_different_manifest(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, evidence)
    manifest_path = evidence / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = _object(manifest_path)
    manifest["comparison_id"] = "different-comparison"
    manifest_path.write_bytes(_canonical(manifest))
    private = tmp_path / "envelope-signer.pem"
    _write_private_key(private, _private_key())

    with pytest.raises(
        AcceptanceAttestationError,
        match="receipt does not bind the supplied evidence manifest",
    ):
        target.write_acceptance_attestation(
            GOLDEN / "verification.receipt.json",
            evidence,
            tmp_path / "acceptance.json",
            signing_key_path=private,
            signer_identity="recipient-tests/envelope",
            policy_identity="recipient-tests/policy",
            issued_at=ISSUED_AT,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.update(extra=True), "receipt fields"),
        (lambda receipt: receipt.__setitem__("statement", []), "statement"),
        (
            lambda receipt: receipt["statement"].update(extra=True),
            "statement fields",
        ),
        (
            lambda receipt: receipt["statement"].__setitem__(
                "format", "invarlock/evidence-verification-receipt-v99"
            ),
            "unsupported",
        ),
        (
            lambda receipt: receipt["statement"].__setitem__("verifier", {}),
            "verifier",
        ),
        (
            lambda receipt: receipt["statement"]["verifier"].__setitem__(
                "identity", "../invalid"
            ),
            "identity",
        ),
        (
            lambda receipt: receipt["statement"]["verifier"].__setitem__(
                "signing_key_fingerprint", "bad"
            ),
            "fingerprint",
        ),
        (
            lambda receipt: receipt["statement"].__setitem__("verdict", {}),
            "verdict",
        ),
        (
            lambda receipt: receipt["statement"]["verdict"].__setitem__("ok", "yes"),
            "verdict fields",
        ),
        (
            lambda receipt: receipt["statement"]["verdict"].__setitem__(
                "integrity_ok", False
            ),
            "successful verdict is inconsistent",
        ),
        (lambda receipt: receipt.__setitem__("signature", None), "signature"),
        (
            lambda receipt: receipt["signature"].update(extra=True),
            "signature fields",
        ),
        (
            lambda receipt: receipt["signature"].__setitem__("format", "wrong"),
            "signature format",
        ),
        (
            lambda receipt: receipt["signature"].__setitem__("algorithm", "rsa"),
            "signature algorithm",
        ),
        (
            lambda receipt: receipt["signature"].__setitem__(
                "public_key", {"encoding": "der", "value": "bad"}
            ),
            "public key",
        ),
        (
            lambda receipt: receipt["signature"].__setitem__("value", "***"),
            "signature verification",
        ),
        (
            lambda receipt: receipt["signature"].__setitem__("value", None),
            "signature verification",
        ),
    ],
)
def test_embedded_receipt_malformed_inputs_are_rejected(
    mutation: Any,
    message: str,
) -> None:
    receipt = _object(GOLDEN / "verification.receipt.json")
    mutation(receipt)

    with pytest.raises(AcceptanceAttestationError, match=message):
        target._authenticate_receipt(receipt)


def test_embedded_receipt_rejects_wrong_key_type_and_fingerprint() -> None:
    receipt = _object(GOLDEN / "verification.receipt.json")
    p256 = ec.generate_private_key(ec.SECP256R1())
    receipt["signature"]["public_key"]["value"] = (
        p256.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )
    with pytest.raises(AcceptanceAttestationError, match="must be Ed25519"):
        target._authenticate_receipt(receipt)

    receipt = _object(GOLDEN / "verification.receipt.json")
    receipt["statement"]["verifier"]["signing_key_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(AcceptanceAttestationError, match="does not match"):
        target._authenticate_receipt(receipt)


def test_embedded_receipt_authenticates_optional_trust_profile_digest() -> None:
    receipt = _object(GOLDEN / "verification.receipt.json")
    receipt["statement"]["verifier"]["trust_profile_digest"] = "sha256:" + "a" * 64
    key = _private_key()
    _resign_receipt(receipt, key)

    statement, identity, fingerprint = target._authenticate_receipt(receipt)

    assert statement["verifier"]["trust_profile_digest"] == "sha256:" + "a" * 64
    assert identity == "recipient-tests/verifier"
    assert fingerprint == public_key_fingerprint(key.public_key())


def test_bound_object_rejects_unsafe_and_unbound_references() -> None:
    manifest = _object(EVIDENCE / "manifest.json")
    reference = manifest["evidence"]["schedule"]
    raw, value = target._bound_object(EVIDENCE, reference, label="schedule")
    assert hashlib.sha256(raw).digest()
    assert value["format_version"].endswith("-v1")

    for invalid in (None, {"path": "/absolute", "digest": reference["digest"]}):
        with pytest.raises(AcceptanceAttestationError):
            target._bound_object(EVIDENCE, invalid, label="schedule")
    with pytest.raises(AcceptanceAttestationError, match="path is invalid"):
        target._bound_object(
            EVIDENCE,
            {"path": "../schedule.json", "digest": reference["digest"]},
            label="schedule",
        )
    with pytest.raises(AcceptanceAttestationError, match="does not match"):
        target._bound_object(
            EVIDENCE,
            {"path": reference["path"], "digest": "sha256:" + "0" * 64},
            label="schedule",
        )


def test_artifact_projection_covers_supported_kinds_and_rejects_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_path = EVIDENCE / "providers/subject/model-artifact.identity.json"
    hf_raw = hf_path.read_bytes()
    hf = _object(hf_path)
    projected = target._artifact(
        hf_raw,
        hf,
        anchor="sha256:" + hashlib.sha256(hf_raw).hexdigest(),
    )
    assert projected["digest_kind"] == "hf_snapshot_tree_sha256"

    with pytest.raises(AcceptanceAttestationError, match="receipt anchor"):
        target._artifact(hf_raw, hf, anchor="sha256:" + "0" * 64)
    invalid_raw = _canonical({"artifact_format": "hf_snapshot"})
    with pytest.raises(AcceptanceAttestationError, match="artifact identity"):
        target._artifact(
            invalid_raw,
            {"artifact_format": "hf_snapshot"},
            anchor="sha256:" + hashlib.sha256(invalid_raw).hexdigest(),
        )

    digest = "1" * 64
    gguf = {
        "format_version": "invarlock/model-artifact-identity-v1",
        "artifact_format": "gguf",
        "artifact_name": "model.gguf",
        "sha256": digest,
        "byte_length": 1,
        "gguf_metadata_sha256": "2" * 64,
        "tensor_inventory_sha256": "3" * 64,
        "tokenizer_metadata_sha256": "4" * 64,
    }
    gguf_raw = _canonical(gguf)
    assert (
        target._artifact(
            gguf_raw,
            gguf,
            anchor="sha256:" + hashlib.sha256(gguf_raw).hexdigest(),
        )["digest_kind"]
        == "file_sha256"
    )

    engine = {
        "format_version": "invarlock/model-artifact-identity-v1",
        "artifact_format": "tensorrt_llm_engine",
        "bundle_name": "engine",
        "engine_bundle_tree_sha256": digest,
        "file_inventory_sha256": "2" * 64,
        "builder_config_sha256": "3" * 64,
        "tokenizer_metadata_sha256": "4" * 64,
        "engine_metadata_sha256": "5" * 64,
        "target_compute_capability": "9.0",
    }
    engine_raw = _canonical(engine)
    assert (
        target._artifact(
            engine_raw,
            engine,
            anchor="sha256:" + hashlib.sha256(engine_raw).hexdigest(),
        )["digest_kind"]
        == "tensorrt_llm_engine_tree_sha256"
    )

    monkeypatch.setattr(target, "load_model_artifact_identity_schema", lambda: {})
    for identity, message in (
        ({"artifact_format": "hf_snapshot"}, "exact content digest"),
        (
            {"artifact_format": "hf_snapshot", "checkpoint_tree_sha256": digest},
            "name is invalid",
        ),
    ):
        raw = _canonical(identity)
        with pytest.raises(AcceptanceAttestationError, match=message):
            target._artifact(
                raw,
                identity,
                anchor="sha256:" + hashlib.sha256(raw).hexdigest(),
            )


def test_metric_projection_rejects_incomplete_sources() -> None:
    with pytest.raises(AcceptanceAttestationError, match="comparison"):
        target._metric({}, {"metric": "exact_match"})
    with pytest.raises(AcceptanceAttestationError, match="report metric"):
        target._metric({"comparison": {}}, {})
    with pytest.raises(AcceptanceAttestationError, match="disagrees"):
        target._metric(
            {"comparison": {"metric": "exact_match"}},
            {"metric": "other"},
        )
    with pytest.raises(AcceptanceAttestationError, match="binding"):
        target._metric(
            {"comparison": {"scorer_extension": []}},
            {"metric": "extension_score"},
        )
    with pytest.raises(AcceptanceAttestationError, match="incomplete"):
        target._metric(
            {"comparison": {"scorer_extension": {}}},
            {"metric": "extension_score"},
        )

    scorer = {
        "scorer_id": "example.score",
        "scorer_version": "1.0",
        "descriptor_sha256": "1" * 64,
        "configuration_sha256": "sha256:" + "2" * 64,
    }
    projection = target._metric(
        {"comparison": {"scorer_extension": scorer}},
        {"metric": "extension_score"},
    )
    assert projection["scorer"]["descriptor_sha256"] == "sha256:" + "1" * 64
    assert projection["scorer"]["configuration_sha256"] == "sha256:" + "2" * 64


def test_subject_file_hashing_rejects_unsafe_paths_and_detects_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"artifact")
    expected = "sha256:" + hashlib.sha256(b"artifact").hexdigest()
    assert target._artifact_path_digest(artifact, digest_kind="file_sha256") == expected

    symlink = tmp_path / "link.gguf"
    symlink.symlink_to(artifact)
    with pytest.raises(AcceptanceAttestationError, match="symlink"):
        target._file_sha256(symlink)
    with pytest.raises(AcceptanceAttestationError, match="regular file"):
        target._file_sha256(tmp_path)
    with pytest.raises(AcceptanceAttestationError, match="read safely"):
        target._file_sha256(tmp_path / "missing")

    before = artifact.stat()
    changed = SimpleNamespace(
        st_dev=before.st_dev,
        st_ino=before.st_ino,
        st_size=before.st_size,
        st_mtime_ns=before.st_mtime_ns + 1,
        st_ctime_ns=before.st_ctime_ns,
    )
    monkeypatch.setattr(target.os, "fstat", lambda _fd: changed)
    with pytest.raises(AcceptanceAttestationError, match="changed"):
        target._file_sha256(artifact)


def test_artifact_path_digest_dispatches_engine_and_rejects_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        target,
        "read_tensorrt_llm_engine_tree_sha256",
        lambda _path: "7" * 64,
    )
    assert (
        target._artifact_path_digest(
            tmp_path,
            digest_kind="tensorrt_llm_engine_tree_sha256",
        )
        == "sha256:" + "7" * 64
    )
    with pytest.raises(AcceptanceAttestationError, match="unsupported"):
        target._artifact_path_digest(tmp_path, digest_kind="other")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda envelope: envelope.update(extra=True), "envelope fields"),
        (
            lambda envelope: envelope.__setitem__("payloadType", "wrong"),
            "payload type",
        ),
        (lambda envelope: envelope.__setitem__("signatures", []), "exactly one"),
        (
            lambda envelope: envelope.__setitem__("signatures", [None]),
            "signature fields",
        ),
        (
            lambda envelope: envelope["signatures"][0].__setitem__("keyid", "bad"),
            "key ID",
        ),
        (
            lambda envelope: envelope.__setitem__("payload", 7),
            "base64 fields",
        ),
        (
            lambda envelope: envelope.__setitem__("payload", "***"),
            "base64 fields",
        ),
        (
            lambda envelope: envelope.__setitem__(
                "payload", base64.b64encode(b"[]").decode("ascii")
            ),
            "JSON object",
        ),
    ],
)
def test_dsse_parser_rejects_malformed_envelopes(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    envelope = _object(GOLDEN / "acceptance.dsse.json")
    mutate(envelope)
    path = tmp_path / "envelope.json"
    path.write_bytes(_canonical(envelope))
    with pytest.raises(AcceptanceAttestationError, match=message):
        target._authenticated_statement(path, {})


def test_dsse_parser_rejects_noncanonical_unknown_and_mismatched_keys(
    tmp_path: Path,
) -> None:
    envelope = _object(GOLDEN / "acceptance.dsse.json")
    statement = json.loads(base64.b64decode(envelope["payload"], validate=True))
    envelope["payload"] = base64.b64encode(
        json.dumps(statement, indent=2).encode()
    ).decode("ascii")
    path = tmp_path / "noncanonical.json"
    path.write_bytes(_canonical(envelope))
    with pytest.raises(AcceptanceAttestationError, match="canonical"):
        target._authenticated_statement(path, {})

    path = GOLDEN / "acceptance.dsse.json"
    with pytest.raises(AcceptanceAttestationError, match="unavailable"):
        target._authenticated_statement(path, {})

    keyid = _object(path)["signatures"][0]["keyid"]
    wrong = _private_key(91).public_key()
    with pytest.raises(AcceptanceAttestationError, match="does not match"):
        target._authenticated_statement(path, {keyid: wrong})

    with pytest.raises(AcceptanceAttestationError, match="public key"):
        target._authenticated_statement(path, {keyid: b"bad key"})


def test_dsse_parser_accepts_public_key_object_and_rejects_bad_signature(
    tmp_path: Path,
) -> None:
    key = _private_key()
    statement = _statement()
    envelope = tmp_path / "valid.json"
    fingerprint, public = _signed_envelope(envelope, statement, key)
    parsed, keyid = target._authenticated_statement(
        envelope,
        {fingerprint: public},
    )
    assert keyid == fingerprint
    assert parsed["predicateType"] == target.ACCEPTANCE_PREDICATE_TYPE

    outer = _object(envelope)
    outer["signatures"][0]["sig"] = base64.b64encode(b"wrong").decode("ascii")
    envelope.write_bytes(_canonical(outer))
    with pytest.raises(AcceptanceAttestationError, match="verification failed"):
        target._authenticated_statement(envelope, {fingerprint: public})


def test_receipt_consistency_reports_every_redundant_binding() -> None:
    predicate = _statement()["predicate"]
    predicate["receipt"]["digest"] = "sha256:" + "0" * 64
    predicate["signers"]["receipt"]["identity"] = "wrong"
    predicate["contracts"]["receipt"] = "invarlock/evidence-verification-receipt-v99"
    predicate["technical_verdict"]["policy_verdict"] = "fail"
    predicate["evaluation_source"]["schedule_digest"] = "sha256:" + "1" * 64
    predicate["policy"]["digest"] = "sha256:" + "2" * 64

    authenticated, errors = target._receipt_consistency_errors(predicate)

    assert authenticated is False
    assert len(errors) >= 6
    assert any("receipt digest" in error for error in errors)
    assert any("evaluation source digest" in error for error in errors)
    assert any("policy digest" in error for error in errors)


def test_receipt_consistency_rejects_raw_content_encoding_and_projection_drift() -> (
    None
):
    content_drift = _statement()["predicate"]
    content_drift["receipt"]["content"]["statement"]["verdict"]["ok"] = False
    authenticated, errors = target._receipt_consistency_errors(content_drift)
    assert authenticated is False
    assert "raw bytes disagree with content" in " ".join(errors)

    invalid_raw = _statement()["predicate"]
    invalid_raw["receipt"]["raw_base64"] = "***"
    authenticated, errors = target._receipt_consistency_errors(invalid_raw)
    assert authenticated is False
    assert "raw bytes are invalid" in " ".join(errors)

    projection_drift = _statement()["predicate"]
    projection_drift["baseline"]["extra"] = True
    authenticated, errors = target._receipt_consistency_errors(projection_drift)
    assert authenticated is True
    assert "artifact disagrees with embedded identity" in " ".join(errors)


def test_recipient_policy_path_loads_one_authenticated_object(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_bytes(_canonical({"format": "fixture-policy"}))

    assert target._load_policy(policy) == {"format": "fixture-policy"}


@pytest.mark.parametrize("anchor_value", [None, {"artifact_digests": None}])
def test_receipt_consistency_rejects_invalid_anchor_shapes(
    anchor_value: object,
) -> None:
    key = _private_key()
    predicate = _statement()["predicate"]
    receipt = copy.deepcopy(predicate["receipt"]["content"])
    if anchor_value is None:
        receipt["statement"]["anchors"] = None
    else:
        receipt["statement"]["anchors"] = anchor_value
    _resign_receipt(receipt, key)
    predicate["receipt"]["content"] = receipt
    receipt_raw = _canonical(receipt)
    predicate["receipt"]["digest"] = "sha256:" + hashlib.sha256(receipt_raw).hexdigest()
    predicate["receipt"]["raw_base64"] = base64.b64encode(receipt_raw).decode("ascii")
    predicate["signers"]["receipt"] = {
        "identity": "recipient-tests/verifier",
        "fingerprint": public_key_fingerprint(key.public_key()),
    }

    authenticated, errors = target._receipt_consistency_errors(predicate)

    assert authenticated is True
    if anchor_value is None:
        assert errors == []
    else:
        assert "receipt artifact anchors are invalid" in " ".join(errors)


def test_receipt_consistency_rejects_identity_payload_disagreement() -> None:
    predicate = _statement()["predicate"]
    predicate["subject"]["artifact_identity"]["model_id"] = "different"

    authenticated, errors = target._receipt_consistency_errors(predicate)

    assert authenticated is True
    assert "payload disagrees" in " ".join(errors)


def test_subject_binding_requires_one_independent_input(tmp_path: Path) -> None:
    statement = _statement()
    predicate = statement["predicate"]
    with pytest.raises(AcceptanceAttestationError, match="exactly one"):
        target._subject_binding_errors(
            statement,
            predicate,
            expected_subject_digest=None,
            subject_artifact_path=None,
        )
    with pytest.raises(AcceptanceAttestationError, match="exactly one"):
        target._subject_binding_errors(
            statement,
            predicate,
            expected_subject_digest=predicate["subject"]["artifact_digest"],
            subject_artifact_path=tmp_path,
        )
    with pytest.raises(AcceptanceAttestationError, match="sha256"):
        target._subject_binding_errors(
            statement,
            predicate,
            expected_subject_digest="bad",
            subject_artifact_path=None,
        )

    statement["subject"] = []
    bound, errors = target._subject_binding_errors(
        statement,
        predicate,
        expected_subject_digest=predicate["subject"]["artifact_digest"],
        subject_artifact_path=None,
    )
    assert bound is True
    assert any("in-toto subject disagrees" in error for error in errors)


def test_recipient_policy_errors_cover_identity_time_and_verdict_rules() -> None:
    statement = _statement()
    predicate = statement["predicate"]
    keyid = predicate["signers"]["envelope"]["fingerprint"]
    policy = _policy(
        keyid,
        envelope_identity=predicate["signers"]["envelope"]["identity"],
    )
    assert not target._recipient_policy_errors(
        statement,
        predicate,
        policy,
        keyid=keyid,
        now=ISSUED_AT + timedelta(minutes=5),
    )

    statement["_type"] = "wrong"
    statement["predicateType"] = "https://example.invalid/predicate"
    predicate["signers"]["envelope"]["fingerprint"] = "sha256:" + "1" * 64
    predicate["signers"]["relationship"] = "same_signer"
    predicate["technical_verdict"]["ok"] = False
    policy["trusted_signers"] = []
    policy["expected_predicate_type"] = "https://example.invalid/other"
    errors = target._recipient_policy_errors(
        statement,
        predicate,
        policy,
        keyid=keyid,
        now=ISSUED_AT + timedelta(minutes=5),
    )
    assert "Statement type" in " ".join(errors)
    assert "predicate type is invalid" in " ".join(errors)
    assert "DSSE key ID" in " ".join(errors)
    assert "not trusted" in " ".join(errors)
    assert "technical verdict" in " ".join(errors)
    assert "relationship is inconsistent" in " ".join(errors)

    predicate = _statement()["predicate"]
    predicate["timestamps"]["attestation_issued_at"] = (
        ISSUED_AT + timedelta(minutes=10)
    ).isoformat()
    future = target._recipient_policy_errors(
        _statement(),
        predicate,
        _policy(
            predicate["signers"]["envelope"]["fingerprint"],
            envelope_identity=predicate["signers"]["envelope"]["identity"],
        ),
        keyid=predicate["signers"]["envelope"]["fingerprint"],
        now=ISSUED_AT,
    )
    assert "future" in " ".join(future)

    with pytest.raises(AcceptanceAttestationError, match="timezone"):
        target._recipient_policy_errors(
            _statement(),
            _statement()["predicate"],
            _policy(
                _statement()["predicate"]["signers"]["envelope"]["fingerprint"],
                envelope_identity=(
                    _statement()["predicate"]["signers"]["envelope"]["identity"]
                ),
            ),
            keyid=_statement()["predicate"]["signers"]["envelope"]["fingerprint"],
            now=datetime(2026, 7, 25, 12, 0),
        )


def test_recipient_policy_errors_require_exactly_one_matching_trust_record() -> None:
    statement = _statement()
    predicate = statement["predicate"]
    keyid = predicate["signers"]["envelope"]["fingerprint"]
    policy = _policy(
        keyid,
        envelope_identity=predicate["signers"]["envelope"]["identity"],
    )
    for registry in ("trusted_signers", "trusted_receipt_verifiers"):
        policy[registry].append({**policy[registry][0], "status": "revoked"})

    errors = target._recipient_policy_errors(
        statement,
        predicate,
        policy,
        keyid=keyid,
        now=ISSUED_AT,
    )

    assert "envelope signer has multiple matching" in " ".join(errors)
    assert "receipt verifier has multiple matching" in " ".join(errors)


def test_public_verifier_rejects_invalid_policy_and_statement_shapes(
    tmp_path: Path,
) -> None:
    key = _private_key()
    statement = _statement()
    envelope = tmp_path / "envelope.json"
    fingerprint, public = _signed_envelope(envelope, statement, key)

    decision = target.verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=[],
        expected_subject_digest=statement["predicate"]["subject"]["artifact_digest"],
        now=ISSUED_AT,
    )
    assert decision.accepted is False
    assert "policy must be an object" in " ".join(decision.errors)

    statement = _statement()
    statement["extra"] = True
    fingerprint, public = _signed_envelope(envelope, statement, key)
    decision = target.verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint),
        expected_subject_digest=statement["predicate"]["subject"]["artifact_digest"],
        now=ISSUED_AT,
    )
    assert "Statement fields" in " ".join(decision.errors)

    statement = _statement()
    statement["predicate"] = []
    fingerprint, public = _signed_envelope(envelope, statement, key)
    decision = target.verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public},
        recipient_policy=_policy(fingerprint),
        expected_subject_digest="sha256:" + "0" * 64,
        now=ISSUED_AT,
    )
    assert "predicate is invalid" in " ".join(decision.errors)


def test_writer_rejects_naive_time_bad_identity_and_existing_output(
    tmp_path: Path,
) -> None:
    private = tmp_path / "envelope-signer.pem"
    _write_private_key(private, _private_key())

    for kwargs, message in (
        ({"signer_identity": "../bad", "issued_at": ISSUED_AT}, "identity"),
        (
            {
                "signer_identity": "recipient-tests/envelope",
                "issued_at": datetime(2026, 7, 25, 12, 0),
            },
            "timezone",
        ),
        (
            {
                "signer_identity": "recipient-tests/envelope",
                "issued_at": ISSUED_AT,
                "evaluation_completed_at": datetime(2026, 7, 25, 11, 0),
            },
            "timezone",
        ),
    ):
        with pytest.raises(AcceptanceAttestationError, match=message):
            target.write_acceptance_attestation(
                GOLDEN / "verification.receipt.json",
                EVIDENCE,
                tmp_path / f"{message}.json",
                signing_key_path=private,
                policy_identity="recipient-tests/policy",
                **kwargs,
            )

    output = tmp_path / "existing-envelope.json"
    output.write_bytes(b"existing")
    with pytest.raises(AcceptanceAttestationError, match="already exists"):
        target.write_acceptance_attestation(
            GOLDEN / "verification.receipt.json",
            EVIDENCE,
            output,
            signing_key_path=private,
            signer_identity="recipient-tests/envelope",
            policy_identity="recipient-tests/policy",
            issued_at=ISSUED_AT,
        )
