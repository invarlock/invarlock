"""Malformed authenticated inputs and independent receipt-consistency checks."""

import base64
import json
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from invarlock import captured_contracts as contracts
from invarlock import captured_evidence_publication as publication
from invarlock import captured_verification as verification
from tests.core.test_captured_contract_freeze import (
    _authenticate,
    _canonical,
    _key,
    _pack,
    _raw,
    _rebind,
    _resign_receipt,
    _sha,
    _value,
    _verify_options,
)
from tests.core.test_captured_filesystem_failures import (
    publication_args as publication_args,
)


@pytest.mark.parametrize(
    "raw,message",
    [
        (b"[]\n", "must be a JSON object"),
        (b'{"x":1}', "must be canonical JSON"),
        (b'{"x":1,"x":2}\n', "not strict JSON"),
        (b'{"x":NaN}\n', "not strict JSON"),
        (b'{"x":1e999}\n', "not strict JSON"),
        (b'{"x":"\xff"}\n', "not strict JSON"),
        (b'{"x":"\\ud800"}\n', "not strict JSON"),
        pytest.param(
            b'{"x":' + b"[" * 10000 + b"0" + b"]" * 10000 + b"}\n",
            "not strict JSON",
            id="excessive-nesting",
        ),
    ],
)
def test_strict_json_rejects_ambiguous_noncanonical_or_unbounded_inputs(raw, message):
    with pytest.raises(contracts.CapturedContractError, match=message):
        contracts.json_object(raw, "submitted input")


@pytest.mark.parametrize(
    "raw",
    [
        b" " * (contracts.DETECTOR_LIMIT + 1),
        b'{"format":"unknown","kind":"captured"}\n',
    ],
)
def test_detector_rejects_oversize_or_wrong_family(raw):
    with pytest.raises(
        contracts.CapturedContractError, match="detector byte limit|format or kind"
    ):
        contracts.detect_manifest(raw)


@pytest.mark.parametrize("mutable", [bytearray(b"{}"), memoryview(b"{}")])
def test_snapshot_rejects_mutable_buffers_and_detaches_input_mapping(mutable):
    with pytest.raises(contracts.CapturedContractError, match="immutable bytes"):
        contracts.CapturedSnapshot({"manifest.json": mutable})
    source = {"manifest.json": b"{}\n"}
    snapshot = contracts.CapturedSnapshot(source)
    source["manifest.json"] = b"changed"
    assert snapshot.manifest_bytes == b"{}\n"


def test_memory_inventory_total_limit_counts_every_byte():
    # Share immutable bytes to exercise the real 384 MiB boundary without
    # allocating a separate 128 MiB buffer for each logical payload.
    payload = b"x" * contracts.PAYLOAD_LIMIT
    files = dict.fromkeys(
        ("records/baseline.json", "records/subject.json", "inputs/policy.json"),
        payload,
    )
    assert sum(map(len, files.values())) == contracts.TOTAL_LIMIT
    contracts.check_sizes(files)
    files["checksums.sha256"] = b"\n"
    with pytest.raises(contracts.CapturedContractError, match="total byte limit"):
        contracts.check_sizes(files)


@pytest.mark.parametrize(
    "inventory", [None, {}, {"unexpected": {"path": "../outside"}}]
)
def test_manifest_role_inventory_rejected_before_file_selection(
    tmp_path, monkeypatch, inventory
):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    manifest = _value("manifest-pass.json")
    manifest["files"] = inventory
    (pack / "manifest.json").write_bytes(_canonical(manifest))
    scan = Mock(side_effect=AssertionError("invalid manifest selected inventory"))
    monkeypatch.setattr(contracts, "_inventory", scan)
    with pytest.raises(
        verification.CapturedVerificationError, match="role inventory"
    ) as failure:
        verification.verify_captured_evidence(pack, **options)
    scan.assert_not_called()
    assert failure.value.exit_code == 4
    assert _authenticate(pack, options).ok


@pytest.mark.parametrize(
    "field,value",
    [
        ("algorithm", "rsa"),
        ("format", "invarlock/evidence-verification-receipt-signature-v1"),
        ("extra", True),
        ("public_key", None),
        ("public_key", {"encoding": "pem", "value": "x", "extra": True}),
        ("public_key", {"encoding": "der", "value": "x"}),
        ("signature", {"encoding": "base64", "value": 7}),
    ],
)
def test_malformed_signature_container_is_a_contract_rejection(tmp_path, field, value):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    signature = _value("signature-pass.json")
    signature[field] = value
    (pack / "manifest.signature.json").write_bytes(_canonical(signature))
    with pytest.raises(
        verification.CapturedVerificationError, match="signature"
    ) as failure:
        verification.verify_captured_evidence(pack, **options)
    assert failure.value.exit_code == 4
    authenticated = _authenticate(pack, options)
    assert authenticated.ok
    assert authenticated.statement["verdict"]["verification_status"] == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "ec-key",
        "bad-pem",
        "non-ascii",
        "bad-base64",
        "wrong-signature",
        "signature-fingerprint",
        "manifest-fingerprint",
    ],
)
def test_invalid_manifest_crypto_and_signer_bindings_are_integrity_rejections(
    tmp_path, mutation
):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    signature = _value("signature-pass.json")
    if mutation == "ec-key":
        signature["public_key"]["value"] = (
            ec.derive_private_key(1, ec.SECP256R1())
            .public_key()
            .public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode("ascii")
        )
    elif mutation in {"bad-pem", "non-ascii"}:
        signature["public_key"]["value"] = (
            "invalid PEM" if mutation == "bad-pem" else "\u00e9"
        )
    elif mutation == "bad-base64":
        signature["signature"]["value"] = "not!base64"
    elif mutation == "wrong-signature":
        signature["signature"]["value"] = base64.b64encode(
            _key("verifier").sign(_raw("manifest-pass.json"))
        ).decode("ascii")
    elif mutation == "signature-fingerprint":
        signature["signing_key_fingerprint"] = "sha256:" + "0" * 64
    else:
        manifest = _value("manifest-pass.json")
        manifest["signing_key_fingerprint"] = "sha256:" + "0" * 64
        raw = _canonical(manifest)
        (pack / "manifest.json").write_bytes(raw)
        signature["signature"]["value"] = base64.b64encode(
            _key("signer").sign(raw)
        ).decode("ascii")
    (pack / "manifest.signature.json").write_bytes(_canonical(signature))
    with pytest.raises(
        verification.CapturedVerificationError,
        match="signature is invalid|signer binding",
    ) as failure:
        verification.verify_captured_evidence(pack, **options)
    assert failure.value.exit_code == 6
    authenticated = _authenticate(pack, options)
    assert authenticated.ok
    assert authenticated.statement["scoring_assurance"] is None


@pytest.mark.parametrize("role", ["request", "policy", "baseline", "subject", "report"])
def test_resigned_noncanonical_payload_never_reaches_replay(
    tmp_path, monkeypatch, role
):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    path = pack / contracts.PAYLOADS[role]
    path.write_bytes(json.dumps(json.loads(path.read_bytes()), indent=2).encode())
    _rebind(pack)
    replay = Mock(side_effect=AssertionError("noncanonical payload reached replay"))
    monkeypatch.setattr(verification, "compare_runs", replay)
    with pytest.raises(
        verification.CapturedVerificationError, match=f"{role} must be canonical JSON"
    ) as failure:
        verification.verify_captured_evidence(pack, **options)
    assert failure.value.exit_code == 4
    replay.assert_not_called()
    assert _authenticate(pack, options).ok


@pytest.mark.parametrize("role", ["request", "policy", "baseline", "subject", "report"])
def test_payload_tampering_is_rejected_before_replay(tmp_path, monkeypatch, role):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    with (pack / contracts.PAYLOADS[role]).open("ab") as output:
        output.write(b" ")
    replay = Mock(side_effect=AssertionError("unbound payload reached replay"))
    monkeypatch.setattr(verification, "compare_runs", replay)
    with pytest.raises(
        verification.CapturedVerificationError, match=f"{role} binding is invalid"
    ) as failure:
        verification.verify_captured_evidence(pack, **options)
    assert failure.value.exit_code == 6
    replay.assert_not_called()
    assert _authenticate(pack, options).ok


@pytest.mark.parametrize("role", ["baseline", "subject"])
def test_oversized_record_schedule_precedes_payload_schema_diagnostics(
    tmp_path, monkeypatch, role
):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    path = pack / contracts.PAYLOADS[role]
    value = json.loads(path.read_bytes())
    value["records"] = [{}] * (contracts.record_contracts.MAX_RECORDS + 1)
    path.write_bytes(_canonical(value))
    _rebind(pack)
    validate = Mock(
        side_effect=AssertionError("oversized schedule reached schema diagnostics")
    )
    monkeypatch.setattr(contracts, "validate", validate)
    with pytest.raises(
        verification.CapturedVerificationError, match="record.*limit|records.*50000"
    ) as failure:
        verification.verify_captured_evidence(pack, **options)
    validate.assert_not_called()
    assert failure.value.exit_code == 4
    assert _authenticate(pack, options).ok


@pytest.mark.parametrize("role", ["policy", "baseline", "subject", "report"])
def test_resigned_malformed_payload_is_a_bounded_contract_rejection(
    tmp_path, monkeypatch, role
):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    path = pack / contracts.PAYLOADS[role]
    value = json.loads(path.read_bytes())
    value["unexpected"] = "x" * 1000
    path.write_bytes(_canonical(value))
    _rebind(pack)
    replay = Mock(side_effect=AssertionError("invalid payload reached replay"))
    monkeypatch.setattr(verification, "compare_runs", replay)
    with pytest.raises(verification.CapturedVerificationError) as failure:
        verification.verify_captured_evidence(pack, **options)
    replay.assert_not_called()
    assert failure.value.exit_code == 4
    assert len(str(failure.value)) <= 240
    assert _authenticate(pack, options).ok


@pytest.mark.parametrize("role", ["request", "policy", "baseline", "subject", "report"])
def test_resigned_unpaired_unicode_is_contract_failure_not_internal_error(
    tmp_path, role
):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    path = pack / contracts.PAYLOADS[role]
    value = json.loads(path.read_bytes())
    value["unpaired_unicode"] = "\ud800"
    path.write_bytes(json.dumps(value, ensure_ascii=True).encode())
    _rebind(pack)
    with pytest.raises(
        verification.CapturedVerificationError, match=f"{role} is not strict JSON"
    ) as failure:
        verification.verify_captured_evidence(pack, **options)
    assert failure.value.exit_code == 4
    authenticated = _authenticate(pack, options)
    assert authenticated.ok
    assert authenticated.statement["verdict"]["verification_status"] == 4


@pytest.mark.parametrize("field", ["request_digest", "comparison_id"])
def test_valid_signature_cannot_authorize_inconsistent_manifest_identity(
    tmp_path, field
):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    manifest = _value("manifest-pass.json")
    manifest[field] = "sha256:" + "0" * 64
    raw = _canonical(manifest)
    (pack / "manifest.json").write_bytes(raw)
    (pack / "manifest.signature.json").write_bytes(
        _canonical(contracts.manifest_signature(raw, _key("signer")))
    )
    with pytest.raises(
        verification.CapturedVerificationError, match="request or comparison identity"
    ):
        verification.verify_captured_evidence(pack, **options)
    assert _authenticate(pack, options).ok


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"unsigned": True}, "cannot be combined"),
        ({"signing_key_path": None}, "requires a signing key"),
        ({"request_digest": "sha256:" + "0" * 64}, "digest does not match"),
        ({"normalized_request": None}, "digest does not match"),
    ],
)
def test_invalid_publication_inputs_do_not_create_output(
    tmp_path, publication_args, overrides, message
):
    destination = tmp_path / "uncreated" / "evidence"
    with pytest.raises(publication.CapturedEvidenceError, match=message):
        publication.publish_captured_evidence(
            destination, **{**publication_args, **overrides}
        )
    assert not destination.parent.exists()


@pytest.mark.parametrize("kind", ["missing", "malformed", "ec", "encrypted"])
def test_unusable_signing_key_has_no_partial_publication(
    tmp_path, publication_args, kind
):
    path = publication_args["signing_key_path"]
    if kind == "missing":
        path.unlink()
    elif kind == "malformed":
        path.write_bytes(b"not a PEM key")
    else:
        key = (
            ec.derive_private_key(1, ec.SECP256R1()) if kind == "ec" else _key("signer")
        )
        path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.BestAvailableEncryption(b"test-only")
                if kind == "encrypted"
                else serialization.NoEncryption(),
            )
        )
    destination = tmp_path / "uncreated" / "evidence"
    with pytest.raises(publication.CapturedEvidenceError, match="signing key"):
        publication.publish_captured_evidence(destination, **publication_args)
    assert not destination.parent.exists()


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"expected_baseline_run": None}, "anchors are required"),
        ({"expected_subject_run": 1}, "anchors are required"),
        ({"expected_request_digest": "sha256:" + "A" * 64}, "anchors are required"),
        ({"expected_signer": ""}, "anchors are required"),
        ({"verifier_identity": None}, "valid verifier identity"),
        ({"verifier_identity": "untrusted\nidentity"}, "valid verifier identity"),
        ({"trust_profile_digest": 1}, "trust profile digest"),
        ({"trust_profile_digest": "invalid"}, "trust profile digest"),
        ({"receipt_path": None}, "receipt and verifier signing key"),
        ({"verifier_signing_key_bytes": None}, "receipt and verifier signing key"),
        ({"policy_bytes": bytearray(b"{}")}, "independent policy exceeds byte limit"),
        ({"policy_bytes": b"[]\n"}, "must be a JSON object"),
        (
            {"verifier_signing_key_bytes": bytearray(b"key")},
            "verifier key could not be loaded",
        ),
        (
            {"verifier_signing_key_bytes": b"x" * (contracts.CONTROL_LIMIT + 1)},
            "verifier key could not be loaded",
        ),
    ],
)
def test_malformed_independent_inputs_refuse_before_examining_evidence(
    tmp_path, monkeypatch, overrides, message
):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    snapshot = Mock(
        side_effect=AssertionError("invalid trust inputs examined evidence")
    )
    monkeypatch.setattr(verification, "captured_snapshot", snapshot)
    with pytest.raises(verification.CapturedVerificationIncomplete, match=message):
        verification.verify_captured_evidence(pack, **{**options, **overrides})
    snapshot.assert_not_called()
    assert not options["receipt_path"].exists()


def test_non_ed25519_verifier_key_cannot_issue_receipt(tmp_path):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    options["verifier_signing_key_bytes"] = ec.derive_private_key(
        1, ec.SECP256R1()
    ).private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    with pytest.raises(
        verification.CapturedVerificationIncomplete, match="must be Ed25519"
    ):
        verification.verify_captured_evidence(pack, **options)
    assert not options["receipt_path"].exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "manifest-digest",
        "ec-key",
        "wrong-key",
        "claimed-fingerprint",
        "bad-signature",
        "bad-pem",
    ],
)
def test_receipt_reader_rejects_wrong_keys_and_manifest_binding(tmp_path, mutation):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    receipt = _value("receipt-pass.json")
    if mutation == "manifest-digest":
        receipt["statement"]["pack_manifest_digest"] = "sha256:" + "0" * 64
    elif mutation in {"ec-key", "wrong-key"}:
        key = (
            ec.derive_private_key(1, ec.SECP256R1())
            if mutation == "ec-key"
            else _key("signer")
        )
        receipt["signature"]["public_key"]["value"] = (
            key.public_key()
            .public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode("ascii")
        )
    elif mutation == "claimed-fingerprint":
        receipt["statement"]["verifier"]["signing_key_fingerprint"] = (
            "sha256:" + "0" * 64
        )
    elif mutation == "bad-pem":
        receipt["signature"]["public_key"]["value"] = "not a PEM key"
    _resign_receipt(options["receipt_path"], receipt)
    if mutation == "bad-signature":
        receipt["signature"]["value"] = base64.b64encode(
            _key("signer").sign(_canonical(receipt["statement"]))
        ).decode("ascii")
        options["receipt_path"].write_bytes(_canonical(receipt))
    authenticated = _authenticate(pack, options)
    assert not authenticated.ok
    assert authenticated.statement is None
    assert authenticated.verifier_fingerprint is None
    message = {
        "manifest-digest": "does not bind the examined manifest",
        "ec-key": "must be Ed25519",
        "wrong-key": "key is unauthorized",
        "claimed-fingerprint": "key is unauthorized",
        "bad-signature": "signature is invalid",
        "bad-pem": "Unable to load PEM file",
    }[mutation]
    assert message in authenticated.errors[0]


@pytest.mark.parametrize(
    "mutation", ["unsigned", "signer", "noncanonical", "request-identity"]
)
def test_authorized_receipt_cannot_claim_success_for_contradictory_manifest(
    tmp_path, mutation
):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    manifest = _value("manifest-pass.json")
    if mutation == "unsigned":
        manifest.update(authentication="unsigned_local", signing_key_fingerprint=None)
        del manifest["signature"]
    elif mutation == "signer":
        manifest["signing_key_fingerprint"] = "sha256:" + "0" * 64
    elif mutation == "request-identity":
        manifest["request_digest"] = "sha256:" + "0" * 64
    raw = (
        json.dumps(manifest, indent=2).encode()
        if mutation == "noncanonical"
        else _canonical(manifest)
    )
    (pack / "manifest.json").write_bytes(raw)
    receipt = _value("receipt-pass.json")
    receipt["statement"]["pack_manifest_digest"] = _sha(raw)
    _resign_receipt(options["receipt_path"], receipt)
    authenticated = _authenticate(pack, options)
    assert not authenticated.ok
    assert "contradicts manifest" in authenticated.errors[0]


def test_internally_inconsistent_replay_assurance_cannot_publish_receipt(
    tmp_path, monkeypatch
):
    pack, options = _pack(tmp_path), _verify_options(tmp_path)
    report = _value("comparison-pass.json")
    report["metrics"][0]["name"] = "unplanned-metric"
    (pack / "reports/evaluation.report.json").write_bytes(_canonical(report))
    _rebind(pack)
    # Simulate a replay regression that agrees byte-for-byte with a bad report,
    # but no longer describes the independent policy's metric schedule.
    monkeypatch.setattr(verification, "compare_runs", Mock(return_value=report))
    with pytest.raises(
        verification.CapturedVerificationIncomplete, match="verification_internal_error"
    ):
        verification.verify_captured_evidence(pack, **options)
    assert not options["receipt_path"].exists()
