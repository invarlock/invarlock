from __future__ import annotations

import base64
import copy
import hashlib
import json
import socket
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jsonschema import Draft202012Validator, ValidationError

from invarlock import public_contracts
from invarlock.acceptance_attestation import verify_acceptance_attestation
from invarlock.judge_measurements.acceptance import (
    verify_judge_evidence,
    verify_judge_evidence_with_policy,
    verify_stored_judge_receipt,
)
from invarlock.judge_measurements.contracts import (
    canonical_payload,
    measurement_plan_digest,
)
from invarlock.judge_measurements.evidence import (
    JudgeEvidenceError,
    object_sha256,
    publish_judge_evidence,
    replay_judge_evidence,
    signed_envelope_bytes,
)
from tests.judge_measurements.test_analysis import _bundle, _retain, _runs

FIXTURES = Path(__file__).parents[1] / "fixtures" / "judge_measurements"
ACCEPTANCE_FIXTURES = (
    Path(__file__).parents[1] / "fixtures" / "judge_measurement_acceptance"
)
KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))


def _json(path):
    return json.loads(path.read_text())


def _write(path, value):
    path.write_bytes(canonical_payload(value) + b"\n")


def _publish(
    tmp_path,
    *,
    role="required",
    baseline=0,
    subject=1,
    incomplete=False,
    signed=True,
    policy_changes=None,
):
    plan, data = _bundle(baseline=baseline, subject=subject)
    runs = _runs(plan)
    if incomplete:
        trial = data["trials"][0]
        trial["status"] = "incomplete"
        trial["parse"] = {"status": "invalid", "rating": None, "value": None}
        response = b'{"rating":"invalid"}'
        trial["attempts"][0]["response"].update(
            text=response.decode(), sha256=hashlib.sha256(response).hexdigest()
        )
        _retain(data)
    analysis_policy = _json(FIXTURES / "analysis_policy.json")
    analysis_policy.update(
        plan_sha256=measurement_plan_digest(plan),
        decision_role=role,
        minimum_units=1,
        maximum_interval_width="2",
        allowed_degradation="0",
    )
    analysis_policy.update(policy_changes or {})
    publication = publish_judge_evidence(
        tmp_path / "evidence",
        plan=plan,
        measurements=data,
        baseline_run=runs[0],
        subject_run=runs[1],
        analysis_policy=analysis_policy,
        signing_key=KEY if signed else None,
        signer_identity="example-signer" if signed else None,
    )
    # Test-owned recipient expectations are supplied separately from the evidence.
    public_key = KEY.public_key().public_bytes_raw()
    recipient = {
        "format": "invarlock/judge-measurement-recipient-policy-v1",
        "decision_scope": "bounded-judge-fixed-benchmark-v1",
        "intended_subject": runs[1]["artifact_digest"],
        "required_metric_name": analysis_policy["metric_name"],
        "trusted_signer": {
            "identity": "example-signer",
            "public_key_sha256": "sha256:" + hashlib.sha256(public_key).hexdigest(),
        },
        "bindings": copy.deepcopy(publication.envelope["bindings"]),
        "required_decision": "pass",
    }
    policy_path = tmp_path / "recipient.json"
    _write(policy_path, recipient)
    return publication, policy_path


def _resign(path):
    envelope = _json(path / "envelope.json")
    envelope["signature"] = base64.b64encode(
        KEY.sign(signed_envelope_bytes(envelope))
    ).decode()
    _write(path / "envelope.json", envelope)


def test_signed_evidence_is_independently_replayed_and_accepted_offline(
    tmp_path, monkeypatch
):
    publication, policy_path = _publish(tmp_path)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: pytest.fail(
            "offline verification attempted a connection"
        ),
    )
    replayed = replay_judge_evidence(publication.path)
    assert replayed.analysis_result == publication.analysis_result
    receipt = verify_judge_evidence_with_policy(publication.path, policy_path)
    assert (
        receipt.authenticated
        and receipt.replayed
        and receipt.verified
        and receipt.accepted
    )
    assert receipt.decision == "pass" and receipt.errors == ()
    assert (
        receipt.bindings.plan_sha256 == publication.envelope["bindings"]["plan_sha256"]
    )
    assert receipt.decision_scope == "bounded-judge-fixed-benchmark-v1"
    Draft202012Validator(
        public_contracts.load_judge_measurement_verification_receipt_schema()
    ).validate(receipt.to_dict())
    assert receipt == verify_judge_evidence_with_policy(publication.path, policy_path)


@pytest.mark.parametrize(
    "field",
    [
        "baseline_run_sha256",
        "subject_run_sha256",
        "case_set_sha256",
        "plan_sha256",
        "measurements_sha256",
        "analysis_policy_sha256",
        "analysis_result_sha256",
    ],
)
def test_each_recipient_binding_is_independently_enforced(tmp_path, field):
    publication, policy_path = _publish(tmp_path)
    policy = _json(policy_path)
    original = policy["bindings"][field]
    policy["bindings"][field] = (
        "sha256:" if original.startswith("sha256:") else ""
    ) + "0" * 64
    _write(policy_path, policy)
    receipt = verify_judge_evidence_with_policy(publication.path, policy_path)
    assert receipt.authenticated and not receipt.replayed
    assert not receipt.verified and not receipt.accepted
    assert field in receipt.errors[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("intended_subject", "sha256:" + "0" * 64),
        ("required_metric_name", "another-metric"),
        ("decision_scope", "native-inference-v1"),
        ("format", "invarlock/recipient-acceptance-policy-v2"),
        ("required_decision", "insufficient_evidence"),
    ],
)
def test_wrong_recipient_subject_metric_scope_or_policy_version_fails(
    tmp_path, field, value
):
    publication, policy_path = _publish(tmp_path)
    policy = _json(policy_path)
    policy[field] = value
    _write(policy_path, policy)
    receipt = verify_judge_evidence_with_policy(publication.path, policy_path)
    assert not receipt.accepted and not receipt.verified


@pytest.mark.parametrize(
    "field,value",
    [("identity", "another-signer"), ("public_key_sha256", "sha256:" + "0" * 64)],
)
def test_wrong_signer_pin_fails_even_with_a_valid_embedded_key(tmp_path, field, value):
    publication, policy_path = _publish(tmp_path)
    policy = _json(policy_path)
    policy["trusted_signer"][field] = value
    _write(policy_path, policy)
    receipt = verify_judge_evidence_with_policy(publication.path, policy_path)
    assert not receipt.authenticated and not receipt.accepted
    assert "signer" in receipt.errors[0]


def test_embedded_public_key_or_signature_substitution_fails(tmp_path):
    publication, policy_path = _publish(tmp_path)
    path = publication.path / "envelope.json"
    original = _json(path)
    forged = copy.deepcopy(original)
    forged["signer"]["public_key"] = base64.b64encode(
        Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    ).decode()
    _write(path, forged)
    assert not verify_judge_evidence_with_policy(
        publication.path, policy_path
    ).authenticated
    forged = copy.deepcopy(original)
    forged["signature"] = base64.b64encode(b"x" * 64).decode()
    _write(path, forged)
    receipt = verify_judge_evidence_with_policy(publication.path, policy_path)
    assert not receipt.accepted and "signature" in receipt.errors[0]


@pytest.mark.parametrize(
    "filename,field,value",
    [
        ("plan.json", "rubric", {}),
        ("baseline_run.json", "run_id", "another-baseline"),
        ("subject_run.json", "artifact_digest", "sha256:" + "c" * 64),
        ("measurements.json", "plan_sha256", "0" * 64),
        ("analysis_policy.json", "allowed_degradation", "1"),
        ("analysis_result.json", "decision", "regression"),
        ("case_set.json", "cases", []),
    ],
)
def test_artifact_substitution_is_not_hidden_by_a_signed_envelope(
    tmp_path, filename, field, value
):
    publication, policy_path = _publish(tmp_path)
    artifact = _json(publication.path / filename)
    artifact[field] = value
    _write(publication.path / filename, artifact)
    receipt = verify_judge_evidence_with_policy(publication.path, policy_path)
    assert not receipt.replayed and not receipt.accepted


def test_validly_signed_forged_analysis_is_recomputed_not_believed(tmp_path):
    publication, policy_path = _publish(tmp_path)
    result_path = publication.path / "analysis_result.json"
    result = _json(result_path)
    result["effect_interval"]["lower"] = "0.999999999999999"
    _write(result_path, result)
    envelope_path = publication.path / "envelope.json"
    envelope = _json(envelope_path)
    envelope["bindings"]["analysis_result_sha256"] = object_sha256(result)
    _write(envelope_path, envelope)
    _resign(publication.path)
    # Even independently pinning this forged claim cannot skip arithmetic replay.
    policy = _json(policy_path)
    policy["bindings"]["analysis_result_sha256"] = object_sha256(result)
    _write(policy_path, policy)
    receipt = verify_judge_evidence_with_policy(publication.path, policy_path)
    assert not receipt.accepted and "independent replay" in receipt.errors[0]


@pytest.mark.parametrize(
    "kwargs,decision,reason",
    [
        ({"incomplete": True}, "insufficient_evidence", "insufficient_evidence"),
        ({"baseline": 1, "subject": 0}, "regression", "regression"),
        ({"role": "advisory"}, "pass", "advisory"),
        (
            {"policy_changes": {"minimum_units": 20}},
            "insufficient_evidence",
            "insufficient_evidence",
        ),
    ],
)
def test_authenticated_insufficient_regression_and_advisory_results_never_accept(
    tmp_path, kwargs, decision, reason
):
    publication, policy_path = _publish(tmp_path, **kwargs)
    receipt = verify_judge_evidence_with_policy(publication.path, policy_path)
    assert receipt.authenticated and receipt.replayed and receipt.verified
    assert not receipt.accepted and receipt.decision == decision
    assert reason in receipt.errors[0]


def test_unsigned_evidence_replays_but_cannot_be_accepted(tmp_path):
    publication, policy_path = _publish(tmp_path, signed=False)
    assert replay_judge_evidence(publication.path).analysis_result.decision == "pass"
    receipt = verify_judge_evidence_with_policy(publication.path, policy_path)
    assert not receipt.replayed and not receipt.authenticated and not receipt.accepted


def test_producer_owned_policy_and_extra_trust_values_do_not_self_authorize(tmp_path):
    publication, policy_path = _publish(tmp_path)
    embedded_policy = publication.path / "recipient.json"
    _write(embedded_policy, _json(policy_path))
    receipt = verify_judge_evidence_with_policy(publication.path, embedded_policy)
    assert not receipt.accepted and "outside submitted" in receipt.errors[0]
    policy = _json(policy_path)
    policy["allow_untrusted_signer"] = True
    _write(policy_path, policy)
    assert not verify_judge_evidence_with_policy(publication.path, policy_path).accepted
    envelope = _json(publication.path / "envelope.json")
    envelope["trust_inputs"] = {"authorized": True}
    _write(publication.path / "envelope.json", envelope)
    with pytest.raises(JudgeEvidenceError, match="envelope is invalid"):
        replay_judge_evidence(publication.path)


def test_optional_external_key_anchor_must_match_the_recipient_pin(tmp_path):
    publication, policy_path = _publish(tmp_path)
    assert verify_judge_evidence(
        publication.path,
        recipient_policy_path=policy_path,
        trusted_public_keys={"example-signer": KEY.public_key()},
    ).accepted
    assert not verify_judge_evidence(
        publication.path,
        recipient_policy_path=policy_path,
        trusted_public_keys={
            "example-signer": Ed25519PrivateKey.generate().public_key()
        },
    ).accepted
    assert not verify_judge_evidence(
        publication.path, recipient_policy_path=policy_path, trusted_public_keys={}
    ).accepted


def test_stored_receipt_is_recomputed_and_cannot_self_authorize(tmp_path):
    publication, policy_path = _publish(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt = verify_judge_evidence_with_policy(publication.path, policy_path)
    _write(receipt_path, receipt.to_dict())
    assert verify_stored_judge_receipt(
        receipt_path, evidence_path=publication.path, recipient_policy_path=policy_path
    ).accepted
    claimed = receipt.to_dict()
    claimed["signer_identity"] = "another-signer"
    _write(receipt_path, claimed)
    assert not verify_stored_judge_receipt(
        receipt_path, evidence_path=publication.path, recipient_policy_path=policy_path
    ).accepted
    _write(receipt_path, {"accepted": True})
    assert not verify_stored_judge_receipt(
        receipt_path, evidence_path=publication.path, recipient_policy_path=policy_path
    ).accepted


def test_native_consumers_reject_new_scopes_and_wrappers(tmp_path):
    publication, policy_path = _publish(tmp_path)
    receipt = verify_judge_evidence_with_policy(publication.path, policy_path).to_dict()
    with pytest.raises(ValidationError):
        Draft202012Validator(
            public_contracts.load_evidence_verification_receipt_v3_schema()
        ).validate(receipt)
    with pytest.raises(ValidationError):
        Draft202012Validator(
            public_contracts.load_recipient_acceptance_policy_schema()
        ).validate(_json(policy_path))
    legacy = verify_acceptance_attestation(
        publication.path / "envelope.json",
        trusted_public_keys={"example-signer": KEY.public_key()},
        recipient_policy=policy_path,
    )
    assert not legacy.accepted


def test_publication_does_not_replace_an_existing_destination(tmp_path):
    publication, policy_path = _publish(tmp_path)
    before = (publication.path / "envelope.json").read_bytes()
    with pytest.raises(OSError):
        _publish(tmp_path)
    assert (publication.path / "envelope.json").read_bytes() == before
    assert verify_judge_evidence_with_policy(publication.path, policy_path).accepted


def test_symlinks_duplicate_json_and_missing_artifacts_fail_closed(tmp_path):
    publication, policy_path = _publish(tmp_path)
    target = publication.path / "analysis_result.json"
    original = target.read_bytes()
    backup = tmp_path / "outside-result.json"
    backup.write_bytes(original)
    target.unlink()
    target.symlink_to(backup)
    assert not verify_judge_evidence_with_policy(publication.path, policy_path).accepted
    target.unlink()
    target.write_text('{"decision":"pass","decision":"pass"}')
    assert not verify_judge_evidence_with_policy(publication.path, policy_path).accepted
    target.unlink()
    assert not verify_judge_evidence_with_policy(publication.path, policy_path).accepted


@pytest.mark.parametrize(
    "artifact,contract,loader",
    [
        (
            "envelope",
            "judge_measurement_evidence",
            public_contracts.load_judge_measurement_evidence_schema,
        ),
        (
            "recipient_policy",
            "judge_measurement_recipient_policy",
            public_contracts.load_judge_measurement_recipient_policy_schema,
        ),
        (
            "verification_receipt",
            "judge_measurement_verification_receipt",
            public_contracts.load_judge_measurement_verification_receipt_schema,
        ),
    ],
)
def test_acceptance_contracts_are_closed_packaged_and_match_golden_fixtures(
    artifact, contract, loader
):
    schema = loader()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    value = _json(ACCEPTANCE_FIXTURES / f"{artifact}.json")
    validator.validate(value)
    root = Path(__file__).parents[2]
    assert (root / "contracts" / f"{contract}.schema.json").read_bytes() == (
        root / "src/invarlock/_data/contracts" / f"{contract}.schema.json"
    ).read_bytes()
    assert loader() == schema and loader() is not schema
    for field in value:
        missing = copy.deepcopy(value)
        del missing[field]
        with pytest.raises(ValidationError):
            validator.validate(missing)
    unknown = copy.deepcopy(value)
    unknown["unapproved"] = True
    with pytest.raises(ValidationError):
        validator.validate(unknown)
    if "bindings" in value:
        unknown = copy.deepcopy(value)
        unknown["bindings"]["extra"] = "0" * 64
        with pytest.raises(ValidationError):
            validator.validate(unknown)


def test_golden_acceptance_fixtures_match_real_offline_verification(tmp_path):
    publication, policy_path = _publish(tmp_path)
    assert publication.envelope == _json(ACCEPTANCE_FIXTURES / "envelope.json")
    assert _json(policy_path) == _json(ACCEPTANCE_FIXTURES / "recipient_policy.json")
    assert verify_judge_evidence_with_policy(
        publication.path, policy_path
    ).to_dict() == _json(ACCEPTANCE_FIXTURES / "verification_receipt.json")


def test_receipt_schema_cannot_claim_acceptance_without_verified_pass():
    schema = Draft202012Validator(
        public_contracts.load_judge_measurement_verification_receipt_schema()
    )
    golden = _json(ACCEPTANCE_FIXTURES / "verification_receipt.json")
    for field, bad in [
        ("verified", False),
        ("authenticated", False),
        ("replayed", False),
        ("decision", "insufficient_evidence"),
        ("bindings", None),
        ("errors", ["unresolved"]),
        ("decision_scope", "native-inference-v1"),
    ]:
        value = copy.deepcopy(golden)
        value[field] = bad
        with pytest.raises(ValidationError):
            schema.validate(value)
