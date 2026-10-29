"""Receipt reporting authenticates independent expectations, including rejection."""

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.client_review import ClientReviewError, build_client_review
from tests.reporting.test_client_handoff import REFERENCE, context


def captured_inputs(tmp_path):
    # The test owns these expectations; production callers must obtain them independently.
    receipt = json.loads((REFERENCE / "verification.receipt.json").read_text())[
        "statement"
    ]
    anchors = receipt["anchors"]
    verifier = receipt["verifier"]
    values = {
        "format": "invarlock/report-verification-inputs-v1",
        "kind": "captured",
        "expectations": {
            "policy_path": str(REFERENCE / "acceptance.json"),
            "expected_pack_signer_fingerprint": anchors["signer_fingerprint"],
            "expected_verifier_identity": verifier["identity"],
            "expected_verifier_fingerprint": verifier["signing_key_fingerprint"],
            "expected_trust_profile_digest": verifier["trust_profile_digest"],
            "expected_run_digests": {
                "baseline": anchors["baseline_run_digest"],
                "subject": anchors["subject_run_digest"],
            },
            "expected_request_digest": anchors["request_digest"],
        },
    }
    # Retained policy is an independently copied test fixture, not inferred by production.
    policy = tmp_path / "policy.json"
    policy.write_bytes((REFERENCE / "evidence/inputs/policy.json").read_bytes())
    values["expectations"]["policy_path"] = str(policy)
    path = tmp_path / "expectations.json"
    path.write_text(json.dumps(values))
    return path


def test_public_authenticated_rejection_is_not_acceptance(tmp_path):
    inputs = captured_inputs(tmp_path)
    out = tmp_path / "handoff"
    r = CliRunner().invoke(
        app,
        [
            "report",
            str(REFERENCE / "evidence"),
            "--package",
            str(out),
            "--review-context",
            str(context(tmp_path)),
            "--retained-receipt",
            str(REFERENCE / "verification.receipt.json"),
            "--receipt-expectations",
            str(inputs),
            "--json",
        ],
    )
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["recipient_verification"] == "authenticated_rejection"
    snap = json.loads((out / "verification-snapshot.json").read_text())
    assert snap["authentic"] and not snap["accepted"]
    for name in [
        "brief.html",
        "brief.md",
        "review.html",
        "review.md",
        "technical-report.html",
        "technical-report.md",
    ]:
        text = (out / name).read_text()
        assert "Authenticated historical rejection" in text
        assert "Not performed for this recipient" not in text
    assert str(tmp_path) not in (out / "verification-snapshot.json").read_text()


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_verifier_identity", "wrong"),
        ("expected_verifier_fingerprint", "sha256:" + "0" * 64),
        ("expected_pack_signer_fingerprint", "sha256:" + "1" * 64),
        ("expected_request_digest", "sha256:" + "2" * 64),
    ],
)
def test_wrong_expectations_do_not_publish(tmp_path, field, value):
    path = captured_inputs(tmp_path)
    inputs = json.loads(path.read_text())
    inputs["expectations"][field] = value
    path.write_text(json.dumps(inputs))
    with pytest.raises(ClientReviewError):
        build_client_review(
            REFERENCE / "evidence",
            context_path=context(tmp_path),
            output=tmp_path / "out",
            retained_receipt=REFERENCE / "verification.receipt.json",
            receipt_expectations=path,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("family", ["runtime", "judge", "evidence_set"])
@pytest.mark.parametrize("reject", [False, True])
def test_family_receipts_keep_their_scope(tmp_path, family, reject):
    from invarlock.evidence_pack_integrity import public_key_fingerprint
    from tests.reporting.test_client_handoff import ROOT

    if family == "runtime":
        source = ROOT / "examples/acceptance-handoff/golden/evidence"
        receipt = source.parent / "verification.receipt.json"
        st = json.loads(receipt.read_text())["statement"]
        a = st["anchors"]
        v = st["verifier"]
        args = {
            "policy_path": str(source.parent / "evaluated-policy.json"),
            "expected_artifact_digests": a["artifact_digests"],
            "expected_schedule_digest": a["schedule_digest"],
            "expected_runtime_digests": a["runtime_digests"],
            "expected_pack_signer_fingerprint": a["pack_signer_fingerprint"],
            "expected_verifier_identity": v["identity"],
            "expected_verifier_fingerprint": v["signing_key_fingerprint"],
        }
        if reject:
            from invarlock.evidence_verification import verify_evidence
            from tests.judge_measurements.test_evidence_acceptance import VERIFIER_KEY

            receipt = tmp_path / "native-rejection.json"
            args["expected_runtime_digests"] = dict(
                args["expected_runtime_digests"], subject="sha256:" + "9" * 64
            )
            args["expected_verifier_identity"] = "test-owner"
            args["expected_verifier_fingerprint"] = public_key_fingerprint(
                VERIFIER_KEY.public_key()
            )
            try:
                verify_evidence(
                    source,
                    policy_path=Path(args["policy_path"]),
                    expected_baseline_artifact=a["artifact_digests"]["baseline"],
                    expected_subject_artifact=a["artifact_digests"]["subject"],
                    expected_schedule=a["schedule_digest"],
                    expected_baseline_runtime=a["runtime_digests"]["baseline"],
                    expected_subject_runtime=args["expected_runtime_digests"][
                        "subject"
                    ],
                    expected_signer=a["pack_signer_fingerprint"],
                    receipt_path=receipt,
                    verifier_identity="test-owner",
                    verifier_signing_key_path=tmp_path / "test-key.pem",
                    verifier_signing_key_bytes=VERIFIER_KEY.private_bytes(
                        serialization.Encoding.PEM,
                        serialization.PrivateFormat.PKCS8,
                        serialization.NoEncryption(),
                    ),
                )
            except ValueError:
                assert receipt.exists()
    elif family == "judge":
        from invarlock.judge_measurements.acceptance import (
            verify_judge_evidence_with_policy,
            write_signed_judge_verification_receipt,
        )
        from tests.judge_measurements.test_evidence_acceptance import (
            VERIFIER_KEY,
            _publish,
        )

        publication, policy = _publish(
            tmp_path, baseline=1 if reject else 0, subject=0 if reject else 1
        )
        source = publication.path
        receipt = tmp_path / "receipt.json"
        result = verify_judge_evidence_with_policy(source, policy)
        write_signed_judge_verification_receipt(
            source,
            result,
            receipt,
            recipient_policy_path=policy,
            verifier_identity="test-owner",
            verifier_signing_key_path=tmp_path / "test-key.pem",
            verifier_signing_key_bytes=VERIFIER_KEY.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
        )
        args = {
            "recipient_policy_path": str(policy),
            "expected_verifier_identity": "test-owner",
            "expected_verifier_fingerprint": public_key_fingerprint(
                VERIFIER_KEY.public_key()
            ),
        }
    else:
        from invarlock.evidence_sets.verification import verify_evidence_set
        from tests.evidence_sets.test_verification import fixture

        source, policy = fixture(tmp_path, captured_minimum=100 if reject else 1)
        receipt = tmp_path / "receipt.json"
        verify_evidence_set(source, recipient_policy=policy, receipt=receipt)
        args = {"recipient_policy_path": str(policy)}
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "format": "invarlock/report-verification-inputs-v1",
                "kind": family,
                "expectations": args,
            }
        )
    )
    out = tmp_path / "out"
    result = build_client_review(
        source,
        context_path=context(tmp_path),
        output=out,
        retained_receipt=receipt,
        receipt_expectations=inputs,
    )
    assert result["recipient_verification"] == (
        "authenticated_rejection" if reject else "authenticated_acceptance"
    )
    snapshot = json.loads((out / "verification-snapshot.json").read_text())
    assert snapshot["accepted"] is not reject
    if family == "evidence_set":
        assert "no signed joint attestation" in snapshot["mode"]
    assert "PRIVATE KEY" not in (out / "verification-snapshot.json").read_text()


@pytest.mark.parametrize(
    "attack",
    ["policy", "tampered-receipt", "missing-anchor", "run", "submitted-policy"],
)
def test_changed_binding_or_attachment_cannot_reuse_snapshot(tmp_path, attack):
    inputs = captured_inputs(tmp_path)
    values = json.loads(inputs.read_text())
    receipt = REFERENCE / "verification.receipt.json"
    if attack == "policy":
        policy = Path(values["expectations"]["policy_path"])
        value = json.loads(policy.read_text())
        value["metrics"][0]["ratio_max"] = 2
        policy.write_text(json.dumps(value))
    elif attack == "tampered-receipt":
        value = json.loads(receipt.read_text())
        value["statement"]["verdict"]["ok"] = True
        receipt = tmp_path / "tampered.json"
        receipt.write_text(json.dumps(value))
    elif attack == "missing-anchor":
        del values["expectations"]["expected_request_digest"]
    elif attack == "run":
        values["expectations"]["expected_run_digests"]["subject"] = "sha256:" + "0" * 64
    else:
        values["expectations"]["policy_path"] = str(
            REFERENCE / "evidence/inputs/policy.json"
        )
    inputs.write_text(json.dumps(values))
    with pytest.raises(ClientReviewError):
        build_client_review(
            REFERENCE / "evidence",
            context_path=context(tmp_path),
            output=tmp_path / "out",
            retained_receipt=receipt,
            receipt_expectations=inputs,
        )
    assert not (tmp_path / "out").exists()
