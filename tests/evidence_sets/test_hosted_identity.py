"""Hosted composition preserves independent pins and forbids identity downgrade."""

import json

import pytest

from invarlock.engine import digest, evaluated_subject_digest
from invarlock.evidence_sets.verification import verify_evidence_set
from invarlock.judge_measurements.evidence import (
    JudgeEvidenceError,
    _validate_native_artifacts,
    replay_judge_evidence,
)
from tests.evaluation_records.test_hosted_service_identity import identity
from tests.evidence_sets.test_verification import fixture, write
from tests.judge_measurements import test_analysis, test_evidence_acceptance


@pytest.fixture
def hosted_runs(monkeypatch):
    original = test_analysis._runs

    def make_hosted(plan):
        runs = original(plan)
        for side, run in zip(("baseline", "subject"), runs, strict=True):
            run.update(artifact_digest=None, service_identity=identity())
            run["service_identity"]["deployment"] = side
        return runs

    monkeypatch.setattr(test_analysis, "_runs", make_hosted)
    monkeypatch.setattr(test_evidence_acceptance, "_runs", make_hosted)


def test_hosted_set_replays_both_signed_components_with_service_subject(
    tmp_path, hosted_runs
):
    root, policy = fixture(tmp_path)
    result = verify_evidence_set(root, recipient_policy=policy)
    assert result.accepted, result.payload
    shared = result.payload["shared_inputs"]
    assert shared["subject_artifact_sha256"] is None
    subject = json.loads((root / "judge/subject_run.json").read_text())
    assert shared["subject_service_identity_sha256"] == evaluated_subject_digest(
        subject
    )
    assert replay_judge_evidence(root / "judge").envelope[
        "intended_subject"
    ] == evaluated_subject_digest(subject)


def test_hosted_set_rejects_different_service_observation_between_components(
    tmp_path, hosted_runs
):
    def change(baseline, subject):
        subject["service_identity"]["exposed_revision"] = "new-revision"

    root, policy = fixture(tmp_path, captured_change=change)
    result = verify_evidence_set(root, recipient_policy=policy)
    assert not result.accepted
    assert not result.payload["shared_inputs_verified"]


@pytest.mark.parametrize("mutation", ["different", "missing", "mixed"])
def test_hosted_set_requires_independent_exclusive_service_pin(
    tmp_path, hosted_runs, mutation
):
    root, policy = fixture(tmp_path)
    approved = json.loads(policy.read_text())
    if mutation == "different":
        approved["shared_inputs"]["subject_service_identity_sha256"] = digest("wrong")
    elif mutation == "missing":
        approved["shared_inputs"].pop("subject_service_identity_sha256")
    else:
        approved["shared_inputs"]["subject_artifact_sha256"] = digest("weights")
    write(policy, approved)
    assert not verify_evidence_set(root, recipient_policy=policy).accepted


def test_hosted_judge_cannot_claim_native_execution_provenance(tmp_path):
    from invarlock.judge_measurements.native_capture import validate_native_capture
    from tests.judge_measurements.test_native_capture import _capture

    capture = _capture(tmp_path)
    baseline, subject = validate_native_capture(capture)
    subject.update(artifact_digest=None, service_identity=identity())
    with pytest.raises(JudgeEvidenceError, match="differs from the frozen"):
        _validate_native_artifacts(
            {
                "native_capture.json": capture,
                "baseline_run.json": baseline,
                "subject_run.json": subject,
            }
        )
    with pytest.raises(JudgeEvidenceError, match="require their original capture"):
        _validate_native_artifacts(
            {"baseline_run.json": baseline, "subject_run.json": subject}
        )


def test_signed_hosted_envelope_cannot_name_another_service(tmp_path, hosted_runs):
    root, _ = fixture(tmp_path)
    envelope_path = root / "judge/envelope.json"
    envelope = json.loads(envelope_path.read_text())
    baseline = json.loads((root / "judge/baseline_run.json").read_text())
    envelope["intended_subject"] = evaluated_subject_digest(baseline)
    write(envelope_path, envelope)
    test_evidence_acceptance._resign(root / "judge")
    with pytest.raises(JudgeEvidenceError, match="intended subject"):
        replay_judge_evidence(root / "judge")
