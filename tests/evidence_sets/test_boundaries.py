"""Regression tests for mismatched snapshots and component verifier boundaries."""

import copy
from dataclasses import replace

import pytest

from invarlock.evidence_reporting import EvidenceReportError, render_evidence
from invarlock.evidence_sets import reporting, verification
from tests.evidence_sets.test_verification import fixture


@pytest.mark.parametrize("change", ["statement", "profile", "bindings", "subject"])
def test_composition_rejects_judge_receipt_for_another_input(
    tmp_path, monkeypatch, change
):
    root, policy = fixture(tmp_path)
    original = verification.verify_judge_evidence

    def altered(*args, **kwargs):
        actual = original(*args, **kwargs)
        if change == "statement":
            return replace(actual, envelope_sha256="f" * 64)
        if change == "profile":
            return replace(actual, recipient_policy_sha256="f" * 64)
        if change == "bindings":
            return replace(actual, bindings=None)
        return replace(actual, intended_subject="sha256:" + "f" * 64)

    monkeypatch.setattr(verification, "verify_judge_evidence", altered)
    result = verification.verify_evidence_set(root, recipient_policy=policy)
    assert not result.accepted and not result.payload["verified"]
    assert result.payload["errors"]


def test_captured_verifier_cannot_return_another_statement(tmp_path, monkeypatch):
    root, policy = fixture(tmp_path)
    original = verification.verify_captured_evidence

    def altered(*args, **kwargs):
        actual = original(*args, **kwargs)
        actual["pack_manifest_digest"] = "sha256:" + "f" * 64
        return actual

    monkeypatch.setattr(verification, "verify_captured_evidence", altered)
    result = verification.verify_evidence_set(root, recipient_policy=policy)
    assert not result.accepted and "verified captured statement" in str(
        result.payload["errors"]
    )


@pytest.mark.parametrize(
    "change", ["native_profile", "profile_digest", "policy_type", "run_digest"]
)
def test_captured_profile_loading_cannot_substitute_authorization(
    tmp_path, monkeypatch, change
):
    root, policy = fixture(tmp_path)
    original = verification.load_trust_inputs

    def altered(path):
        actual = original(path)
        if change == "native_profile":
            return object()
        if change == "profile_digest":
            return replace(actual, profile_digest="sha256:" + "f" * 64)
        if change == "policy_type":
            return replace(actual, policy_bytes=b"[]")
        return replace(
            actual,
            expected_run_digests={
                "baseline": "sha256:" + "f" * 64,
                "subject": actual.expected_run_digests["subject"],
            },
        )

    monkeypatch.setattr(verification, "load_trust_inputs", altered)
    result = verification.verify_evidence_set(root, recipient_policy=policy)
    assert not result.accepted and result.payload["errors"]


@pytest.mark.parametrize("target", [verification, reporting])
def test_different_captured_snapshot_is_rejected(tmp_path, monkeypatch, target):
    root, policy = fixture(tmp_path)
    name = "load_captured_snapshot" if target is verification else "captured_snapshot"
    original = getattr(target, name)

    def altered(*args, **kwargs):
        manifest, payloads, signer, raw = original(*args, **kwargs)
        manifest = copy.deepcopy(manifest)
        manifest["comparison_id"] = "sha256:" + "f" * 64
        return manifest, payloads, signer, raw

    monkeypatch.setattr(target, name, altered)
    if target is verification:
        result = verification.verify_evidence_set(root, recipient_policy=policy)
        assert not result.accepted and "changed" in str(result.payload["errors"])
    else:
        with pytest.raises(EvidenceReportError, match="changed"):
            render_evidence(root)


@pytest.mark.parametrize("target", [verification, reporting])
def test_judge_statement_changed_during_preparation_rejected(
    tmp_path, monkeypatch, target
):
    root, policy = fixture(tmp_path)
    original = target.read_object

    def altered(path, *args, **kwargs):
        value, raw = original(path, *args, **kwargs)
        if path.name == "envelope.json":
            raw += b" "
        return value, raw

    monkeypatch.setattr(target, "read_object", altered)
    if target is verification:
        result = verification.verify_evidence_set(root, recipient_policy=policy)
        assert not result.accepted and "changed" in str(result.payload["errors"])
    else:
        with pytest.raises(EvidenceReportError, match="changed"):
            render_evidence(root)


def test_report_cannot_render_a_different_judge_snapshot(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path)
    original = reporting.judge_snapshot

    def altered(path):
        publication, artifacts = original(path)
        envelope = copy.deepcopy(publication.envelope)
        envelope["intended_subject"] = "sha256:" + "f" * 64
        return replace(publication, envelope=envelope), artifacts

    monkeypatch.setattr(reporting, "judge_snapshot", altered)
    with pytest.raises(EvidenceReportError, match="changed"):
        render_evidence(root)


def test_report_rechecks_index_after_snapshot(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path)
    original = reporting.judge_snapshot

    def altered(path):
        actual = original(path)
        index = root / "evidence-set.json"
        index.write_bytes(index.read_bytes() + b" ")
        return actual

    monkeypatch.setattr(reporting, "judge_snapshot", altered)
    with pytest.raises(EvidenceReportError, match="changed"):
        render_evidence(root)


def test_report_checks_subject_artifact_even_when_run_reader_disagrees(
    tmp_path, monkeypatch
):
    root, _ = fixture(tmp_path)
    original = reporting.shared_captured_inputs

    def altered(payloads):
        actual = original(payloads)
        actual["subject_artifact_sha256"] = "sha256:" + "f" * 64
        return actual

    monkeypatch.setattr(reporting, "shared_captured_inputs", altered)
    with pytest.raises(EvidenceReportError, match="artifacts differ"):
        render_evidence(root)


def test_result_json_and_aggregate_size_limit(tmp_path, monkeypatch):
    import json

    root, policy = fixture(tmp_path)
    result = verification.verify_evidence_set(root, recipient_policy=policy)
    assert json.loads(result.as_json()) == result.payload
    monkeypatch.setattr(verification, "RESULT_LIMIT", 1)
    with pytest.raises(ValueError, match="byte limit"):
        verification.verify_evidence_set(root, recipient_policy=policy)
