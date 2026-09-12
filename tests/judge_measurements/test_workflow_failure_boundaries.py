"""Trust, publication, and preparation failures must retain their distinct scope."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.judge_measurements import acceptance, analysis, evidence, workflow
from tests.judge_measurements.test_contracts import _plan
from tests.judge_measurements.test_evidence_acceptance import (
    KEY,
    _json,
    _publish,
    _write,
)
from tests.judge_measurements.test_workflow import staged as staged


@pytest.mark.parametrize("kind", ["raw", "pem", "path"])
def test_additional_recipient_key_material_authenticates_same_signer(tmp_path, kind):
    publication, policy = _publish(tmp_path)
    public = KEY.public_key()
    material = (
        public.public_bytes_raw()
        if kind == "raw"
        else public.public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    if kind == "path":
        path = tmp_path / "trusted-public.pem"
        path.write_bytes(material)
        material = path
    result = acceptance.verify_judge_evidence(
        publication.path,
        recipient_policy_path=policy,
        trusted_public_keys={"example-signer": material},
    )
    assert result.accepted and result.authenticated and result.replayed


def test_wrong_public_key_algorithm_is_not_a_recipient_anchor(tmp_path):
    publication, policy = _publish(tmp_path)
    material = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    result = acceptance.verify_judge_evidence(
        publication.path,
        recipient_policy_path=policy,
        trusted_public_keys={"example-signer": material},
    )
    assert not result.accepted and not result.replayed
    assert "must be Ed25519" in result.errors[0]


def publication_inputs(publication):
    return {
        field: _json(publication.path / filename)
        for field, filename in (
            ("plan", "plan.json"),
            ("measurements", "measurements.json"),
            ("baseline_run", "baseline_run.json"),
            ("subject_run", "subject_run.json"),
            ("analysis_policy", "analysis_policy.json"),
        )
    }


@pytest.mark.parametrize("signing_key,identity", [(KEY, None), (None, "signer")])
def test_publication_requires_key_and_signer_identity_together(
    tmp_path, signing_key, identity
):
    with pytest.raises(evidence.JudgeEvidenceError, match="supplied together"):
        evidence.publish_judge_evidence(
            tmp_path / "new",
            plan={},
            measurements={},
            baseline_run={},
            subject_run={},
            analysis_policy={},
            signing_key=signing_key,
            signer_identity=identity,
        )
    assert not (tmp_path / "new").exists()


def test_wrong_private_key_algorithm_does_not_publish(tmp_path):
    publication, _ = _publish(tmp_path)
    path = tmp_path / "ec.pem"
    path.write_bytes(
        ec.generate_private_key(ec.SECP256R1()).private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    with pytest.raises(evidence.JudgeEvidenceError, match="must be Ed25519"):
        evidence.publish_judge_evidence(
            tmp_path / "new",
            **publication_inputs(publication),
            signing_key=path,
            signer_identity="signer",
        )
    assert not (tmp_path / "new").exists()


def test_publication_size_failure_leaves_no_partial_evidence(tmp_path, monkeypatch):
    publication, _ = _publish(tmp_path)
    monkeypatch.setitem(evidence.ARTIFACT_BYTE_LIMITS, "analysis_result.json", 1)
    with pytest.raises(
        evidence.JudgeEvidenceError, match="analysis_result.json exceeds"
    ):
        evidence.publish_judge_evidence(
            tmp_path / "new", **publication_inputs(publication)
        )
    assert not (tmp_path / "new").exists()
    assert not list(tmp_path.glob(".judge-evidence-*"))


@pytest.mark.parametrize("mutation", ["case_set", "intended_subject"])
def test_rebound_envelope_cannot_change_case_membership_or_intended_subject(
    tmp_path, mutation
):
    publication, _ = _publish(tmp_path)
    root = publication.path
    envelope = _json(root / "envelope.json")
    if mutation == "case_set":
        cases = _json(root / "case_set.json")
        cases["cases"][0]["input"] = "Changed case"
        _write(root / "case_set.json", cases)
        envelope["bindings"]["case_set_sha256"] = evidence.case_set_digest(cases)
    else:
        envelope["intended_subject"] = "sha256:" + "0" * 64
    _write(root / "envelope.json", envelope)
    with pytest.raises(
        evidence.JudgeEvidenceError,
        match="case set does not match|intended subject does not match",
    ):
        evidence.replay_judge_evidence(root)


def test_evidence_object_loader_rejects_nonobject_json(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("[]")
    with pytest.raises(evidence.JudgeEvidenceError, match="must contain a JSON object"):
        evidence.read_object(path)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"direction": "sideways"}, "direction"),
        ({"allowed_degradation": Decimal("1.01")}, "allowed_degradation"),
        ({"subject_bound": Decimal("0.0000000000000001")}, "fractional digits"),
        ({"plan_sha256": "sha256:" + "a" * 64}, "bare lowercase"),
        ({"metric_name": "bad\nname"}, "bounded identifier"),
    ],
)
def test_in_memory_analysis_policy_rejects_ambiguous_identifiers_and_thresholds(
    changes, message
):
    with pytest.raises(ValueError, match=message):
        analysis.JudgeAnalysisPolicy(
            **{"direction": "higher", "allowed_degradation": Decimal("0"), **changes}
        )


def test_analysis_policy_rejects_nonserializable_input():
    with pytest.raises(ValueError, match="not canonical JSON"):
        analysis.validate_analysis_policy({"unknown": object()}, plan=_plan())


def test_request_overrides_stay_within_request_root(staged):
    path, _ = staged
    root = path.parent
    loaded = workflow.load_judge_request(
        path,
        baseline_run=root / "baseline_run.json",
        subject_run=root / "subject_run.json",
    )
    assert loaded.inputs["baseline_run"] == root / "baseline_run.json"
    assert workflow.preflight_judge_request(loaded).payload["ready"]
    with pytest.raises(workflow.JudgeWorkflowError):
        workflow.load_judge_request(path, baseline_run=root.parent / "outside.json")


def test_evidence_destination_cannot_contain_future_input_files(staged):
    path, value = staged
    value["comparison"]["plan"] = "evidence/plan.json"
    path.write_text(json.dumps(value))
    with pytest.raises(
        workflow.JudgeWorkflowError, match="outside the evidence destination"
    ):
        workflow.load_judge_request(path)


@pytest.mark.parametrize(
    "filename", ["plan.json", "analysis_policy.json", "baseline_run.json"]
)
def test_missing_inputs_keep_preflight_and_evaluation_unready(staged, filename):
    path, _ = staged
    (path.parent / filename).unlink()
    loaded = workflow.load_judge_request(path)
    result = workflow.preflight_judge_request(loaded).payload
    assert not result["ready"] and result["missing_inputs"]
    with pytest.raises(workflow.JudgeWorkflowError) as caught:
        workflow.evaluate_judge_request(loaded, signing_key=None, unsigned=True)
    assert caught.value.payload["missing_inputs"] == result["missing_inputs"]
    assert not loaded.evidence.exists()


def test_collection_without_a_plan_reports_configuration_not_yet_validated(staged):
    path, value = staged
    value["execution"] = {
        "mode": "judge_collect",
        "collection": {
            "integration": "inspect-judge",
            "configuration": "collection.json",
        },
    }
    value["comparison"]["measurements"] = None
    path.write_text(json.dumps(value))
    (path.parent / "collection.json").write_text("{}")
    (path.parent / "plan.json").unlink()
    result = workflow.preflight_judge_request(workflow.load_judge_request(path)).payload
    assert not result["ready"] and result["budgets"] is None
    assert any("Supply the plan" in error for error in result["errors"])


@pytest.mark.parametrize("kind", ["nonobject", "aggregate_bytes", "publication_io"])
def test_workflow_failures_are_bounded_and_do_not_publish(staged, monkeypatch, kind):
    path, _ = staged
    loaded = workflow.load_judge_request(path)
    if kind == "nonobject":
        (path.parent / "plan.json").write_text("[]")
        message = "must be a JSON object"
    elif kind == "aggregate_bytes":
        monkeypatch.setattr(workflow, "MAX_WORKFLOW_BYTES", 1)
        message = "combined judge inputs"
    else:

        def fail(*args, **kwargs):
            raise OSError("publication unavailable")

        monkeypatch.setattr(evidence, "publish_judge_evidence", fail)
        message = "publication unavailable"
    with pytest.raises(workflow.JudgeWorkflowError, match=message):
        workflow.evaluate_judge_request(loaded, signing_key=None, unsigned=True)
    assert not loaded.evidence.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 2),
        ("tools", True),
        ("inspect_version", "0.3.254"),
        ("concurrency", 0),
        ("max_input_tokens", True),
    ],
)
def test_collection_preflight_rejects_unsupported_controls(field, value):
    collection = json.loads(
        (
            Path(__file__).parents[2] / "examples/judge-measurements/collection.json"
        ).read_text()
    )
    collection[field] = value
    with pytest.raises(
        workflow.JudgeWorkflowError, match="settings differ|bounded positive integer"
    ):
        workflow._collection_budgets(collection, _plan())


def test_collection_preflight_cannot_enable_transport_retries():
    plan = _plan()
    plan["schedule"]["max_attempts"] = 2
    collection = json.loads(
        (
            Path(__file__).parents[2] / "examples/judge-measurements/collection.json"
        ).read_text()
    )
    with pytest.raises(workflow.JudgeWorkflowError, match="one attempt per trial"):
        workflow._collection_budgets(collection, plan)


@pytest.mark.parametrize("failure", ["bootstrap", "inside_receipt", "existing_receipt"])
def test_recipient_cli_rejects_unsupported_options_and_receipt_publication_failures(
    tmp_path, failure
):
    publication, policy = _publish(tmp_path)
    args = ["verify", str(publication.path), "--trust-profile", str(policy), "--json"]
    if failure == "bootstrap":
        args.extend(["--max-bootstrap-draws", "100"])
    else:
        receipt = (
            publication.path / "receipt.json"
            if failure == "inside_receipt"
            else tmp_path / "receipt.json"
        )
        if failure == "existing_receipt":
            receipt.write_text("retained earlier receipt")
        args.extend(["--receipt", str(receipt)])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == (2 if failure == "bootstrap" else 4)
    payload = json.loads(result.stdout)
    assert payload["kind"] == "judge" and not payload["accepted"]
    if failure == "existing_receipt":
        assert receipt.read_text() == "retained earlier receipt"
