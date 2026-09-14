"""Native answers retain and replay the exact provider evidence boundary."""

import base64
import copy
import json
from html import unescape
from pathlib import Path

import pytest

from invarlock.evidence_pack_contract import canonical_json_bytes, sha256_digest
from tests.evidence_packs.test_evidence_pack import _request, _schedule, _side_evidence


def _capture(tmp_path, *, observations=()):
    from invarlock.judge_measurements.native_capture import create_native_capture

    schedule = _schedule()
    fixtures = Path(__file__).parents[1] / "fixtures" / "judge_measurements"
    plan = json.loads((fixtures / "plan.json").read_text())
    for key in (
        "case_set_sha256",
        "baseline_run_sha256",
        "subject_run_sha256",
        "answer_bindings",
    ):
        del plan[key]
    del plan["rubric"]["sha256"]
    del plan["schedule"]["expected_trials"]
    plan["sampling"]["case_units"] = [
        {"case_id": row.record_id, "unit_id": row.record_id} for row in schedule.records
    ]
    analysis = json.loads((fixtures / "analysis_policy.json").read_text())
    del analysis["plan_sha256"]
    analysis.update(
        minimum_units=1, maximum_interval_width="2", allowed_degradation="0"
    )
    recipe = {
        "format": "invarlock/native-judge-policy-v1",
        "plan": plan,
        "analysis": analysis,
        "runner": {
            "scorer_id": "factual-correctness",
            "invocation_timeout_seconds": 120,
        },
        "collection": {
            "grader": "example-judge",
            "inspect_version": "0.3.263",
            "profile": "inspect-text-frozen-answer-v1",
            "epochs": 1,
            "log_model_api": True,
            "log_samples": True,
            "sdk_max_retries": 0,
            "tools": False,
            "max_calls": 4,
            "max_input_tokens": 4000,
            "max_output_tokens": 512,
            "max_cost_microusd": 2000000,
            "input_tokens_per_call": 1000,
            "cost_microusd_per_call": 500000,
            "concurrency": 1,
            "requests_per_minute": 10,
            "request_timeout_seconds": 30,
        },
    }
    policy = canonical_json_bytes(recipe)
    sides = [
        _side_evidence(
            tmp_path / side,
            schedule=schedule,
            image_digest="sha256:" + marker * 64,
            artifact_marker=artifact,
            outputs=("A", "B"),
            role=side,
            policy_digest=sha256_digest(policy),
        )
        for side, marker, artifact in (("baseline", "a", "c"), ("subject", "b", "d"))
    ]
    request = _request("judge")
    request["comparison"]["judge"] = {
        "workspace": "judge",
        "signer_identity": "example-signer",
    }
    if observations:
        request["observations"] = [
            {
                "id": item.observation_id,
                "kind": item.kind,
                "scope": item.scope,
                "payload_digest": sha256_digest(item.payload),
            }
            for item in observations
        ]
    return create_native_capture(
        request,
        schedule.to_payload(),
        policy,
        recipe,
        *sides,
        observations=observations,
    )


def test_capture_replays_exact_bytes_and_deterministic_answers(tmp_path):
    from invarlock.judge_measurements.native_capture import validate_native_capture

    capture = _capture(tmp_path)
    first = validate_native_capture(capture)
    assert first == validate_native_capture(json.loads(json.dumps(capture)))
    assert first[0]["records"][0]["output"] == "A"
    assert first[0]["source"]["name"] == "invarlock-native-judge"
    assert first[0]["source_digest"] == sha256_digest(
        canonical_json_bytes(capture, newline=False)
    )


@pytest.mark.parametrize(
    "field",
    [
        "run_report",
        "runtime_manifest",
        "runtime_config",
        "artifact_identity",
        "provider_receipt",
        "scoring_observation",
    ],
)
def test_capture_rejects_tampered_original_side_bytes(tmp_path, field):
    from invarlock.judge_measurements.native_capture import validate_native_capture

    capture = _capture(tmp_path)
    capture["baseline"][field] = base64.b64encode(b"{}").decode()
    with pytest.raises(ValueError):
        validate_native_capture(capture)


def test_capture_rejects_policy_schedule_and_request_substitution(tmp_path):
    from invarlock.judge_measurements.native_capture import validate_native_capture

    capture = _capture(tmp_path)
    for mutate in (
        lambda c: c["recipe"].update(extra=True),
        lambda c: c.update(policy_base64=base64.b64encode(b"{}").decode()),
        lambda c: c["schedule"]["records"][0].update(expected_output="C"),
        lambda c: c["normalized_request"]["comparison"]["baseline"]["runtime"].update(
            provider="other"
        ),
        lambda c: c.update(unrecognized=True),
    ):
        changed = copy.deepcopy(capture)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_native_capture(changed)


def _publication(tmp_path, *, observations=()):
    import hashlib

    from invarlock.judge_measurements.contracts import (
        expected_trial_id,
        measurement_plan_digest,
        render_judge_request,
    )
    from invarlock.judge_measurements.evidence import publish_judge_evidence
    from invarlock.judge_measurements.native_capture import validate_native_capture
    from invarlock.judge_measurements.native_recipe import finalize_native_plan
    from tests.judge_measurements.test_analysis import _bundle, _retain
    from tests.judge_measurements.test_evidence_acceptance import KEY

    capture = _capture(tmp_path, observations=observations)
    baseline, subject = validate_native_capture(capture)
    plan, policy = finalize_native_plan(capture["recipe"], baseline, subject)
    _, measurements = _bundle(groups=("u0", "u1"))
    digest = measurement_plan_digest(plan)
    measurements["plan_sha256"] = digest
    for trial in measurements["trials"]:
        index = int(trial["case_id"].split("-")[1])
        run = baseline if trial["side"] == "baseline" else subject
        row = run["records"][index]
        trial.update(
            case_id=row["id"],
            plan_sha256=digest,
            trial_id=expected_trial_id(
                digest, row["id"], trial["side"], trial["repetition"]
            ),
            answer_sha256=hashlib.sha256(row["output"].encode()).hexdigest(),
        )
        request = render_judge_request(
            plan, input_text=row["input"], answer_text=row["output"]
        )
        trial["attempts"][0]["request"].update(
            text=request.decode(), sha256=hashlib.sha256(request).hexdigest()
        )
    _retain(measurements)
    publication = publish_judge_evidence(
        tmp_path / "evidence",
        plan=plan,
        measurements=measurements,
        baseline_run=baseline,
        subject_run=subject,
        analysis_policy=policy,
        native_capture=capture,
        signing_key=KEY,
        signer_identity="example-signer",
    )
    return publication


def test_native_publication_replay_and_receipt_retain_capture(tmp_path, monkeypatch):
    import socket

    from invarlock.judge_measurements.acceptance import verify_judge_evidence
    from invarlock.judge_measurements.evidence import (
        object_sha256,
        replay_judge_evidence,
    )
    from invarlock.judge_measurements.reporting import _snapshot
    from tests.judge_measurements.test_evidence_acceptance import _write

    publication = _publication(tmp_path)
    recipient = {
        "format": "invarlock/judge-measurement-recipient-policy-v1",
        "decision_scope": publication.envelope["decision_scope"],
        "intended_subject": publication.envelope["intended_subject"],
        "required_metric_name": "factual-correctness",
        "trusted_signer": {
            key: publication.envelope["signer"][key]
            for key in ("identity", "public_key_sha256")
        },
        "bindings": publication.envelope["bindings"],
        "required_decision": "pass",
    }
    policy_path = tmp_path / "recipient.json"
    _write(policy_path, recipient)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *_a, **_k: pytest.fail("network during replay"),
    )
    assert replay_judge_evidence(publication.path).envelope == publication.envelope
    result = verify_judge_evidence(publication.path, recipient_policy_path=policy_path)
    assert result.verified and result.authenticated and result.replayed, result.errors
    capture = json.loads((publication.path / "native_capture.json").read_text())
    assert result.to_dict()["bindings"]["native_capture_sha256"] == object_sha256(
        capture
    )
    assert _snapshot(publication.path)[1]["native_capture"] == capture

    from invarlock.judge_measurements.acceptance import (
        verify_signed_judge_verification_receipt,
        write_signed_judge_verification_receipt,
    )
    from tests.judge_measurements.test_evidence_acceptance import _verifier_key_path

    receipt_path = tmp_path / "receipt.json"
    fingerprint = write_signed_judge_verification_receipt(
        publication.path,
        result,
        receipt_path,
        recipient_policy_path=policy_path,
        verifier_identity="recipient",
        verifier_signing_key_path=_verifier_key_path(tmp_path),
    )
    verified = verify_signed_judge_verification_receipt(
        receipt_path,
        expected_verifier_identity="recipient",
        expected_verifier_fingerprint=fingerprint,
        expected_recipient_policy_sha256=object_sha256(recipient),
    )
    assert verified.ok, verified.errors
    assert verified.result.to_dict()["bindings"][
        "native_capture_sha256"
    ] == object_sha256(capture)


def test_native_capture_cannot_be_stripped_or_replaced(tmp_path):
    from invarlock.judge_measurements.evidence import (
        JudgeEvidenceError,
        replay_judge_evidence,
    )
    from tests.judge_measurements.test_evidence_acceptance import _write

    publication = _publication(tmp_path)
    capture_path = publication.path / "native_capture.json"
    capture_path.unlink()
    with pytest.raises((OSError, ValueError)):
        replay_judge_evidence(publication.path)
    envelope = copy.deepcopy(publication.envelope)
    del envelope["bindings"]["native_capture_sha256"]
    _write(publication.path / "envelope.json", envelope)
    with pytest.raises(JudgeEvidenceError, match="require their original capture"):
        replay_judge_evidence(publication.path)


def test_native_replay_rejects_resigned_tampered_capture(tmp_path):
    from invarlock.judge_measurements.evidence import (
        object_sha256,
        replay_judge_evidence,
    )
    from tests.judge_measurements.test_evidence_acceptance import _write

    publication = _publication(tmp_path)
    path = publication.path / "native_capture.json"
    capture = json.loads(path.read_text())
    capture["baseline"]["runtime_config"] = base64.b64encode(b"{}").decode()
    _write(path, capture)
    # Even recomputing all direct hashes cannot replace the original provider files.
    envelope = copy.deepcopy(publication.envelope)
    envelope["bindings"]["native_capture_sha256"] = object_sha256(capture)
    _write(publication.path / "envelope.json", envelope)
    with pytest.raises(ValueError):
        replay_judge_evidence(publication.path)


def test_native_capture_retains_and_authenticates_observations(tmp_path):
    from invarlock.evidence_pack_contract import EvidenceObservation
    from invarlock.judge_measurements.native_capture import validate_native_capture

    payload = canonical_json_bytes({"runtime_note": "retained diagnostic"})
    capture = _capture(
        tmp_path,
        observations=(
            EvidenceObservation("diagnostic", "baseline", "runtime.note", payload),
        ),
    )
    assert base64.b64decode(capture["observations"][0]["payload_base64"]) == payload
    validate_native_capture(capture)
    for mutate in (
        lambda c: c["observations"][0].update(scope="subject"),
        lambda c: c["observations"][0].update(
            payload_base64=base64.b64encode(b"{}\n").decode()
        ),
        lambda c: c.update(observations=[]),
        lambda c: c["observations"].append(c["observations"][0]),
    ):
        changed = copy.deepcopy(capture)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_native_capture(changed)


@pytest.mark.parametrize("artifact", ["plan.json", "analysis_policy.json"])
def test_native_capture_rejects_changed_analysis_choices(tmp_path, artifact):
    from invarlock.judge_measurements.evidence import (
        object_sha256,
        replay_judge_evidence,
    )
    from tests.judge_measurements.test_evidence_acceptance import _write

    publication = _publication(tmp_path)
    path = publication.path / artifact
    value = json.loads(path.read_text())
    if artifact == "plan.json":
        value["sampling"]["case_units"][0]["unit_id"] = "changed-unit"
        binding = "plan_sha256"
    else:
        value["allowed_degradation"] = "0.1"
        binding = "analysis_policy_sha256"
    _write(path, value)
    envelope = copy.deepcopy(publication.envelope)
    envelope["bindings"][binding] = object_sha256(value)
    _write(publication.path / "envelope.json", envelope)
    with pytest.raises(ValueError, match="differs from the captured recipe"):
        replay_judge_evidence(publication.path)


def test_native_report_names_models_and_replayed_execution_provenance(tmp_path):
    from invarlock.evidence_pack_contract import EvidenceObservation
    from invarlock.judge_measurements.reporting import render_judge_evidence

    publication = _publication(
        tmp_path,
        observations=(
            EvidenceObservation(
                "diagnostic",
                "baseline",
                "runtime.note",
                canonical_json_bytes({"note": "retained"}),
            ),
        ),
    )
    markdown = tmp_path / "report.md"
    html = tmp_path / "report.html"
    report = render_judge_evidence(
        publication.path, markdown_path=markdown, html_path=html
    )
    assert not report.errors
    assert report.facts["baseline"] == "model-c.gguf"
    assert report.facts["subject"] == "model-d.gguf"
    native = report.facts["native_capture"]
    assert native["same_artifact"] is False
    assert native["same_runtime_settings"] is False
    assert report.facts["comparison"]["baseline"]["runtime"]["provider"] == "llama_cpp"
    for rendered in (markdown.read_text(), html.read_text()):
        assert "model-c.gguf" in rendered and "model-d.gguf" in rendered
        assert "Native runtime provenance" in rendered
        assert "diagnostic (baseline, runtime.note)" in unescape(rendered)
        assert "Different artifacts; different runtime settings" in rendered
        assert "does not establish model execution" not in rendered
        assert "Rate factual correctness." in rendered
        assert "example-judge" in rendered
