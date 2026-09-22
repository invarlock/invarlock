"""Native judge selection preserves runtime authentication and public results."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.core.evaluation_request import load_evaluation_request
from invarlock.core.runtime_provider import (
    RuntimeScoringRecord,
    build_runtime_behavioral_schedule,
)
from invarlock.evaluation_run import EvaluationRunResult, load_runtime_side_evidence
from invarlock.evaluation_runtime import CallerRuntimeResources
from invarlock.evaluation_transaction import (
    EvaluationPreflightError,
    EvaluationTransactionError,
    evaluate_request_file,
    preflight_evaluation_request,
)
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.judge_measurements import native_workflow
from invarlock.judge_measurements.contracts import (
    canonical_payload,
    expected_trial_id,
    measurement_plan_digest,
    render_judge_request,
)
from invarlock.judge_measurements.native_capture import validate_native_capture
from invarlock.judge_measurements.workflow import (
    JudgeWorkflowError,
    JudgeWorkflowResult,
)
from tests.cli.test_evaluation_preflight import _materialize_run_request
from tests.cli.test_import_journey import (
    _key,
    _materialize_request,
    _settings,
    _side_evidence,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "judge_measurements"


def _recipe(case_ids=("one", "two")):
    plan = json.loads((FIXTURES / "plan.json").read_text())
    for field in (
        "case_set_sha256",
        "baseline_run_sha256",
        "subject_run_sha256",
        "answer_bindings",
    ):
        del plan[field]
    del plan["rubric"]["sha256"]
    del plan["schedule"]["expected_trials"]
    plan["sampling"]["case_units"] = [
        {"case_id": name, "unit_id": name} for name in case_ids
    ]
    plan["judge"].update(
        provider="openai",
        requested_model="openai/gpt-4o-2024-08-06",
        approved_resolved_models=["gpt-4o-2024-08-06"],
    )
    analysis = json.loads((FIXTURES / "analysis_policy.json").read_text())
    del analysis["plan_sha256"]
    analysis.update(
        minimum_units=1, maximum_interval_width="2", allowed_degradation="0"
    )
    collection = json.loads(
        (
            Path(__file__).parents[2] / "tests/judge_measurements/fixtures/export.json"
        ).read_text()
    )["collection"]
    collection["grader"] = plan["judge"]["requested_model"]
    return {
        "format": "invarlock/native-judge-policy-v1",
        "plan": plan,
        "analysis": analysis,
        "collection": collection,
        "runner": {
            "scorer_id": "factual-correctness",
            "invocation_timeout_seconds": 30,
        },
    }


def _select_judge(path):
    request = yaml.safe_load(path.read_text())
    request["comparison"].update(
        metric="judge",
        judge={"workspace": "judge-workspace", "signer_identity": "release"},
    )
    path.write_text(yaml.safe_dump(request))


def _incomplete_measurements(*, plan, **kwargs):
    if kwargs.get("status") is not None:
        kwargs["status"]["stop_reason"] = "deadline"
    digest = measurement_plan_digest(plan)
    trials = [
        {
            "trial_id": expected_trial_id(digest, binding["case_id"], side, repetition),
            "case_id": binding["case_id"],
            "side": side,
            "repetition": repetition,
            "answer_sha256": binding[f"{side}_answer_sha256"],
            "plan_sha256": digest,
            "status": "incomplete",
            "attempts": [],
            "selected_attempt": None,
            "parse": {"status": "unavailable", "rating": None, "value": None},
        }
        for binding in plan["answer_bindings"]
        for side in ("baseline", "subject")
        for repetition in range(1, plan["schedule"]["repetitions"] + 1)
    ]
    source = canonical_payload(
        {"format": "invarlock/retained-judge-json-v1", "trials": trials}
    )
    return {
        "format": "invarlock/judge-measurements-v1",
        "profile_id": plan["profile_id"],
        "plan_sha256": digest,
        "source_profile": "retained-judge-json-v1",
        "sources": [
            {
                "source_id": "offline",
                "profile": "retained-judge-json-v1",
                "encoding": "utf-8",
                "byte_size": len(source),
                "media_type": "application/json",
                "content": source.decode(),
                "sha256": hashlib.sha256(source).hexdigest(),
            }
        ],
        "trials": trials,
        "completeness": {
            "status": "incomplete",
            "expected_trials": len(trials),
            "recorded_trials": len(trials),
            "completed_trials": 0,
        },
    }


def _completed_measurements(*, plan, baseline_run, subject_run, **kwargs):
    if kwargs.get("status") is not None:
        kwargs["status"]["stop_reason"] = "complete"
    result = _incomplete_measurements(plan=plan)
    template = json.loads((FIXTURES / "measurements.json").read_bytes())["trials"][0][
        "attempts"
    ][0]
    runs = {"baseline": baseline_run, "subject": subject_run}
    for index, trial in enumerate(result["trials"]):
        row = next(
            row
            for row in runs[trial["side"]]["records"]
            if row["id"] == trial["case_id"]
        )
        request = render_judge_request(
            plan, input_text=row["input"], answer_text=row["output"]
        )
        attempt = copy.deepcopy(template)
        attempt.update(
            resolved_model=plan["judge"]["approved_resolved_models"][0],
            request={
                "text": request.decode(),
                "sha256": hashlib.sha256(request).hexdigest(),
                "media_type": "application/json",
            },
            request_id=f"request-{index}",
        )
        attempt["source"].update(
            source_id="offline", record_index=index, model_event_id=f"event-{index}"
        )
        trial.update(
            attempts=[attempt],
            status="complete",
            selected_attempt=1,
            parse={"status": "ok", "rating": "correct", "value": "1"},
        )
    source = canonical_payload(
        {"format": "invarlock/retained-judge-json-v1", "trials": result["trials"]}
    )
    result["sources"][0].update(
        content=source.decode(),
        byte_size=len(source),
        sha256=hashlib.sha256(source).hexdigest(),
    )
    result["completeness"].update(
        status="complete", completed_trials=len(result["trials"])
    )
    return result


@pytest.fixture
def collection(monkeypatch):
    preflight = Mock(return_value={"credential_available": True, "network_calls": 0})
    calls = Mock(side_effect=_completed_measurements)
    monkeypatch.setattr(native_workflow, "collection_preflight", preflight)
    monkeypatch.setattr(native_workflow, "collect_frozen", calls)
    return preflight, calls


def _import(tmp_path):
    material = _materialize_request(tmp_path, policy_document=_recipe())
    path = material["request"]
    _select_judge(path)
    key, _ = _key(tmp_path / "evidence.pem")
    return path, key, material


def test_import_judge_preflight_qualifies_units_without_numerical_policy(
    tmp_path, collection
):
    path, key, _ = _import(tmp_path)
    result = preflight_evaluation_request(path, signing_key_path=key)
    assert result.judge["independent_units"] == 2
    assert result.judge["planned_trials"] == 4
    assert result.sample_qualification is None
    assert "judge_measurements" in result.checks
    assert json.loads(result.as_json())["judge"]["collection"]["credential_available"]
    collection[1].assert_not_called()
    assert not (tmp_path / "judge-workspace").exists()
    assert not (tmp_path / "artifacts").exists()


def test_native_judge_text_preflight_describes_collection_without_calls(
    tmp_path, collection
):
    path, key, _ = _import(tmp_path)
    result = CliRunner().invoke(
        app, ["evaluate", str(path), "--signing-key", str(key), "--preflight"]
    )
    assert result.exit_code == 0, result.output
    assert "Judge: openai/gpt-4o-2024-08-06" in result.output
    assert "Independent units: 2; planned judge trials: 4" in result.output
    assert "Collection budgets:" in result.output
    collection[1].assert_not_called()


@pytest.mark.parametrize("preflight", [False, True])
def test_judge_import_rejects_attacker_paired_scores_before_collection(
    tmp_path, collection, preflight
):
    path, key, material = _import(tmp_path)
    records = json.loads(material["records"].read_bytes())
    records["records"][0]["subject"]["score"] = 0.75
    material["records"].write_bytes(canonical_json_bytes(records))
    with pytest.raises(
        (EvaluationTransactionError, EvaluationPreflightError),
        match="verifier-derived pairs",
    ):
        (preflight_evaluation_request if preflight else evaluate_request_file)(
            path, signing_key_path=key
        )
    collection[1].assert_not_called()
    assert not (tmp_path / "judge-workspace").exists()


def test_judge_environment_failure_precedes_runtime_artifact_work(
    tmp_path, monkeypatch
):
    path, key = _materialize_run_request(tmp_path)
    _select_judge(path)
    (tmp_path / "inputs/policy.json").write_bytes(
        canonical_json_bytes(_recipe(("one",)))
    )
    artifacts = Mock(side_effect=AssertionError("artifact authentication ran"))
    from invarlock.runtime_providers.hf_transformers import HFTransformersProvider

    monkeypatch.setattr(HFTransformersProvider, "authenticate_artifact", artifacts)
    monkeypatch.setattr(
        native_workflow,
        "collection_preflight",
        Mock(side_effect=ValueError("OPENAI_API_KEY must be set")),
    )
    with pytest.raises(EvaluationPreflightError, match="OPENAI_API_KEY"):
        preflight_evaluation_request(path, signing_key_path=key)
    artifacts.assert_not_called()
    assert not (tmp_path / "judge-workspace").exists()


@pytest.mark.parametrize("json_output", [False, True])
def test_native_import_judge_publishes_and_renders_judge_result(
    tmp_path, collection, json_output
):
    path, key, _ = _import(tmp_path)
    args = ["evaluate", str(path), "--signing-key", str(key)]
    if json_output:
        args.append("--json")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.stdout
    if json_output:
        payload = json.loads(result.stdout)
        assert payload["kind"] == "judge"
        assert payload["execution_mode"] == "import"
        assert payload["decision"] == "insufficient_evidence"
    else:
        assert "Bounded judge evidence created" in result.stdout
        assert "Recorded policy result: insufficient_evidence" in result.stdout
    capture = json.loads(
        (tmp_path / "judge-workspace/native_capture.json").read_bytes()
    )
    baseline, subject = validate_native_capture(capture)
    assert [row["output"] for row in baseline["records"]] == ["A", "B"]
    assert [row["output"] for row in subject["records"]] == ["A", "wrong"]
    assert capture["observations"][0]["id"] == "subject-variance"
    assert capture["normalized_request"]["comparison"]["judge"] == {
        "workspace": "judge-workspace",
        "signer_identity": "release",
    }
    collection[1].assert_called_once()


def test_native_judge_sdk_result_is_typed_and_same_capture_resumes(
    tmp_path, collection
):
    path, key, _ = _import(tmp_path)
    first = evaluate_request_file(path, signing_key_path=key)
    assert isinstance(first, JudgeWorkflowResult)
    frozen = (tmp_path / "judge-workspace/native_capture.json").read_bytes()
    request = yaml.safe_load(path.read_text())
    request["output"]["evidence"] = "artifacts/resumed"
    path.write_text(yaml.safe_dump(request))
    loaded = load_evaluation_request(path)
    second = evaluate_request_file(loaded, signing_key_path=key)
    assert isinstance(second, JudgeWorkflowResult)
    assert second.payload["decision"] == "insufficient_evidence"
    assert (tmp_path / "judge-workspace/native_capture.json").read_bytes() == frozen
    assert collection[1].call_count == 2


def test_incomplete_native_collection_retains_capture_for_same_output_resume(
    tmp_path, collection
):
    path, key, _ = _import(tmp_path)
    collection[1].side_effect = _incomplete_measurements
    with pytest.raises(JudgeWorkflowError, match="incomplete"):
        evaluate_request_file(path, signing_key_path=key)
    assert not (tmp_path / "artifacts/evidence").exists()
    frozen = (tmp_path / "judge-workspace/native_capture.json").read_bytes()
    collection[1].side_effect = _completed_measurements
    result = evaluate_request_file(path, signing_key_path=key)
    assert result.payload["decision"] == "insufficient_evidence"
    assert (tmp_path / "judge-workspace/native_capture.json").read_bytes() == frozen


@pytest.mark.parametrize("runtime_drift", [False, True])
def test_native_run_captures_once_and_checks_runtime_identity_on_resume(
    tmp_path, collection, runtime_drift
):
    path, key = _materialize_run_request(tmp_path)
    _select_judge(path)
    document = yaml.safe_load(path.read_text())
    for role in ("baseline", "subject"):
        side = document["comparison"][role]
        digest = side["runtime"]["settings"]["checkpoint_tree_sha256"]
        side["runtime"]["settings"] = {**_settings(), "checkpoint_tree_sha256": digest}
        side["artifact"]["model_id"] = f"org/{role}"
    path.write_text(yaml.safe_dump(document))
    (tmp_path / "inputs/policy.json").write_bytes(
        canonical_json_bytes(_recipe(("one",)))
    )
    images = {"baseline": "sha256:" + "1" * 64, "subject": "sha256:" + "2" * 64}

    (tmp_path / "worker").mkdir()

    class Executor:
        calls = 0

        def resolve(self, **kwargs):
            return CallerRuntimeResources(
                container_image_digest=images[kwargs["role"]]
            ).resolve(**kwargs)

        def execute(self, request, *, registry, schedule_bytes, policy_digest):
            self.calls += 1
            assert request.comparison.collection_metric == "exact_match"
            schedule = build_runtime_behavioral_schedule(json.loads(schedule_bytes))
            sides = {}
            for role in ("baseline", "subject"):
                side = getattr(request.comparison, role)
                rows = tuple(
                    RuntimeScoringRecord(
                        record_id=row.record_id,
                        input_sha256=row.input_sha256,
                        status="ok",
                        output_text="A",
                        output_sha256=hashlib.sha256(b"A").hexdigest(),
                    )
                    for row in schedule.records
                )
                written = _side_evidence(
                    tmp_path / "worker" / role,
                    role=role,
                    model_id=side.artifact.model_id,
                    schedule=schedule,
                    records=rows,
                    runtime_digest=images[role],
                    policy_digest=policy_digest,
                    checkpoint_digest=side.runtime.settings["checkpoint_tree_sha256"],
                )
                sides[role] = load_runtime_side_evidence(written.directory)
            return EvaluationRunResult(
                baseline=sides["baseline"],
                subject=sides["subject"],
                baseline_runtime_digest=images["baseline"],
                subject_runtime_digest=images["subject"],
            )

    executor = Executor()
    collection[1].side_effect = _incomplete_measurements
    with pytest.raises(JudgeWorkflowError, match="incomplete"):
        evaluate_request_file(
            path,
            signing_key_path=key,
            runtime_executor=executor,
            runtime_image_digests=images,
        )
    assert executor.calls == 1
    collection[1].side_effect = _completed_measurements
    if runtime_drift:
        images["subject"] = "sha256:" + "3" * 64
        with pytest.raises(
            JudgeWorkflowError, match="different frozen request|current preflight"
        ):
            evaluate_request_file(
                path,
                signing_key_path=key,
                runtime_executor=executor,
                runtime_image_digests=images,
            )
        assert collection[1].call_count == 1
    else:
        result = evaluate_request_file(
            path,
            signing_key_path=key,
            runtime_executor=executor,
            runtime_image_digests=images,
        )
        assert isinstance(result, JudgeWorkflowResult)
        assert result.payload["execution_mode"] == "run"
        assert collection[1].call_count == 2
    assert executor.calls == 1


def test_native_judge_evaluate_verify_report_uses_recipient_owned_policy(
    tmp_path, collection
):
    path, key, _ = _import(tmp_path)
    result = CliRunner().invoke(
        app, ["evaluate", str(path), "--signing-key", str(key), "--json"]
    )
    assert result.exit_code == 0, result.output
    evidence = tmp_path / "artifacts/evidence"
    envelope = json.loads((evidence / "envelope.json").read_bytes())
    recipient = {
        "format": "invarlock/judge-measurement-recipient-policy-v1",
        "decision_scope": envelope["decision_scope"],
        "intended_subject": envelope["intended_subject"],
        "required_metric_name": "factual-correctness",
        "trusted_signer": {
            name: envelope["signer"][name] for name in ("identity", "public_key_sha256")
        },
        "bindings": envelope["bindings"],
        "required_decision": "pass",
    }
    policy_path = tmp_path / "recipient.json"
    policy_path.write_bytes(canonical_json_bytes(recipient))
    verified = CliRunner().invoke(
        app, ["verify", str(evidence), "--trust-profile", str(policy_path), "--json"]
    )
    assert verified.exit_code == 7, verified.output
    verification = json.loads(verified.stdout)
    assert verification["kind"] == "judge"
    assert verification["replayed"] and verification["authenticated"]
    assert not verification["accepted"]
    report = CliRunner().invoke(app, ["report", str(evidence), "--json"])
    assert report.exit_code == 0, report.output
    rendered = json.loads(report.stdout)
    assert rendered["analysis"]["decision"] == "insufficient_evidence"
    assert (
        rendered["assurance"]["authentication"]
        == "signature_present_not_recipient_authorized"
    )


@pytest.mark.parametrize("destination", ["outside", "input", "root", "bad_signer"])
def test_typed_native_judge_request_rechecks_workspace_boundary(
    tmp_path, collection, destination
):
    from dataclasses import replace

    path, key, _ = _import(tmp_path)
    request = load_evaluation_request(path)
    if destination == "outside":
        judge = replace(
            request.comparison.judge, workspace=tmp_path.parent / "outside-judge"
        )
    elif destination == "input":
        judge = replace(
            request.comparison.judge, workspace=request.comparison.policy.parent
        )
    elif destination == "root":
        judge = replace(request.comparison.judge, workspace=tmp_path)
    else:
        judge = replace(request.comparison.judge, signer_identity="release\n")
    request = replace(request, comparison=replace(request.comparison, judge=judge))
    with pytest.raises(EvaluationPreflightError):
        preflight_evaluation_request(request, signing_key_path=key)
    with pytest.raises(EvaluationTransactionError):
        evaluate_request_file(request, signing_key_path=key)
    collection[1].assert_not_called()
