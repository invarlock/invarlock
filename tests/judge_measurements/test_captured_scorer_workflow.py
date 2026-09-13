"""Existing evaluator exports enter the installed bounded judge workflow."""

import json

import pytest
import yaml
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.core.evaluation_request import load_evaluation_request
from invarlock.engine import evaluate_request_file, preflight_evaluation_request
from invarlock.evaluation_comparison.comparison import make_run
from invarlock.judge_measurements import native_workflow
from invarlock.judge_measurements.native_recipe import finalize_native_plan
from tests.core.test_native_judge_transaction import _incomplete_measurements, _recipe


def _request(tmp_path):
    records = [
        {"id": case, "input": f"Task {case}", "expected": answer, "output": answer}
        for case, answer in (("one", "A"), ("two", "B"))
    ]
    runs = []
    for side, marker in (("baseline", "a"), ("subject", "b")):
        run = make_run(
            records,
            source={"name": "existing-evaluator", "version": "1"},
            run_id=side,
            artifact_digest="sha256:" + marker * 64,
        )
        (tmp_path / f"{side}.json").write_text(json.dumps(run))
        runs.append(run)
    recipe = _recipe()
    (tmp_path / "policy.json").write_text(json.dumps(recipe))
    value = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {"path": "baseline.json", "adapter": "invarlock"},
            "subject": {"path": "subject.json", "adapter": "invarlock"},
            "metric": "judge",
            "policy": "policy.json",
            "judge": {"workspace": "judge-work", "signer_identity": "reviewed-signer"},
        },
        "output": {"evidence": "evidence"},
    }
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(value))
    return path, value, recipe, runs


def test_captured_judge_preflight_never_collects_or_writes(tmp_path, monkeypatch):
    path, _, _, _ = _request(tmp_path)
    monkeypatch.setattr(native_workflow, "collection_preflight", lambda _: {})
    monkeypatch.setattr(
        native_workflow, "collect_frozen", lambda **_: pytest.fail("called")
    )
    result = preflight_evaluation_request(
        load_evaluation_request(path), signing_key_path=None, unsigned=True
    )
    payload = json.loads(result.as_json())
    assert payload["kind"] == "judge"
    assert payload["planned_trials"] == 4
    assert payload["network_calls"] == 0
    assert not (tmp_path / "judge-work").exists()
    assert not (tmp_path / "evidence").exists()


def test_captured_import_judgments_uses_same_offline_cli_and_report(tmp_path):
    path, value, recipe, runs = _request(tmp_path)
    plan, _ = finalize_native_plan(recipe, *runs)
    measurements = _incomplete_measurements(plan=plan)
    (tmp_path / "measurements.json").write_text(json.dumps(measurements))
    value["comparison"]["judge"]["measurements"] = "measurements.json"
    path.write_text(yaml.safe_dump(value))
    result = CliRunner().invoke(app, ["evaluate", str(path), "--unsigned", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["decision"] == "insufficient_evidence"
    report = CliRunner().invoke(
        app,
        ["report", str(tmp_path / "evidence"), "--html", str(tmp_path / "report.html")],
    )
    assert report.exit_code == 0, report.output
    assert "existing-evaluator" in (tmp_path / "report.html").read_text()


@pytest.mark.parametrize(
    "change", ["workspace-input", "workspace-output", "run-pin", "missing-judge"]
)
def test_captured_judge_rejects_bad_intent_before_collection(
    tmp_path, monkeypatch, change
):
    path, value, _, _ = _request(tmp_path)
    if change == "workspace-input":
        value["comparison"]["judge"]["workspace"] = "baseline.json"
    elif change == "workspace-output":
        value["comparison"]["judge"]["workspace"] = "evidence/inner"
    elif change == "run-pin":
        value["comparison"]["baseline"]["expected_run_digest"] = "sha256:" + "f" * 64
    else:
        del value["comparison"]["judge"]
    path.write_text(yaml.safe_dump(value))
    monkeypatch.setattr(
        native_workflow, "collection_preflight", lambda _: pytest.fail("called")
    )
    with pytest.raises(ValueError):
        evaluate_request_file(
            load_evaluation_request(path), signing_key_path=None, unsigned=True
        )
    assert not (tmp_path / "evidence").exists()


def _loaded(tmp_path):
    path, _, _, _ = _request(tmp_path)
    return load_evaluation_request(path)


@pytest.fixture
def collector(monkeypatch):
    from unittest.mock import Mock

    from tests.core.test_native_judge_transaction import _completed_measurements

    preflight = Mock(return_value={"credential_available": True, "network_calls": 0})
    collect = Mock(side_effect=_completed_measurements)
    monkeypatch.setattr(native_workflow, "collection_preflight", preflight)
    monkeypatch.setattr(native_workflow, "collect_frozen", collect)
    return preflight, collect


@pytest.mark.parametrize("signed", [False, True])
def test_collection_freezes_exact_captured_answers_and_publishes_once(
    tmp_path, collector, signed
):
    from invarlock.judge_measurements.captured_workflow import evaluate_captured_judge
    from invarlock.judge_measurements.evidence import replay_judge_evidence
    from tests.core.test_captured_evaluation import _key

    request = _loaded(tmp_path)
    key = _key(tmp_path) if signed else None
    result = evaluate_captured_judge(request, signing_key_path=key, unsigned=not signed)
    payload = json.loads(result.as_json())
    assert payload["decision"] == "insufficient_evidence"
    assert payload["authentication"] == ("signed" if signed else "unsigned_local")
    assert payload["collection"] == {
        "pending_trials": 0,
        "stop_reason": "complete",
        "resumable": False,
    }
    assert payload["source_assurance"] == "captured_inputs"
    assert payload["independent_verification"] == "not_performed"
    collector[0].assert_called_once()
    collector[1].assert_called_once()
    arguments = collector[1].call_args.kwargs
    assert arguments["baseline_run"]["source"]["name"] == "existing-evaluator"
    assert [row["output"] for row in arguments["subject_run"]["records"]] == ["A", "B"]
    assert (request.judge.workspace / "identity.json").is_file()
    replayed = replay_judge_evidence(request.evidence)
    assert replayed.analysis_result.to_dict()["decision"] == "insufficient_evidence"
    with pytest.raises(ValueError):
        evaluate_captured_judge(request, signing_key_path=key, unsigned=not signed)
    assert collector[1].call_count == 1


def test_signed_preflight_authenticates_key_without_writing(tmp_path, collector):
    from invarlock.judge_measurements.captured_workflow import preflight_captured_judge
    from tests.core.test_captured_evaluation import _key

    request = _loaded(tmp_path)
    result = preflight_captured_judge(
        request, signing_key_path=_key(tmp_path), unsigned=False
    )
    payload = json.loads(result.as_json())
    assert payload["requested_authentication"] == "signed"
    assert payload["collection_available"] is True
    assert payload["maximum_admitted_calls"] >= payload["planned_trials"]
    assert payload["network_calls"] == 0
    collector[1].assert_not_called()
    assert not request.judge.workspace.exists()
    assert not request.evidence.exists()


@pytest.mark.parametrize("entry", ["preflight", "evaluate"])
@pytest.mark.parametrize(
    "unsigned,key_present", [(False, False), (True, True), (1, False)]
)
def test_ambiguous_authentication_fails_before_environment_checks(
    tmp_path, collector, entry, unsigned, key_present
):
    from invarlock.judge_measurements import captured_workflow

    request = _loaded(tmp_path)
    action = getattr(captured_workflow, entry + "_captured_judge")
    with pytest.raises(captured_workflow.JudgeWorkflowError, match="Choose either"):
        action(
            request,
            signing_key_path=tmp_path / "key.pem" if key_present else None,
            unsigned=unsigned,
        )
    collector[0].assert_not_called()
    collector[1].assert_not_called()


@pytest.mark.parametrize(
    "fault",
    [
        "metric",
        "judge",
        "identity",
        "identity-type",
        "outside-root",
        "workspace-evidence",
        "evidence-workspace",
        "workspace-file",
        "workspace-symlink",
        "ancestor-symlink",
        "ancestor-file",
        "readable-workspace",
        "occupied-evidence",
        "signing-key-overlap",
        "measurements-overlap",
    ],
)
def test_unsafe_locations_fail_before_collection_preflight(tmp_path, collector, fault):
    from dataclasses import replace

    from invarlock.judge_measurements.captured_workflow import (
        JudgeWorkflowError,
        preflight_captured_judge,
    )

    request = _loaded(tmp_path)
    judge = request.judge
    signing_key = None
    if fault == "metric":
        request = replace(request, metric="exact_match")
    elif fault == "judge":
        request = replace(request, judge=None)
    elif fault == "identity":
        request = replace(
            request, judge=replace(judge, signer_identity="bad\nidentity")
        )
    elif fault == "identity-type":
        request = replace(request, judge=replace(judge, signer_identity=5))
    elif fault == "outside-root":
        request = replace(
            request, judge=replace(judge, workspace=tmp_path.parent / "outside")
        )
    elif fault == "workspace-evidence":
        request = replace(
            request, judge=replace(judge, workspace=request.evidence / "child")
        )
    elif fault == "evidence-workspace":
        request = replace(request, evidence=judge.workspace / "child")
    elif fault == "workspace-file":
        judge.workspace.write_text("occupied")
    elif fault == "workspace-symlink":
        (tmp_path / "real").mkdir(mode=0o700)
        judge.workspace.symlink_to(tmp_path / "real", target_is_directory=True)
    elif fault == "ancestor-symlink":
        (tmp_path / "real").mkdir(mode=0o700)
        (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
        request = replace(
            request, judge=replace(judge, workspace=tmp_path / "alias" / "child")
        )
    elif fault == "ancestor-file":
        (tmp_path / "parent-file").write_text("occupied")
        request = replace(
            request, judge=replace(judge, workspace=tmp_path / "parent-file" / "child")
        )
    elif fault == "readable-workspace":
        judge.workspace.mkdir(mode=0o755)
    elif fault == "occupied-evidence":
        request.evidence.mkdir()
    elif fault == "signing-key-overlap":
        signing_key = judge.workspace / "key.pem"
    else:
        request = replace(
            request, judge=replace(judge, measurements=judge.workspace / "input.json")
        )
    with pytest.raises(JudgeWorkflowError):
        preflight_captured_judge(
            request, signing_key_path=signing_key, unsigned=signing_key is None
        )
    collector[0].assert_not_called()
    collector[1].assert_not_called()


@pytest.mark.parametrize(
    "fault",
    [
        "max_calls",
        "max_input_tokens",
        "max_output_tokens",
        "max_cost_microusd",
        "minimum_units",
    ],
)
def test_budget_and_independent_unit_checks_precede_environment_checks(
    tmp_path, collector, fault
):
    from invarlock.judge_measurements.captured_workflow import (
        JudgeWorkflowError,
        preflight_captured_judge,
    )

    request = _loaded(tmp_path)
    recipe = json.loads(request.policy.read_text())
    if fault == "minimum_units":
        recipe["analysis"][fault] = 3
    else:
        recipe["collection"][fault] = 1
    request.policy.write_text(json.dumps(recipe))
    with pytest.raises(
        JudgeWorkflowError, match="reserve every planned call|fewer independent units"
    ):
        preflight_captured_judge(request, signing_key_path=None, unsigned=True)
    collector[0].assert_not_called()
    collector[1].assert_not_called()


def test_native_claim_cannot_skip_native_runtime_evidence(tmp_path, collector):
    from invarlock.judge_measurements.captured_workflow import (
        JudgeWorkflowError,
        prepare_evaluator_judge,
    )
    from invarlock.judge_measurements.native_capture import NATIVE_RUN_SOURCE

    _, _, recipe, runs = _request(tmp_path)
    runs[0]["source"]["name"] = NATIVE_RUN_SOURCE
    with pytest.raises(JudgeWorkflowError, match="runtime capture"):
        prepare_evaluator_judge(recipe, *runs)
    collector[0].assert_not_called()


def test_imported_completed_measurements_are_offline_in_preflight_and_execution(
    tmp_path, collector
):
    from dataclasses import replace

    from invarlock.judge_measurements.captured_workflow import (
        evaluate_captured_judge,
        preflight_captured_judge,
    )
    from tests.core.test_native_judge_transaction import _completed_measurements

    path, _, recipe, runs = _request(tmp_path)
    request = load_evaluation_request(path)
    plan, _ = finalize_native_plan(recipe, *runs)
    measurements = _completed_measurements(
        plan=plan, baseline_run=runs[0], subject_run=runs[1]
    )
    measurement_path = tmp_path / "measurements.json"
    measurement_path.write_text(json.dumps(measurements))
    request = replace(
        request, judge=replace(request.judge, measurements=measurement_path)
    )
    preflight = json.loads(
        preflight_captured_judge(
            request, signing_key_path=None, unsigned=True
        ).as_json()
    )
    assert preflight["collection_available"] is False
    assert preflight["measurements"]["status"] == "complete"
    result = json.loads(
        evaluate_captured_judge(request, signing_key_path=None, unsigned=True).as_json()
    )
    assert result["decision"] == "insufficient_evidence"
    assert "collection" not in result
    assert not request.judge.workspace.exists()
    collector[0].assert_not_called()
    collector[1].assert_not_called()


def test_existing_private_workspace_is_ready_without_preflight_writes(
    tmp_path, collector
):
    from invarlock.judge_measurements.captured_workflow import preflight_captured_judge

    request = _loaded(tmp_path)
    request.judge.workspace.mkdir(mode=0o700)
    preflight_captured_judge(request, signing_key_path=None, unsigned=True)
    assert list(request.judge.workspace.iterdir()) == []
    collector[1].assert_not_called()


def test_workspace_owned_by_someone_else_fails_preflight(
    tmp_path, collector, monkeypatch
):
    import os

    from invarlock.judge_measurements import captured_workflow

    request = _loaded(tmp_path)
    request.judge.workspace.mkdir(mode=0o700)
    original_uid = os.geteuid()
    monkeypatch.setattr(captured_workflow.os, "geteuid", lambda: original_uid + 1)
    with pytest.raises(captured_workflow.JudgeWorkflowError, match="caller-owned"):
        captured_workflow.preflight_captured_judge(
            request, signing_key_path=None, unsigned=True
        )
    collector[0].assert_not_called()


@pytest.mark.parametrize(
    "fault",
    [
        "missing-key",
        "invalid-key",
        "missing-input",
        "malformed-input",
        "invalid-measurements",
    ],
)
@pytest.mark.parametrize("entry", ["preflight", "evaluate"])
def test_bad_retained_inputs_and_keys_never_reach_collection(
    tmp_path, collector, fault, entry
):
    from dataclasses import replace

    from invarlock.judge_measurements import captured_workflow

    request = _loaded(tmp_path)
    key = tmp_path / "key.pem" if fault.endswith("key") else None
    if fault == "invalid-key":
        key.write_text("not a private key")
    elif fault == "missing-input":
        request.baseline.path.unlink()
    elif fault == "malformed-input":
        request.baseline.path.write_text("not JSON")
    elif fault == "invalid-measurements":
        measurements = tmp_path / "measurements.json"
        measurements.write_text("{}")
        request = replace(
            request, judge=replace(request.judge, measurements=measurements)
        )
    with pytest.raises(captured_workflow.JudgeWorkflowError):
        getattr(captured_workflow, entry + "_captured_judge")(
            request, signing_key_path=key, unsigned=key is None
        )
    collector[0].assert_not_called()
    collector[1].assert_not_called()
    assert not request.evidence.exists()
    assert not request.judge.workspace.exists()


def test_subject_run_pin_is_checked_before_environment(tmp_path, collector):
    from dataclasses import replace

    from invarlock.judge_measurements.captured_workflow import (
        JudgeWorkflowError,
        preflight_captured_judge,
    )

    request = _loaded(tmp_path)
    request = replace(
        request,
        subject=replace(request.subject, expected_run_digest="sha256:" + "f" * 64),
    )
    with pytest.raises(JudgeWorkflowError, match="subject run digest"):
        preflight_captured_judge(request, signing_key_path=None, unsigned=True)
    collector[0].assert_not_called()


def test_deadline_retains_same_frozen_workspace_and_can_resume(tmp_path, collector):
    from invarlock.judge_measurements.captured_workflow import (
        JudgeWorkflowError,
        evaluate_captured_judge,
    )
    from tests.core.test_native_judge_transaction import _completed_measurements

    request = _loaded(tmp_path)
    collector[1].side_effect = _incomplete_measurements
    with pytest.raises(JudgeWorkflowError, match="rerun the same request") as stopped:
        evaluate_captured_judge(request, signing_key_path=None, unsigned=True)
    assert stopped.value.payload["resumable"] is True
    assert stopped.value.payload["pending_trials"] == 4
    assert not request.evidence.exists()
    identity = (request.judge.workspace / "identity.json").read_bytes()
    collector[1].side_effect = _completed_measurements
    result = evaluate_captured_judge(request, signing_key_path=None, unsigned=True)
    assert json.loads(result.as_json())["collection"]["pending_trials"] == 0
    assert (request.judge.workspace / "identity.json").read_bytes() == identity
    assert collector[1].call_count == 2


def test_resume_rejects_changed_frozen_answers_without_new_calls(tmp_path, collector):
    from invarlock.judge_measurements.captured_workflow import (
        JudgeWorkflowError,
        evaluate_captured_judge,
    )

    request = _loaded(tmp_path)
    collector[1].side_effect = _incomplete_measurements
    with pytest.raises(JudgeWorkflowError, match="rerun the same request"):
        evaluate_captured_judge(request, signing_key_path=None, unsigned=True)
    run = json.loads(request.subject.path.read_text())
    run["records"][0]["output"] = "replacement answer"
    request.subject.path.write_text(json.dumps(run))
    with pytest.raises(JudgeWorkflowError, match="different frozen request"):
        evaluate_captured_judge(request, signing_key_path=None, unsigned=True)
    assert collector[1].call_count == 1
    assert not request.evidence.exists()


def test_retained_capacity_exhaustion_publishes_honest_incomplete_evidence(
    tmp_path, collector
):
    from invarlock.judge_measurements.captured_workflow import evaluate_captured_judge

    request = _loaded(tmp_path)

    def exhausted(**kwargs):
        result = _incomplete_measurements(**kwargs)
        kwargs["status"]["stop_reason"] = "capacity_exhausted"
        return result

    collector[1].side_effect = exhausted
    result = json.loads(
        evaluate_captured_judge(request, signing_key_path=None, unsigned=True).as_json()
    )
    assert result["decision"] == "insufficient_evidence"
    assert result["collection"] == {
        "pending_trials": 4,
        "stop_reason": "retained_capacity_exhausted",
        "resumable": False,
    }
    assert request.evidence.is_dir()


def test_workspace_substitution_during_collection_cannot_publish(tmp_path, collector):
    from invarlock.judge_measurements.captured_workflow import (
        JudgeWorkflowError,
        evaluate_captured_judge,
    )
    from tests.core.test_native_judge_transaction import _completed_measurements

    request = _loaded(tmp_path)

    def substitute(**kwargs):
        result = _completed_measurements(**kwargs)
        request.judge.workspace.rename(tmp_path / "detached")
        request.judge.workspace.mkdir(mode=0o700)
        return result

    collector[1].side_effect = substitute
    with pytest.raises(JudgeWorkflowError, match="ancestry changed"):
        evaluate_captured_judge(request, signing_key_path=None, unsigned=True)
    assert not request.evidence.exists()


def test_publication_io_failure_is_a_workflow_error_and_retains_collection(
    tmp_path, collector, monkeypatch
):
    from invarlock.judge_measurements import captured_workflow

    request = _loaded(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("cannot publish evidence")

    monkeypatch.setattr(captured_workflow, "publish_judge_evidence", fail)
    with pytest.raises(
        captured_workflow.JudgeWorkflowError, match="cannot publish evidence"
    ):
        captured_workflow.evaluate_captured_judge(
            request, signing_key_path=None, unsigned=True
        )
    assert (request.judge.workspace / "identity.json").exists()
    assert not request.evidence.exists()
