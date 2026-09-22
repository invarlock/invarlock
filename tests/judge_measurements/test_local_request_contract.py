"""Local judge request and dispatch contracts belong to the runtime test gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.judge_measurements.workflow import JudgeWorkflowError, load_judge_request
from tests.core.test_evaluation_request_contract import _write_request
from tests.core.test_native_judge_request import collected


def test_v3_runtime_collection_retains_parsed_model_and_integration(tmp_path):
    value = collected(tmp_path)
    request = load_judge_request(_write_request(tmp_path / "request.yaml", value))
    assert request.integration == "runtime-provider-judge"
    assert request.model.artifact.path == tmp_path / "judge-model"
    assert dict(request.runner) == {"scorer_id": "judge"}
    assert request.inputs["collection"] == tmp_path / "collection.json"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_model",
        "hosted_model",
        "missing_path",
        "unknown_model_field",
        "bool_seed",
        "workspace_overlap",
        "evidence_overlap",
        "unknown_integration",
    ],
)
def test_v3_collection_model_is_closed_and_required_only_for_local(tmp_path, mutation):
    value = collected(tmp_path)
    collection = value["execution"]["collection"]
    if mutation == "missing_model":
        collection.pop("model")
    elif mutation == "hosted_model":
        collection["integration"] = "inspect-judge"
    elif mutation == "missing_path":
        collection["model"]["artifact"].pop("path")
    elif mutation == "unknown_model_field":
        collection["model"]["extra"] = "ignored"
    elif mutation == "bool_seed":
        collection["model"]["runtime"]["settings"]["seed"] = True
    elif mutation == "workspace_overlap":
        collection["workspace"] = "judge-model/work"
    elif mutation == "evidence_overlap":
        value["output"]["evidence"] = "judge-model/evidence"
    else:
        collection["integration"] = "arbitrary-plugin"
    with pytest.raises(JudgeWorkflowError):
        load_judge_request(_write_request(tmp_path / "request.yaml", value))


def test_hosted_v3_keeps_legacy_request_shape(tmp_path):
    value = collected(tmp_path)
    collection = value["execution"]["collection"]
    collection.pop("model")
    collection["integration"] = "inspect-judge"
    request = load_judge_request(_write_request(tmp_path / "request.yaml", value))
    assert request.integration == "inspect-judge"
    assert request.model is None


def staged_local_collection(tmp_path):
    import shutil

    value = collected(tmp_path)
    fixtures = Path(__file__).parents[1] / "fixtures/judge_measurements"
    for target, source in (
        ("baseline.json", "baseline_run.json"),
        ("subject.json", "subject_run.json"),
        ("plan.json", "plan.json"),
        ("policy.json", "analysis_policy.json"),
    ):
        shutil.copyfile(fixtures / source, tmp_path / target)
    (tmp_path / "collection.json").write_text(
        json.dumps(
            {
                "profile": "runtime-provider-text-frozen-answer-v1",
                "max_calls": 2,
                "max_output_tokens": 256,
            }
        )
    )
    return load_judge_request(_write_request(tmp_path / "request.yaml", value))


def test_v3_local_preflight_uses_local_budget_and_explicit_binding(
    tmp_path, monkeypatch
):
    from invarlock.judge_measurements import native_workflow, workflow

    request = staged_local_collection(tmp_path)
    checked = []

    def preflight(configuration, **kwargs):
        assert kwargs["integration"] == "runtime-provider-judge"
        assert kwargs["model"] == request.model
        assert kwargs["request_root"] == tmp_path
        assert kwargs["plan"]["schedule"]["expected_trials"] == 2
        checked.append(configuration)
        return {"model_loaded": False}

    monkeypatch.setattr(native_workflow, "collection_preflight", preflight)
    result = workflow.preflight_judge_request(request).payload
    assert result["ready"] and result["collection_available"]
    assert result["budget_capacity"]["maximum_admitted_calls"] == 2
    assert result["collection_integration"]["api"].endswith("collect_runtime_provider")
    assert len(checked) == 1
    assert not request.workspace.exists()


def test_v3_local_collection_identity_and_dispatch_retain_model(tmp_path, monkeypatch):
    from invarlock.evaluation_transaction import _normalized_side
    from invarlock.judge_measurements import native_workflow, workflow

    request = staged_local_collection(tmp_path)
    monkeypatch.setattr(
        native_workflow, "collection_preflight", lambda *args, **kwargs: {}
    )

    def collected_call(**kwargs):
        from invarlock.judge_measurements.evidence import object_sha256

        assert kwargs["integration"] == "runtime-provider-judge"
        assert kwargs["model"] == request.model
        assert kwargs["request_root"] == tmp_path
        identity = json.loads((request.workspace / "identity.json").read_bytes())
        expected = {
            "format": "invarlock/judge-collection-workspace-v1",
            "inputs": {
                key: object_sha256(json.loads(path.read_bytes()))
                for key, path in request.inputs.items()
            },
            "runner": dict(request.runner),
            "integration": "runtime-provider-judge",
            "model": _normalized_side(request.model),
        }
        assert identity["sha256"] == object_sha256(expected)
        raise RuntimeError("test boundary reached before inference")

    monkeypatch.setattr(native_workflow, "collect_frozen", collected_call)
    with pytest.raises(RuntimeError, match="test boundary reached"):
        workflow.evaluate_judge_request(request, signing_key=None, unsigned=True)
    assert not request.evidence.exists()


def test_v3_local_rejects_hosted_only_timeout_setting(tmp_path):
    value = collected(tmp_path)
    value["execution"]["collection"]["invocation_timeout_seconds"] = 60
    with pytest.raises(JudgeWorkflowError, match="judge request is invalid"):
        load_judge_request(_write_request(tmp_path / "request.yaml", value))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_model",
        "hosted_model",
        "import_model",
        "ignored_timeout",
        "runner_override",
        "retained_measurements",
        "escaped_input",
        "escaped_workspace",
        "artifact_overlap",
        "invalid_runner_type",
        "invalid_inputs_type",
        "invalid_path_type",
        "input_in_workspace",
        "missing_workspace",
        "missing_configuration",
        "import_configuration",
        "input_in_evidence",
    ],
)
def test_programmatic_v3_cannot_bypass_closed_request_contract(tmp_path, mutation):
    from dataclasses import replace

    from invarlock.judge_measurements.workflow import preflight_judge_request

    request = staged_local_collection(tmp_path)
    if mutation == "missing_model":
        request = replace(request, model=None)
    elif mutation == "hosted_model":
        request = replace(request, integration="inspect-judge")
    elif mutation == "import_model":
        request = replace(request, mode="judge_import")
    elif mutation == "ignored_timeout":
        request = replace(
            request, runner={"scorer_id": "judge", "invocation_timeout_seconds": 60}
        )
    elif mutation == "runner_override":
        request = replace(request, runner={"scorer_id": "judge", "model": "ignored"})
    elif mutation == "retained_measurements":
        request = replace(
            request,
            inputs=dict(request.inputs) | {"measurements": tmp_path / "retained.json"},
        )
    elif mutation == "escaped_input":
        request = replace(
            request,
            inputs=dict(request.inputs)
            | {"baseline_run": tmp_path.parent / "outside.json"},
        )
    elif mutation == "escaped_workspace":
        request = replace(request, workspace=tmp_path.parent / "work")
    elif mutation == "invalid_runner_type":
        request = replace(request, runner=True)
    elif mutation == "invalid_inputs_type":
        request = replace(request, inputs=True)
    elif mutation == "invalid_path_type":
        request = replace(
            request, inputs=dict(request.inputs) | {"baseline_run": "not-a-Path"}
        )
    elif mutation == "missing_workspace":
        request = replace(request, workspace=None)
    elif mutation == "missing_configuration":
        request = replace(
            request,
            inputs={
                key: value
                for key, value in request.inputs.items()
                if key != "collection"
            },
        )
    elif mutation == "import_configuration":
        request = replace(
            request,
            mode="judge_import",
            model=None,
            integration=None,
            workspace=None,
            runner=None,
        )
    elif mutation == "input_in_evidence":
        request = replace(
            request,
            inputs=dict(request.inputs)
            | {"baseline_run": request.evidence / "baseline.json"},
        )
    elif mutation == "input_in_workspace":
        request = replace(
            request,
            inputs=dict(request.inputs)
            | {"baseline_run": request.workspace / "baseline.json"},
        )
    else:
        request = replace(request, evidence=request.model.artifact.path / "evidence")
    with pytest.raises(JudgeWorkflowError):
        preflight_judge_request(request)


def test_programmatic_local_runner_requires_its_actual_scorer_binding(tmp_path):
    from dataclasses import replace

    from invarlock.judge_measurements.workflow import preflight_judge_request

    request = staged_local_collection(tmp_path)
    with pytest.raises(JudgeWorkflowError, match="ignored or misplaced"):
        preflight_judge_request(replace(request, runner={}))
