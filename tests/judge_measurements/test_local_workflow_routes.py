"""Local judge routes publish and replay with an explicitly synthetic provider."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.core.evaluation_request import (
    ArtifactRequest,
    ComparisonSideRequest,
    RuntimeRequest,
    load_evaluation_request,
)
from invarlock.evidence_pack_contract import canonical_json_bytes, sha256_digest
from invarlock.judge_measurements import native_workflow
from invarlock.judge_measurements.captured_workflow import (
    evaluate_captured_judge,
    preflight_captured_judge,
)
from invarlock.judge_measurements.contracts import canonical_payload
from invarlock.judge_measurements.evidence import replay_judge_evidence
from invarlock.judge_measurements.native_recipe import finalize_native_plan
from invarlock.judge_measurements.workflow import (
    JudgeWorkflowError,
    evaluate_judge_request,
    load_judge_request,
    preflight_judge_request,
)
from tests.core.test_native_judge_transaction import _recipe
from tests.judge_measurements.test_captured_scorer_workflow import (
    _request as captured_request,
)
from tests.judge_measurements.test_runtime_provider_judge import (
    _load,
    _plan,
    _Provider,
    _resources,
    _spec,
)
from tests.judge_measurements.test_runtime_provider_judge import (
    _strict_boundary as _strict_boundary,
)


def recipe(case_ids):
    value = _recipe(case_ids)
    value["plan"]["judge"] = _plan()["judge"]
    value["plan"]["schedule"]["retry_on"] = []
    value["collection"] = {
        "profile": "runtime-provider-text-frozen-answer-v1",
        "max_calls": len(case_ids) * 2,
        "max_output_tokens": len(case_ids) * 256,
    }
    value["runner"] = {"scorer_id": "judge"}
    return value


@pytest.fixture
def local(tmp_path, monkeypatch):
    provider = _Provider()
    resources = _resources(tmp_path)
    spec = _spec()
    model = ComparisonSideRequest(
        ArtifactRequest(
            resources.primary_path(), spec.model_id, "artifact://local-judge"
        ),
        RuntimeRequest(spec.provider_name, spec.settings),
    )

    def resolve(**kwargs):
        assert kwargs["role"] == "judge"
        assert kwargs["request_root"] == tmp_path
        assert kwargs["side"] == model
        return resources

    monkeypatch.setattr(
        native_workflow,
        "get_registry",
        lambda: SimpleNamespace(get_runtime_provider=lambda name: provider),
    )
    monkeypatch.setattr(
        native_workflow,
        "caller_runtime_resources_from_environment",
        lambda: SimpleNamespace(resolve=resolve),
    )
    return provider, model


def binding(model, root):
    return {
        "artifact": {
            "path": model.artifact.path.relative_to(root).as_posix(),
            "model_id": model.artifact.model_id,
            "locator": model.artifact.locator,
        },
        "runtime": {
            "provider": model.runtime.provider,
            "settings": dict(model.runtime.settings),
        },
    }


def captured(tmp_path, model):
    path, value, _, runs = captured_request(tmp_path)
    policy = recipe([row["id"] for row in runs[0]["records"]])
    (tmp_path / "policy.json").write_bytes(canonical_payload(policy))
    value["comparison"]["judge"]["model"] = binding(model, tmp_path)
    path.write_text(yaml.safe_dump(value))
    return load_evaluation_request(path), policy, runs


def test_captured_local_preflight_collection_and_evidence_replay(tmp_path, local):
    provider, model = local
    request, _, _ = captured(tmp_path, model)
    preflight = preflight_captured_judge(
        request, signing_key_path=None, unsigned=True
    ).payload
    assert preflight["collection_environment"]["model_loaded"] is False
    assert preflight["maximum_admitted_calls"] == 4
    assert provider.score_calls == 0
    result = evaluate_captured_judge(
        request, signing_key_path=None, unsigned=True
    ).payload
    assert result["ok"]
    assert provider.score_calls == 4
    assert (
        replay_judge_evidence(request.evidence).analysis_result.to_dict()["decision"]
        == "insufficient_evidence"
    )
    measured = json.loads((request.evidence / "measurements.json").read_bytes())
    assert measured["completeness"]["completed_trials"] == 4
    assert measured["source_profile"] == "retained-runtime-provider-judge-v1"


def test_v3_local_collection_publishes_replayable_frozen_answers(tmp_path, local):
    provider, model = local
    _, policy, runs = captured(tmp_path, model)
    plan, analysis = finalize_native_plan(policy, *runs)
    for name, value in (
        ("plan", plan),
        ("analysis", analysis),
        ("collection", policy["collection"]),
    ):
        (tmp_path / (name + ".json")).write_bytes(canonical_payload(value))
    value = {
        "format_version": "invarlock/evaluation-request-v3",
        "execution": {
            "mode": "judge_collect",
            "collection": {
                "integration": "runtime-provider-judge",
                "configuration": "collection.json",
                "model": binding(model, tmp_path),
            },
        },
        "comparison": {
            "baseline_run": "baseline.json",
            "subject_run": "subject.json",
            "plan": "plan.json",
            "measurements": None,
            "policy": "analysis.json",
        },
        "output": {"evidence": "v3-evidence", "signer_identity": "test-local"},
    }
    path = tmp_path / "v3.json"
    path.write_bytes(canonical_payload(value))
    request = load_judge_request(path)
    assert preflight_judge_request(request).payload["ready"]
    assert provider.score_calls == 0
    result = evaluate_judge_request(request, signing_key=None, unsigned=True).payload
    assert result["ok"] and provider.score_calls == 4
    assert (
        replay_judge_evidence(request.evidence).analysis_result.to_dict()["decision"]
        == "insufficient_evidence"
    )


def native(tmp_path, model):
    from invarlock.evaluation_transaction import _normalized_side
    from tests.evidence_packs.test_evidence_pack import (
        _request,
        _schedule,
        _side_evidence,
    )

    schedule = _schedule()
    policy = recipe([row.record_id for row in schedule.records])
    raw = canonical_json_bytes(policy)
    sides = [
        _side_evidence(
            tmp_path / side,
            schedule=schedule,
            image_digest="sha256:" + image * 64,
            artifact_marker=artifact,
            outputs=("A", "B"),
            role=side,
            policy_digest=sha256_digest(raw),
        )
        for side, image, artifact in (("baseline", "a", "c"), ("subject", "b", "d"))
    ]
    normalized = _request("judge")
    normalized["comparison"]["judge"] = {
        "workspace": "judge-work",
        "signer_identity": "local-test",
        "model": _normalized_side(model),
    }
    request = SimpleNamespace(
        root=tmp_path,
        comparison=SimpleNamespace(
            judge=SimpleNamespace(
                model=model,
                workspace=tmp_path / "judge-work",
                signer_identity="local-test",
            )
        ),
        execution=SimpleNamespace(mode="import"),
        output=SimpleNamespace(evidence=tmp_path / "native-evidence"),
    )
    return (
        request,
        {
            "request": request,
            "normalized_request": normalized,
            "schedule": schedule.to_payload(),
            "policy_bytes": raw,
            "signing_key": Ed25519PrivateKey.generate(),
            "expected_artifact_digests": {
                role: sha256_digest(side.artifact_identity)
                for role, side in zip(("baseline", "subject"), sides, strict=True)
            },
            "expected_runtime_digests": {
                "baseline": "sha256:" + "a" * 64,
                "subject": "sha256:" + "b" * 64,
            },
        },
        sides,
    )


def test_native_local_route_retains_runtime_capture_and_signed_judge_evidence(
    tmp_path, local
):
    provider, model = local
    request, kwargs, sides = native(tmp_path, model)
    result = native_workflow.preflight_native_judge(
        request, kwargs["schedule"], kwargs["policy_bytes"]
    )
    assert result["collection"]["model_loaded"] is False
    assert provider.score_calls == 0
    result = native_workflow.evaluate_native_judge(
        **kwargs, capture=lambda: tuple(sides)
    ).payload
    assert result["ok"] and provider.score_calls == 4
    assert (request.comparison.judge.workspace / "native_capture.json").is_file()
    assert (
        replay_judge_evidence(request.output.evidence).analysis_result.to_dict()[
            "decision"
        ]
        == "insufficient_evidence"
    )


@pytest.mark.parametrize("missing", ["model", "request_root", "plan"])
def test_local_preflight_refuses_missing_binding_before_provider_execution(
    tmp_path, local, missing
):
    provider, model = local
    kwargs = {"model": model, "request_root": tmp_path, "plan": _plan()}
    kwargs.pop(missing)
    with pytest.raises(JudgeWorkflowError, match="requires its model"):
        native_workflow.collection_preflight(recipe(["case-1"])["collection"], **kwargs)
    assert provider.open_calls == provider.score_calls == 0


@pytest.mark.parametrize("fault", ["missing_model", "missing_root", "ignored_timeout"])
def test_local_collection_refuses_incomplete_or_ignored_execution_binding(
    tmp_path, local, fault
):
    provider, model = local
    kwargs = {
        "plan": _plan(),
        "collection": recipe(["case-1"])["collection"],
        "runner": {"scorer_id": "judge"},
        "workspace": tmp_path / "work",
        "baseline_run": {},
        "subject_run": {},
        "model": model,
        "request_root": tmp_path,
    }
    if fault == "missing_model":
        kwargs.pop("model")
    elif fault == "missing_root":
        kwargs.pop("request_root")
    else:
        kwargs["runner"]["invocation_timeout_seconds"] = 1
    with pytest.raises(JudgeWorkflowError, match="requires its model|only scorer_id"):
        native_workflow.collect_frozen(**kwargs)
    assert provider.score_calls == 0


def test_captured_programmatic_import_cannot_ignore_a_local_model(tmp_path, local):
    provider, model = local
    request, _, _ = captured(tmp_path, model)
    request = replace(
        request,
        judge=replace(request.judge, measurements=tmp_path / "measurements.json"),
    )
    with pytest.raises(JudgeWorkflowError, match="cannot combine"):
        preflight_captured_judge(request, signing_key_path=None, unsigned=True)
    assert provider.score_calls == 0


def test_local_collection_can_complete_without_a_status_observer(tmp_path, local):
    provider, model = local
    measurements = native_workflow.collect_frozen(
        plan=_plan(),
        collection=recipe(["case-1"])["collection"],
        runner={"scorer_id": "judge"},
        workspace=tmp_path / "work",
        baseline_run=_load("baseline_run.json"),
        subject_run=_load("subject_run.json"),
        model=model,
        request_root=tmp_path,
    )
    assert measurements["completeness"]["status"] == "complete"
    assert provider.score_calls == 2
