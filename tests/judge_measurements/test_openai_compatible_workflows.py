"""Explicit endpoint bindings traverse each supported judge workflow."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import MappingProxyType

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.core.evaluation_request import load_evaluation_request
from invarlock.evaluation_comparison.comparison import make_run
from invarlock.judge_measurements import native_workflow
from invarlock.judge_measurements.captured_workflow import preflight_captured_judge
from invarlock.judge_measurements.contracts import canonical_payload
from invarlock.judge_measurements.native_recipe import finalize_native_plan
from invarlock.judge_measurements.workflow import (
    JudgeWorkflowError,
    load_judge_request,
    preflight_judge_request,
)
from tests.core.test_native_judge_request import collected, model
from tests.core.test_native_judge_transaction import _recipe
from tests.judge_measurements.test_captured_scorer_workflow import _request


def configuration(max_calls=4):
    return {
        "profile": "openai-compatible-text-frozen-answer-v1",
        "service": "vllm",
        "base_url": "http://127.0.0.1:8000/v1/",
        "model": "test/judge",
        "authentication": "none",
        "request_timeout_seconds": 30,
        "max_calls": max_calls,
        "max_input_bytes": 100000,
        "max_output_tokens": max_calls * 128,
    }


def recipe(case_ids=("one", "two")):
    value = _recipe(list(case_ids))
    value["plan"]["judge"].update(
        provider="openai_compatible",
        requested_model="test/judge",
        approved_resolved_models=["test/judge"],
        model_identity={"kind": "hosted_api", "weights_sha256": None},
        service_identity={
            "service": configuration()["service"],
            "endpoint_sha256": hashlib.sha256(
                configuration()["base_url"].encode()
            ).hexdigest(),
        },
    )
    value["plan"]["judge"]["config"].update(
        temperature="0", top_p="1", max_output_tokens=128, reasoning_effort=None, seed=7
    )
    value["plan"]["schedule"]["retry_on"] = []
    value["collection"] = configuration(len(case_ids) * 2)
    value["runner"] = {"scorer_id": "judge"}
    return value


def stage(tmp_path, route):
    path, value, _, runs = _request(tmp_path)
    policy = recipe()
    (tmp_path / "policy.json").write_bytes(canonical_payload(policy))
    if route == "captured":
        return load_evaluation_request(path)
    plan, analysis = finalize_native_plan(policy, *runs)
    for name, payload in (
        ("plan", plan),
        ("analysis", analysis),
        ("collection", policy["collection"]),
    ):
        (tmp_path / (name + ".json")).write_bytes(canonical_payload(payload))
    value = collected(tmp_path)
    value["execution"]["collection"].pop("model")
    value["execution"]["collection"]["integration"] = "openai-compatible-judge"
    value["comparison"]["policy"] = "analysis.json"
    path.write_text(yaml.safe_dump(value))
    return load_judge_request(path)


def stage_functional_control(tmp_path):
    path, value, _, _ = _request(tmp_path)
    case_ids = tuple(f"case-{index:02d}" for index in range(16))
    runs = []
    for side, marker in (("baseline", "a"), ("subject", "b")):
        records = [
            {
                "id": case_id,
                "input": f"Task {case_id}",
                "expected": "expected",
                "output": f"{side}-{case_id}",
            }
            for case_id in case_ids
        ]
        run = make_run(
            records,
            source={"name": "functional-control", "version": "1"},
            run_id=side,
            artifact_digest="sha256:" + marker * 64,
        )
        (tmp_path / f"{side}.json").write_bytes(canonical_payload(run))
        runs.append(run)
    policy = recipe(case_ids)
    (tmp_path / "policy.json").write_bytes(canonical_payload(policy))
    plan, analysis = finalize_native_plan(policy, *runs)
    for name, payload in (
        ("plan", plan),
        ("analysis", analysis),
        ("collection", policy["collection"]),
    ):
        (tmp_path / f"{name}.json").write_bytes(canonical_payload(payload))
    value = collected(tmp_path)
    value["execution"]["collection"].pop("model")
    value["execution"]["collection"]["integration"] = "openai-compatible-judge"
    value["comparison"]["policy"] = "analysis.json"
    path.write_text(yaml.safe_dump(value))
    return load_judge_request(path), policy["collection"]


def install_functional_control_transport(monkeypatch, config, outcome):
    from types import SimpleNamespace

    from invarlock.judge_measurements import openai_compatible

    calls = []
    monkeypatch.setenv("INVARLOCK_ALLOW_JUDGE_NETWORK", "1")

    class Stream:
        def __init__(self, response_bytes):
            self.response_bytes = response_bytes

        async def __aenter__(self):
            async def chunks(chunk_size):
                assert chunk_size == 64 * 1024
                yield self.response_bytes

            return SimpleNamespace(
                status_code=200,
                headers={"content-type": "application/json"},
                aiter_raw=chunks,
            )

        async def __aexit__(self, *_):
            return False

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def stream(self, method, path, *, content):
            assert method == "POST" and path == "chat/completions"
            body = json.loads(content)
            calls.append(body)
            normalized = json.loads(body["messages"][-1]["content"])
            baseline = normalized["answer"].startswith("baseline-")
            rating = (
                "unknown"
                if outcome == "incomplete" and len(calls) == 1
                else (
                    "incorrect"
                    if baseline == (outcome == "accepted")
                    else "correct"
                )
            )
            response_bytes = canonical_payload(
                {
                    "model": "test/judge",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"rating": rating}),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 4},
                }
            )
            return Stream(response_bytes)

    def client(actual, headers):
        assert actual == config
        assert headers.get("accept-encoding") == "identity"
        return Client()

    monkeypatch.setattr(openai_compatible, "_client", client)
    return calls


@pytest.mark.parametrize("invalid", ["model", "invocation_timeout_seconds", "unknown"])
def test_endpoint_request_rejects_ignored_local_or_runner_settings(tmp_path, invalid):
    value = collected(tmp_path)
    collection = value["execution"]["collection"]
    collection.pop("model")
    collection["integration"] = "openai-compatible-judge"
    collection[invalid] = model(tmp_path) if invalid == "model" else 1
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(JudgeWorkflowError, match="invalid"):
        load_judge_request(path)


def test_endpoint_request_preserves_explicit_integration_without_artifact(tmp_path):
    request = stage(tmp_path, "v3")
    assert request.integration == "openai-compatible-judge"
    assert request.model is None
    assert dict(request.runner) == {"scorer_id": "judge"}
    assert json.loads(request.inputs["collection"].read_bytes()) == configuration()


@pytest.mark.parametrize("route", ["v3", "captured"])
def test_endpoint_preflight_is_execution_free_and_reserves_complete_plan(
    tmp_path, monkeypatch, route
):
    from invarlock.judge_measurements import openai_compatible

    monkeypatch.delenv("INVARLOCK_ALLOW_JUDGE_NETWORK", raising=False)
    monkeypatch.setattr(openai_compatible, "network_policy_allows", lambda: False)
    request = stage(tmp_path, route)
    if route == "v3":
        result = preflight_judge_request(request).payload
        assert result["collection_integration"]["api"].endswith(
            "collect_openai_compatible"
        )
        assert result["budget_capacity"]["full_plan_reserved"]
    else:
        result = preflight_captured_judge(
            request, signing_key_path=None, unsigned=True
        ).payload
        assert result["maximum_admitted_calls"] == 4
    assert result["network_calls"] == 0
    environment = result["collection_environment"]
    assert environment["network_authorized"] is False
    assert (
        environment["network_authorization_variable"] == "INVARLOCK_ALLOW_JUDGE_NETWORK"
    )
    assert result["judge"]["model_identity"] == {
        "kind": "hosted_api",
        "weights_sha256": None,
    }
    assert not request.evidence.exists()
    workspace = request.workspace if route == "v3" else request.judge.workspace
    assert not workspace.exists()


@pytest.mark.parametrize("fault", ["integration", "runner"])
def test_programmatic_endpoint_request_cannot_switch_profile_or_ignore_runner(
    tmp_path, fault
):
    request = stage(tmp_path, "v3")
    if fault == "integration":
        request = replace(request, integration="inspect-judge")
    else:
        request = replace(
            request,
            runner=MappingProxyType(
                {
                    "scorer_id": "judge",
                    "invocation_timeout_seconds": 30,
                }
            ),
        )
    with pytest.raises(JudgeWorkflowError, match="Inspect judge fields|ignored"):
        preflight_judge_request(request)


@pytest.mark.parametrize("operation", ["preflight", "collect"])
@pytest.mark.parametrize("fault", ["profile", "model", "missing_plan_or_runner"])
def test_endpoint_wrapper_rejects_misbound_configuration(operation, fault, tmp_path):
    kwargs = {"model": None, "integration": "openai-compatible-judge"}
    if fault == "profile":
        kwargs["integration"] = "inspect-judge"
    elif fault == "model":
        kwargs["model"] = object()
    with pytest.raises(JudgeWorkflowError, match="differs|requires"):
        if operation == "preflight":
            kwargs["plan"] = None if fault == "missing_plan_or_runner" else {}
            native_workflow.collection_preflight(configuration(), **kwargs)
        else:
            native_workflow.collect_frozen(
                plan={},
                collection=configuration(),
                runner={"scorer_id": "judge", "invocation_timeout_seconds": 1}
                if fault == "missing_plan_or_runner"
                else {"scorer_id": "judge"},
                workspace=tmp_path,
                baseline_run={},
                subject_run={},
                **kwargs,
            )


@pytest.mark.parametrize("operation", ["preflight", "collect"])
def test_hosted_service_wrapper_rejects_unused_local_artifact(operation, tmp_path):
    collection = _recipe()["collection"]
    kwargs = {"model": object()}
    with pytest.raises(JudgeWorkflowError, match="cannot use a local artifact"):
        if operation == "preflight":
            native_workflow.collection_preflight(collection, **kwargs)
        else:
            native_workflow.collect_frozen(
                plan={},
                collection=collection,
                runner={
                    "scorer_id": "judge",
                    "invocation_timeout_seconds": 30,
                },
                workspace=tmp_path,
                baseline_run={},
                subject_run={},
                **kwargs,
            )


def native_request(tmp_path, monkeypatch):
    from invarlock.core.evaluation_request import (
        ArtifactRequest,
        ComparisonSideRequest,
        RuntimeRequest,
    )
    from tests.judge_measurements import test_local_workflow_routes as routes

    # Reuse authentic synthetic native side receipts with the endpoint policy.
    # The temporary local binding is removed before the authored request is used.
    with monkeypatch.context() as patch:
        patch.setattr(routes, "recipe", recipe)
        request, kwargs, sides = routes.native(
            tmp_path,
            ComparisonSideRequest(
                ArtifactRequest(tmp_path, "unused", "artifact://unused"),
                RuntimeRequest("hf_transformers", {}),
            ),
        )
    request.comparison.judge.model = None
    kwargs["normalized_request"]["comparison"]["judge"].pop("model")
    return request, kwargs, sides


def test_native_endpoint_preflight_keeps_service_identity_and_never_captures(
    tmp_path, monkeypatch
):
    from invarlock.judge_measurements import openai_compatible

    monkeypatch.delenv("INVARLOCK_ALLOW_JUDGE_NETWORK", raising=False)
    monkeypatch.setattr(openai_compatible, "network_policy_allows", lambda: False)
    request, kwargs, _ = native_request(tmp_path, monkeypatch)
    result = native_workflow.preflight_native_judge(
        request, kwargs["schedule"], kwargs["policy_bytes"]
    )
    assert result["network_calls"] == 0
    assert result["judge"]["model_identity"]["kind"] == "hosted_api"
    assert result["maximum_admitted_calls"] == 4
    assert result["collection"]["network_authorized"] is False
    assert not request.comparison.judge.workspace.exists()
    assert not request.output.evidence.exists()


@pytest.mark.parametrize("route", ["native", "captured"])
def test_service_recipe_rejects_local_artifact_binding_before_collection(
    tmp_path, monkeypatch, route
):
    from invarlock.core.evaluation_request import (
        ArtifactRequest,
        ComparisonSideRequest,
        RuntimeRequest,
    )

    binding = ComparisonSideRequest(
        ArtifactRequest(tmp_path / "judge-model", "unused", "artifact://unused"),
        RuntimeRequest("hf_transformers", model(tmp_path)["runtime"]["settings"]),
    )
    if route == "native":
        request, kwargs, _ = native_request(tmp_path, monkeypatch)
        request.comparison.judge.model = binding
        with pytest.raises(JudgeWorkflowError, match="cannot use a local artifact"):
            native_workflow.preflight_native_judge(
                request, kwargs["schedule"], kwargs["policy_bytes"]
            )
    else:
        request = stage(tmp_path, "captured")
        request = replace(request, judge=replace(request.judge, model=binding))
        with pytest.raises(JudgeWorkflowError, match="cannot use a local artifact"):
            preflight_captured_judge(request, signing_key_path=None, unsigned=True)


@pytest.fixture
def transport(monkeypatch):
    from types import SimpleNamespace

    from invarlock.judge_measurements import openai_compatible

    calls = []
    monkeypatch.setenv("INVARLOCK_ALLOW_JUDGE_NETWORK", "1")

    response_bytes = canonical_payload(
        {
            "model": "test/judge",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"rating":"correct"}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 4},
        }
    )

    class Stream:
        async def __aenter__(self):
            async def chunks(chunk_size):
                assert chunk_size == 64 * 1024
                yield response_bytes

            return SimpleNamespace(
                status_code=200,
                headers={"content-type": "application/json"},
                aiter_raw=chunks,
            )

        async def __aexit__(self, *_):
            return False

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def stream(self, method, path, *, content):
            assert method == "POST"
            assert path == "chat/completions"
            calls.append(json.loads(content))
            return Stream()

    def client(config, headers):
        assert config == configuration()
        assert headers.get("accept-encoding") == "identity"
        assert "authorization" not in headers
        return Client()

    monkeypatch.setattr(openai_compatible, "_client", client)
    return calls


@pytest.mark.parametrize("route", ["v3", "captured", "native"])
def test_endpoint_collection_publishes_exact_service_evidence_and_replays(
    tmp_path, monkeypatch, transport, route
):
    from invarlock.judge_measurements.captured_workflow import evaluate_captured_judge
    from invarlock.judge_measurements.evidence import replay_judge_evidence
    from invarlock.judge_measurements.workflow import evaluate_judge_request

    if route == "native":
        request, kwargs, sides = native_request(tmp_path, monkeypatch)
        result = native_workflow.evaluate_native_judge(
            **kwargs, capture=lambda: tuple(sides)
        ).payload
        evidence = request.output.evidence
    else:
        request = stage(tmp_path, route)
        if route == "v3":
            result = evaluate_judge_request(
                request, signing_key=None, unsigned=True
            ).payload
        else:
            result = evaluate_captured_judge(
                request, signing_key_path=None, unsigned=True
            ).payload
        evidence = request.evidence
    assert result["ok"] and len(transport) == 4
    assert all(body["model"] == "test/judge" for body in transport)
    assert (
        replay_judge_evidence(evidence).analysis_result.to_dict()["decision"]
        == "insufficient_evidence"
    )
    measurements = json.loads((evidence / "measurements.json").read_bytes())
    assert measurements["source_profile"] == "retained-openai-compatible-judge-v1"
    assert measurements["completeness"]["completed_trials"] == 4
    retained_plan = json.loads((evidence / "plan.json").read_bytes())
    assert (
        retained_plan["judge"]["service_identity"]
        == recipe()["plan"]["judge"]["service_identity"]
    )


@pytest.mark.parametrize(
    ("outcome", "decision", "accepted"),
    [
        ("accepted", "pass", True),
        ("adverse", "regression", False),
        ("incomplete", "insufficient_evidence", False),
    ],
)
def test_endpoint_functional_controls_collect_replay_verify_and_report(
    tmp_path, monkeypatch, outcome, decision, accepted
):
    from invarlock.judge_measurements.acceptance import (
        verify_judge_evidence_with_policy,
    )
    from invarlock.judge_measurements.evidence import replay_judge_evidence
    from invarlock.judge_measurements.reporting import render_judge_evidence
    from invarlock.judge_measurements.workflow import evaluate_judge_request

    request, config = stage_functional_control(tmp_path)
    calls = install_functional_control_transport(monkeypatch, config, outcome)
    key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "signing-key.pem"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)

    result = evaluate_judge_request(
        request, signing_key=key_path, unsigned=False
    ).payload
    assert result["decision"] == decision
    assert len(calls) == 32
    replayed = replay_judge_evidence(request.evidence)
    assert replayed.analysis_result.decision == decision
    completeness = json.loads(
        (request.evidence / "measurements.json").read_bytes()
    )["completeness"]
    assert completeness["expected_trials"] == 32
    assert completeness["completed_trials"] == (
        31 if outcome == "incomplete" else 32
    )

    envelope = json.loads((request.evidence / "envelope.json").read_bytes())
    recipient = {
        "format": "invarlock/judge-measurement-recipient-policy-v1",
        "decision_scope": envelope["decision_scope"],
        "intended_subject": envelope["intended_subject"],
        "required_metric_name": "factual-correctness",
        "trusted_signer": {
            name: envelope["signer"][name]
            for name in ("identity", "public_key_sha256")
        },
        "bindings": copy.deepcopy(envelope["bindings"]),
        "required_decision": "pass",
    }
    recipient_path = tmp_path / "recipient.json"
    recipient_path.write_bytes(canonical_payload(recipient))
    verification = verify_judge_evidence_with_policy(
        request.evidence, recipient_path
    )
    assert verification.authenticated and verification.replayed and verification.verified
    assert verification.decision == decision
    assert verification.accepted is accepted

    html = tmp_path / "report.html"
    report = render_judge_evidence(request.evidence, html_path=html)
    assert report.facts["analysis"]["decision"] == decision
    assert html.is_file()


@pytest.mark.parametrize("route", ["v3", "captured", "native"])
def test_endpoint_execution_requires_network_authorization_before_side_effects(
    tmp_path, monkeypatch, route
):
    from invarlock.judge_measurements import openai_compatible
    from invarlock.judge_measurements.captured_workflow import evaluate_captured_judge
    from invarlock.judge_measurements.workflow import evaluate_judge_request

    monkeypatch.delenv("INVARLOCK_ALLOW_JUDGE_NETWORK", raising=False)
    monkeypatch.setattr(openai_compatible, "network_policy_allows", lambda: False)
    monkeypatch.setattr(native_workflow, "network_policy_allows", lambda: False)
    if route == "native":
        request, kwargs, _ = native_request(tmp_path, monkeypatch)
        with pytest.raises(JudgeWorkflowError, match="allowed network policy"):
            native_workflow.evaluate_native_judge(
                **kwargs,
                capture=lambda: pytest.fail(
                    "native answers were captured without network authorization"
                ),
            )
        workspace = request.comparison.judge.workspace
        evidence = request.output.evidence
    else:
        request = stage(tmp_path, route)
        with pytest.raises(JudgeWorkflowError, match="allowed network policy"):
            if route == "v3":
                evaluate_judge_request(request, signing_key=None, unsigned=True)
            else:
                evaluate_captured_judge(request, signing_key_path=None, unsigned=True)
        workspace = request.workspace if route == "v3" else request.judge.workspace
        evidence = request.evidence
    assert not workspace.exists()
    assert not evidence.exists()


def test_endpoint_collection_wrapper_does_not_require_status_callback(
    tmp_path, transport
):
    request = stage(tmp_path, "v3")
    inputs = {
        name: json.loads(request.inputs[name].read_bytes())
        for name in ("plan", "collection", "baseline_run", "subject_run")
    }
    result = native_workflow.collect_frozen(
        plan=inputs["plan"],
        collection=inputs["collection"],
        runner={"scorer_id": "judge"},
        workspace=tmp_path / "direct-workspace",
        baseline_run=inputs["baseline_run"],
        subject_run=inputs["subject_run"],
        integration="openai-compatible-judge",
    )
    assert result["completeness"]["completed_trials"] == 4
    assert len(transport) == 4


@pytest.mark.parametrize("route", ["v3", "captured", "native"])
def test_endpoint_changes_cannot_reuse_the_expected_judge_identity(
    tmp_path, monkeypatch, route
):
    if route == "native":
        request, kwargs, _ = native_request(tmp_path, monkeypatch)
        policy = json.loads(kwargs["policy_bytes"])
        policy["collection"]["base_url"] = "http://127.0.0.1:9000/v1"
        with pytest.raises(ValueError, match="identity|endpoint"):
            native_workflow.preflight_native_judge(
                request, kwargs["schedule"], canonical_payload(policy)
            )
        assert not request.comparison.judge.workspace.exists()
    else:
        request = stage(tmp_path, route)
        path = request.inputs["collection"] if route == "v3" else request.policy
        payload = json.loads(path.read_bytes())
        config = payload if route == "v3" else payload["collection"]
        config["base_url"] = "http://127.0.0.1:9000/v1"
        path.write_bytes(canonical_payload(payload))
        with pytest.raises(ValueError, match="identity|endpoint"):
            if route == "v3":
                preflight_judge_request(request)
            else:
                preflight_captured_judge(request, signing_key_path=None, unsigned=True)
        assert not request.evidence.exists()


@pytest.mark.parametrize("route", ["native", "captured"])
def test_service_recipe_rejects_unused_runner_timeout(tmp_path, monkeypatch, route):
    if route == "native":
        request, kwargs, _ = native_request(tmp_path, monkeypatch)
        policy = json.loads(kwargs["policy_bytes"])
        policy["runner"]["invocation_timeout_seconds"] = 1
        with pytest.raises(ValueError, match="judge runner must contain exactly"):
            native_workflow.preflight_native_judge(
                request, kwargs["schedule"], canonical_payload(policy)
            )
    else:
        request = stage(tmp_path, route)
        policy = json.loads(request.policy.read_bytes())
        policy["runner"]["invocation_timeout_seconds"] = 1
        request.policy.write_bytes(canonical_payload(policy))
        with pytest.raises(
            JudgeWorkflowError, match="judge runner must contain exactly"
        ):
            preflight_captured_judge(request, signing_key_path=None, unsigned=True)
