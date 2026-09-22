"""Local judge request bindings stay closed, confined and identity preserving."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from invarlock.core.evaluation_request import (
    EvaluationRequestError,
    load_evaluation_request,
)
from invarlock.evaluation_transaction import _normalized_request
from tests.core.test_evaluation_request_contract import (
    _materialize_import_inputs,
    _materialize_run_inputs,
    _request_payload,
    _write_request,
)


def model(tmp_path):
    (tmp_path / "judge-model").mkdir(exist_ok=True)
    value = deepcopy(_request_payload()["comparison"]["baseline"])
    value["artifact"].update(path="judge-model", model_id="org/judge")
    return value


def native(tmp_path, *, mode="run"):
    _materialize_run_inputs(tmp_path)
    if mode == "import":
        _materialize_import_inputs(tmp_path)
    value = _request_payload(mode=mode)
    value["comparison"].update(
        metric="judge",
        judge={
            "workspace": "judge-work",
            "signer_identity": "release",
            "model": model(tmp_path),
        },
    )
    return value


def collected(tmp_path):
    return {
        "format_version": "invarlock/evaluation-request-v3",
        "execution": {
            "mode": "judge_collect",
            "collection": {
                "integration": "runtime-provider-judge",
                "configuration": "collection.json",
                "model": model(tmp_path),
            },
        },
        "comparison": {
            "baseline_run": "baseline.json",
            "subject_run": "subject.json",
            "plan": "plan.json",
            "measurements": None,
            "policy": "policy.json",
        },
        "output": {"evidence": "evidence", "signer_identity": "release"},
    }


@pytest.mark.parametrize("mode", ["run", "import"])
def test_native_model_binding_is_typed_and_not_discarded(tmp_path, mode):
    value = native(tmp_path, mode=mode)
    request = load_evaluation_request(_write_request(tmp_path / "request.yaml", value))
    binding = request.comparison.judge.model
    assert binding.artifact.path == tmp_path / "judge-model"
    assert binding.artifact.model_id == "org/judge"
    assert binding.runtime.provider == "hf_transformers"
    with pytest.raises(TypeError):
        binding.runtime.settings["seed"] = 1
    if mode == "import":
        normalized = _normalized_request(request, None, ())
        actual = normalized["comparison"]["judge"]["model"]
        expected = deepcopy(value["comparison"]["judge"]["model"])
        expected["artifact"].pop("path")
        assert actual == expected


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "missing_path",
        "escape",
        "bool_seed",
        "non_scalar",
        "workspace_overlap",
        "evidence_overlap",
        "symlink",
    ],
)
def test_native_model_rejects_unsafe_or_invalid_binding(tmp_path, mutation):
    value = native(tmp_path)
    binding = value["comparison"]["judge"]["model"]
    if mutation == "unknown":
        binding["unrecognized"] = True
    elif mutation == "missing_path":
        binding["artifact"].pop("path")
    elif mutation == "escape":
        binding["artifact"]["path"] = "../model"
    elif mutation == "bool_seed":
        binding["runtime"]["settings"]["seed"] = True
    elif mutation == "non_scalar":
        binding["runtime"]["settings"]["seed"] = {"value": 0}
    elif mutation == "workspace_overlap":
        value["comparison"]["judge"]["workspace"] = "judge-model/work"
    elif mutation == "evidence_overlap":
        value["output"]["evidence"] = "judge-model/evidence"
    else:
        (tmp_path / "linked-model").symlink_to(
            tmp_path / "judge-model", target_is_directory=True
        )
        binding["artifact"]["path"] = "linked-model"
    with pytest.raises(EvaluationRequestError):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", value))


@pytest.mark.parametrize(
    "filename",
    [
        "evaluation_request.schema.json",
        "evaluation_request_v2.schema.json",
        "evaluation_request_v3.schema.json",
    ],
)
def test_packaged_local_judge_schema_matches_source(filename):
    root = Path(__file__).parents[2]
    assert (root / "contracts" / filename).read_bytes() == (
        root / "src/invarlock/_data/contracts" / filename
    ).read_bytes()
    assert json.loads((root / "contracts" / filename).read_bytes())


def captured(tmp_path):
    from tests.core.test_evaluation_request_contract import _captured_payload

    value = _captured_payload()
    (tmp_path / "inputs").mkdir()
    for name in ("run", "subject"):
        (tmp_path / "inputs" / (name + ".json")).write_text("{}")
    (tmp_path / "policy.json").write_text("{}")
    value["comparison"].update(
        metric="judge",
        judge={
            "workspace": "judge-work",
            "signer_identity": "release",
            "model": model(tmp_path),
        },
    )
    return value


def test_captured_model_binding_is_not_dropped(tmp_path):
    value = captured(tmp_path)
    request = load_evaluation_request(_write_request(tmp_path / "request.yaml", value))
    assert request.judge.model.artifact.path == tmp_path / "judge-model"
    assert request.judge.model.runtime.provider == "hf_transformers"


@pytest.mark.parametrize(
    "mutation",
    ["measurements", "workspace_overlap", "evidence_overlap", "symlink", "unknown"],
)
def test_captured_local_binding_rejects_ignored_and_unsafe_settings(tmp_path, mutation):
    value = captured(tmp_path)
    judge = value["comparison"]["judge"]
    if mutation == "measurements":
        (tmp_path / "measurements.json").write_text("{}")
        judge["measurements"] = "measurements.json"
    elif mutation == "workspace_overlap":
        judge["workspace"] = "judge-model/work"
    elif mutation == "evidence_overlap":
        value["output"]["evidence"] = "judge-model/evidence"
    elif mutation == "symlink":
        (tmp_path / "linked").symlink_to(
            tmp_path / "judge-model", target_is_directory=True
        )
        judge["model"]["artifact"]["path"] = "linked"
    else:
        judge["model"]["ignored"] = True
    with pytest.raises(EvaluationRequestError):
        load_evaluation_request(_write_request(tmp_path / "request.yaml", value))


@pytest.mark.parametrize("route", ["v3", "captured", "captured-auto"])
def test_cli_local_model_uses_installed_llama_cpp_registry(
    tmp_path, monkeypatch, route
):
    from dataclasses import fields

    import invarlock.evaluation_transaction as transaction
    from invarlock.cli.evaluation_workflow import EvaluationOptions, execute_evaluation
    from invarlock.judge_measurements import workflow
    from tests.runtime.test_llama_cpp import _runtime_inputs

    spec, bindings, _ = _runtime_inputs(tmp_path)
    value = collected(tmp_path) if route == "v3" else captured(tmp_path)
    binding = {
        "artifact": {
            "path": bindings.gguf_path.name,
            "model_id": spec.model_id,
            "locator": "gguf://test/judge",
        },
        "runtime": {"provider": spec.provider_name, "settings": dict(spec.settings)},
    }
    if route == "v3":
        value["execution"]["collection"]["model"] = binding
    else:
        value["comparison"]["judge"]["model"] = binding
    path = _write_request(tmp_path / "request.yaml", value)
    seen = []

    def preflight(request, **kwargs):
        local = request.model if route == "v3" else request.judge.model
        assert local.runtime.provider == "llama_cpp"
        assert local.artifact.path == bindings.gguf_path
        seen.append(local)
        return workflow.JudgeWorkflowResult({"ready": True})

    monkeypatch.setattr(workflow, "preflight_judge_request", preflight)
    monkeypatch.setattr(transaction, "preflight_evaluation_request", preflight)
    options = EvaluationOptions(
        **(
            {field.name: None for field in fields(EvaluationOptions)}
            | {
                "request": path,
                "preflight": True,
                "unsigned": True,
                "allow_installed_scorers": False,
                "max_bootstrap_draws": 100,
            }
        )
    )
    result = execute_evaluation(
        options,
        command_line=frozenset(),
        initial_mode="judge_collect"
        if route == "v3"
        else None
        if route == "captured-auto"
        else "captured",
    )
    assert result.result.payload["ready"] is True
    assert len(seen) == 1


@pytest.mark.parametrize("route", ["judge_collect", "captured"])
def test_cli_local_judge_rejects_bootstrap_override_before_execution(tmp_path, route):
    from dataclasses import fields

    from invarlock.cli.evaluation_workflow import EvaluationOptions, execute_evaluation
    from invarlock.judge_measurements.workflow import JudgeWorkflowError

    value = collected(tmp_path) if route == "judge_collect" else captured(tmp_path)
    path = _write_request(tmp_path / "request.yaml", value)
    options = EvaluationOptions(
        **(
            {field.name: None for field in fields(EvaluationOptions)}
            | {
                "request": path,
                "preflight": True,
                "unsigned": True,
                "allow_installed_scorers": False,
                "max_bootstrap_draws": 100,
            }
        )
    )
    with pytest.raises(JudgeWorkflowError, match="bootstrap options do not apply"):
        execute_evaluation(
            options,
            command_line=frozenset({"max_bootstrap_draws"}),
            initial_mode=route,
        )
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize(
    "mutation",
    ["untyped", "no_path", "escape", "nonfinite", "invalid_locator", "symlink"],
)
def test_programmatic_native_model_revalidates_structure_and_paths(tmp_path, mutation):
    from dataclasses import replace

    from invarlock.core.evaluation_request import _validate_judge_workspace_inputs

    request = load_evaluation_request(
        _write_request(tmp_path / "request.yaml", native(tmp_path))
    )
    local = request.comparison.judge.model
    if mutation == "untyped":
        local = {"artifact": "ignored"}
    elif mutation == "no_path":
        local = replace(local, artifact=replace(local.artifact, path=None))
    elif mutation == "escape":
        local = replace(
            local, artifact=replace(local.artifact, path=tmp_path.parent / "model")
        )
    elif mutation == "nonfinite":
        local = replace(
            local, runtime=replace(local.runtime, settings={"seed": float("nan")})
        )
    elif mutation == "invalid_locator":
        local = replace(local, artifact=replace(local.artifact, locator=None))
    else:
        (tmp_path / "judge-model").rmdir()
        (tmp_path / "judge-model").symlink_to(
            tmp_path / "models/baseline", target_is_directory=True
        )
    request = replace(
        request,
        comparison=replace(
            request.comparison, judge=replace(request.comparison.judge, model=local)
        ),
    )
    with pytest.raises(EvaluationRequestError):
        _validate_judge_workspace_inputs(request)


def test_programmatic_captured_model_cannot_be_ignored_with_measurements(tmp_path):
    from dataclasses import replace

    from invarlock.captured_evaluation import CapturedEvaluationError
    from invarlock.evaluation_transaction import _captured_request

    request = load_evaluation_request(
        _write_request(tmp_path / "request.yaml", captured(tmp_path))
    )
    request = replace(
        request,
        judge=replace(request.judge, measurements=tmp_path / "measurements.json"),
    )
    with pytest.raises(CapturedEvaluationError, match="cannot accompany"):
        _captured_request(request)


def test_public_captured_file_loader_resolves_llama_cpp(tmp_path):
    from invarlock.evaluation_transaction import _captured_request
    from tests.runtime.test_llama_cpp import _runtime_inputs

    spec, bindings, _ = _runtime_inputs(tmp_path)
    value = captured(tmp_path)
    value["comparison"]["judge"]["model"] = {
        "artifact": {
            "path": bindings.gguf_path.name,
            "model_id": spec.model_id,
            "locator": "gguf://test/judge",
        },
        "runtime": {"provider": spec.provider_name, "settings": dict(spec.settings)},
    }
    request = _captured_request(_write_request(tmp_path / "request.yaml", value))
    assert request.judge.model.runtime.provider == "llama_cpp"


def test_normalized_request_binds_judge_identity_and_settings(tmp_path):
    from dataclasses import replace

    from invarlock.judge_measurements.evidence import object_sha256

    value = native(tmp_path, mode="import")
    request = load_evaluation_request(_write_request(tmp_path / "request.yaml", value))
    before = _normalized_request(request, None, ())
    local = request.comparison.judge.model
    changed = replace(
        local,
        runtime=replace(
            local.runtime, settings=dict(local.runtime.settings) | {"seed": 99}
        ),
    )
    request = replace(
        request,
        comparison=replace(
            request.comparison, judge=replace(request.comparison.judge, model=changed)
        ),
    )
    after = _normalized_request(request, None, ())
    assert before["comparison"]["judge"]["model"]["runtime"]["settings"]["seed"] == 0
    assert after["comparison"]["judge"]["model"]["runtime"]["settings"]["seed"] == 99
    assert object_sha256(before) != object_sha256(after)


def test_programmatic_native_cannot_ignore_judge_model_under_another_metric(tmp_path):
    from dataclasses import replace

    from invarlock.core.evaluation_request import _validate_judge_workspace_inputs

    request = load_evaluation_request(
        _write_request(tmp_path / "request.yaml", native(tmp_path))
    )
    request = replace(
        request, comparison=replace(request.comparison, metric="exact_match")
    )
    with pytest.raises(EvaluationRequestError, match="must accompany"):
        _validate_judge_workspace_inputs(request)


def test_programmatic_captured_cannot_ignore_judge_model_under_another_metric(tmp_path):
    from dataclasses import replace

    from invarlock.captured_evaluation import CapturedEvaluationError
    from invarlock.evaluation_transaction import _captured_request

    request = load_evaluation_request(
        _write_request(tmp_path / "request.yaml", captured(tmp_path))
    )
    with pytest.raises(CapturedEvaluationError, match="must accompany"):
        _captured_request(replace(request, metric=None))


def test_all_request_formats_share_the_existing_model_binding_contract():
    root = Path(__file__).parents[2] / "contracts"
    native_schema = json.loads((root / "evaluation_request.schema.json").read_bytes())
    for name in (
        "evaluation_request_v2.schema.json",
        "evaluation_request_v3.schema.json",
    ):
        schema = json.loads((root / name).read_bytes())
        for definition in (
            "safeReference",
            "providerName",
            "settingName",
            "modelId",
            "artifactLocator",
            "jsonScalar",
            "artifact",
            "runtime",
            "comparisonSide",
        ):
            assert schema["$defs"][definition] == native_schema["$defs"][definition]
