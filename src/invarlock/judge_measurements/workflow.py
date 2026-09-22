"""Typed, execution-free preparation and publication of bounded judge evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from jsonschema import Draft202012Validator

from invarlock.captured_contracts import read_file
from invarlock.core.evaluation_request import (
    MAX_EVALUATION_REQUEST_BYTES,
    ComparisonSideRequest,
    EvaluationRequestError,
    ProviderResolver,
    _build_judge_request,
    _build_side,
    _default_provider_resolver,
    _judge_model_payload,
    _load_yaml,
    _reference_parts,
    _reject_include_directives,
    _resolve_output_reference,
    _validate_judge_model_binding,
)
from invarlock.evidence_pack_json import parse_json_bytes
from invarlock.judge_measurement_types import (
    JudgeAnalysisPolicyDocument,
    JudgeMeasurementPlan,
    JudgeMeasurements,
)
from invarlock.judge_measurements.analysis import decode_analysis_policy
from invarlock.judge_measurements.contracts import (
    _validate_frozen_answer_bindings,
    _validate_inspect_plan_collection_identity,
    canonical_payload,
    validate_measurement_plan,
    validate_measurements,
)
from invarlock.public_contracts import load_judge_evaluation_request_schema

MAX_WORKFLOW_BYTES = 384 * 1024 * 1024
_INPUT_LIMITS = {
    "plan": 64 * 1024 * 1024,
    "measurements": MAX_WORKFLOW_BYTES,
    "baseline_run": 128 * 1024 * 1024,
    "subject_run": 128 * 1024 * 1024,
    "policy": 64 * 1024,
    "collection": 64 * 1024,
}
_COLLECTION_FIELDS = {
    "grader",
    "inspect_version",
    "profile",
    "epochs",
    "log_model_api",
    "log_samples",
    "sdk_max_retries",
    "tools",
    "concurrency",
    "requests_per_minute",
    "request_timeout_seconds",
    "max_calls",
    "max_input_tokens",
    "max_output_tokens",
    "max_cost_microusd",
    "input_tokens_per_call",
    "cost_microusd_per_call",
}


class JudgeWorkflowError(ValueError):
    """A requested measurement workflow cannot complete safely."""

    exit_code = 2

    def __init__(self, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {
            "format_version": "invarlock/judge-evaluation-result-v1",
            "kind": "judge",
            "ok": False,
            "errors": [message[:1024]],
        }

    def as_json(self) -> str:
        return canonical_payload(self.payload).decode()


@dataclass(frozen=True)
class JudgeEvaluationRequest:
    root: Path
    mode: Literal["judge_import", "judge_collect"]
    inputs: Mapping[str, Path]
    evidence: Path
    signer_identity: str
    workspace: Path | None = None
    runner: Mapping[str, Any] | None = None
    integration: (
        Literal["inspect-judge", "runtime-provider-judge", "openai-compatible-judge"]
        | None
    ) = None
    model: ComparisonSideRequest | None = None


@dataclass(frozen=True)
class JudgeWorkflowResult:
    payload: dict[str, Any]

    @property
    def policy_verdict(self) -> str:
        return str(self.payload.get("decision", "insufficient_evidence"))

    def as_json(self) -> str:
        return canonical_payload(self.payload).decode()


def load_judge_request(
    path: Path,
    *,
    request_root: Path | None = None,
    baseline_run: Path | None = None,
    subject_run: Path | None = None,
    output: Path | None = None,
    provider_resolver: ProviderResolver | None = None,
) -> JudgeEvaluationRequest:
    """Load a closed request; missing input files remain visible to preflight."""
    try:
        path = Path(path)
        root = (request_root or path.parent).resolve(strict=True)
        value = _load_yaml(read_file(path, MAX_EVALUATION_REQUEST_BYTES))
        _reject_include_directives(value)
        error = next(
            Draft202012Validator(load_judge_evaluation_request_schema()).iter_errors(
                value
            ),
            None,
        )
        if error is not None:
            raise EvaluationRequestError(
                f"judge request is invalid: {error.message[:240]}"
            )
        inputs: dict[str, Path] = {}
        for name, reference in value["comparison"].items():
            if reference is not None:
                inputs[name] = root.joinpath(
                    *_reference_parts(reference, label=f"comparison.{name}")
                )
        collection = value["execution"]["collection"]
        model = None
        if collection is not None and "model" in collection:
            model = _build_side(
                collection["model"],
                side_name="judge.model",
                execution_mode="run",
                root=root,
                provider_cache={},
                provider_resolver=provider_resolver or _default_provider_resolver,
            )
        if collection is not None:
            inputs["collection"] = root.joinpath(
                *_reference_parts(
                    collection["configuration"],
                    label="execution.collection.configuration",
                )
            )
        for name, override in (
            ("baseline_run", baseline_run),
            ("subject_run", subject_run),
        ):
            if override is not None:
                relative = Path(override).absolute().relative_to(root).as_posix()
                inputs[name] = root.joinpath(*_reference_parts(relative, label=name))
        output_reference = value["output"]["evidence"]
        _reference_parts(output_reference, label="output.evidence")
        if output is not None:
            output_reference = Path(output).absolute().relative_to(root).as_posix()
        evidence = _resolve_output_reference(
            root, output_reference, label="output.evidence"
        )
        workspace = None
        runner = None
        if collection is not None:
            workspace = root.joinpath(
                *_reference_parts(
                    collection.get("workspace", output_reference + ".judge-work"),
                    label="execution.collection.workspace",
                )
            )
            if (
                workspace == evidence
                or workspace.is_relative_to(evidence)
                or evidence.is_relative_to(workspace)
            ):
                raise EvaluationRequestError(
                    "judge workspace and evidence destination must be separate"
                )
            runner = MappingProxyType(
                {
                    "scorer_id": collection.get("scorer_id", "judge"),
                    **(
                        {
                            "invocation_timeout_seconds": collection.get(
                                "invocation_timeout_seconds", 3600
                            )
                        }
                        if collection["integration"] == "inspect-judge"
                        else {}
                    ),
                }
            )
        if model is not None:
            artifact = model.artifact.path
            assert artifact is not None and workspace is not None
            if any(
                destination.is_relative_to(artifact)
                or artifact.is_relative_to(destination)
                for destination in (workspace, evidence)
            ):
                raise EvaluationRequestError(
                    "judge model must not overlap workspace or evidence destination"
                )
        for source in inputs.values():
            if workspace is not None and (
                source == workspace or source.is_relative_to(workspace)
            ):
                raise EvaluationRequestError(
                    "input files must remain outside the judge workspace"
                )
            if source == evidence or source.is_relative_to(evidence):
                raise EvaluationRequestError(
                    "input files must remain outside the evidence destination"
                )
        return JudgeEvaluationRequest(
            root,
            value["execution"]["mode"],
            MappingProxyType(inputs),
            evidence,
            value["output"]["signer_identity"],
            workspace,
            runner,
            collection["integration"] if collection is not None else None,
            model,
        )
    except (OSError, ValueError) as exc:
        if isinstance(exc, JudgeWorkflowError):
            raise
        raise JudgeWorkflowError(str(exc)) from exc


def _validate_request_binding(request: JudgeEvaluationRequest) -> None:
    """Apply the same closed authored contract to programmatic requests."""

    if (
        not isinstance(request.root, Path)
        or not isinstance(request.inputs, Mapping)
        or any(not isinstance(name, str) for name in request.inputs)
    ):
        raise JudgeWorkflowError("judge request root and input bindings are invalid")

    def reference(path: Path, label: str) -> str:
        if not isinstance(path, Path):
            raise JudgeWorkflowError(f"{label} must be a path")
        try:
            value = path.relative_to(request.root).as_posix()
        except ValueError as exc:
            raise JudgeWorkflowError(
                f"{label} must remain inside request root"
            ) from exc
        _reference_parts(value, label=label)
        return value

    comparison: dict[str, Any] = {
        name: reference(path, name)
        for name, path in request.inputs.items()
        if name != "collection"
    }
    collection = None
    if request.mode == "judge_collect":
        if (
            request.workspace is None
            or not isinstance(request.runner, Mapping)
            or "collection" not in request.inputs
        ):
            raise JudgeWorkflowError(
                "judge collection requires workspace, runner and configuration"
            )
        allowed_runner = {"scorer_id"}
        if request.integration == "inspect-judge":
            allowed_runner.add("invocation_timeout_seconds")
        if (
            set(request.runner) - allowed_runner
            or "measurements" in request.inputs
            or (
                request.integration != "inspect-judge"
                and set(request.runner) != {"scorer_id"}
            )
        ):
            raise JudgeWorkflowError(
                "judge collection contains ignored or misplaced settings"
            )
        comparison["measurements"] = None
        collection = {
            "integration": request.integration,
            "configuration": reference(request.inputs["collection"], "collection"),
            "workspace": reference(request.workspace, "workspace"),
            **dict(request.runner),
        }
        if request.model is not None:
            collection["model"] = _judge_model_payload(request.model, root=request.root)
    elif (
        any(
            value is not None
            for value in (
                request.integration,
                request.model,
                request.workspace,
                request.runner,
            )
        )
        or "collection" in request.inputs
    ):
        raise JudgeWorkflowError(
            "judge import must not contain ignored collection settings"
        )
    document: dict[str, Any] = {
        "format_version": "invarlock/evaluation-request-v3",
        "execution": {"mode": request.mode, "collection": collection},
        "comparison": comparison,
        "output": {
            "evidence": reference(request.evidence, "evidence"),
            "signer_identity": request.signer_identity,
        },
    }
    error = next(
        Draft202012Validator(load_judge_evaluation_request_schema()).iter_errors(
            document
        ),
        None,
    )
    if error is not None:
        raise JudgeWorkflowError(
            f"judge request binding is invalid: {error.message[:240]}"
        )
    _resolve_output_reference(
        request.root, document["output"]["evidence"], label="output.evidence"
    )
    if request.workspace is not None:
        assert collection is not None
        _build_judge_request(
            {
                "workspace": collection["workspace"],
                "signer_identity": request.signer_identity,
            },
            root=request.root,
            evidence_reference=document["output"]["evidence"],
        )
        if any(
            path.is_relative_to(request.workspace)
            or path.is_relative_to(request.evidence)
            for path in request.inputs.values()
        ):
            raise JudgeWorkflowError(
                "judge workspace and evidence must remain separate from input files"
            )
        if request.model is not None:
            _validate_judge_model_binding(
                request.model,
                root=request.root,
                workspace=request.workspace,
                evidence=request.evidence,
            )


def _read_inputs(request: JudgeEvaluationRequest) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    missing: list[str] = []
    total = 0
    for name, path in request.inputs.items():
        try:
            raw = read_file(path, _INPUT_LIMITS[name])
        except FileNotFoundError:
            missing.append(name)
            continue
        total += len(raw)
        if total > MAX_WORKFLOW_BYTES:
            raise JudgeWorkflowError(
                "combined judge inputs exceed the 384 MiB allowance"
            )
        value = parse_json_bytes(raw, label=f"judge {name}")
        if not isinstance(value, dict):
            raise JudgeWorkflowError(f"judge {name} must be a JSON object")
        values[name] = value
    return values, missing


def _collection_budgets(
    value: dict[str, Any], plan: JudgeMeasurementPlan
) -> dict[str, int]:
    if set(value) != _COLLECTION_FIELDS:
        raise JudgeWorkflowError(
            "collection configuration must contain exactly the supported Inspect judge fields"
        )
    if (
        value["inspect_version"] != "0.3.263"
        or value["profile"] != "inspect-text-frozen-answer-v1"
        or type(value["epochs"]) is not int
        or value["epochs"] != 1
        or value["log_model_api"] is not True
        or value["log_samples"] is not True
        or type(value["sdk_max_retries"]) is not int
        or value["sdk_max_retries"] != 0
        or value["tools"] is not False
    ):
        raise JudgeWorkflowError(
            "collection identity, logging, retry, epoch, or tool settings differ from the supported Inspect judge profile"
        )
    budgets: dict[str, int] = {}
    for key, maximum in (
        ("concurrency", 32),
        ("requests_per_minute", 10000),
        ("request_timeout_seconds", 3600),
        ("max_calls", 600000),
        ("max_input_tokens", 10**12),
        ("max_output_tokens", 10**12),
        ("max_cost_microusd", 10**12),
        ("input_tokens_per_call", 1048576),
        ("cost_microusd_per_call", 10**9),
    ):
        item = value[key]
        if type(item) is not int or not 1 <= item <= maximum:
            raise JudgeWorkflowError(
                f"collection {key} must be an explicit bounded positive integer"
            )
        budgets[key] = item
    if plan["schedule"]["max_attempts"] != 1:
        raise JudgeWorkflowError(
            "live Inspect collection currently requires one attempt per trial"
        )
    try:
        _validate_inspect_plan_collection_identity(
            plan,
            grader=value["grader"],
            inspect_version=value["inspect_version"],
        )
        from invarlock.judge_measurements.runner import (
            _require_qualified_live_provider_model,
        )

        _require_qualified_live_provider_model(value["grader"])
    except ValueError as exc:
        raise JudgeWorkflowError(str(exc)) from exc
    return budgets


def _prepare(request: JudgeEvaluationRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_request_binding(request)
    values, missing = _read_inputs(request)
    result: dict[str, Any] = {
        "format_version": "invarlock/judge-evaluation-preflight-v1",
        "kind": "judge",
        "execution_mode": request.mode,
        "ok": False,
        "ready": False,
        "evidence": str(request.evidence),
        "signer_identity": request.signer_identity,
        "missing_inputs": missing,
        "cases": None,
        "independent_units": None,
        "planned_trials": None,
        "maximum_attempts": None,
        "measurements": None,
        "judge": None,
        "rubric": None,
        "scale": None,
        "budgets": None,
        "budget_capacity": None,
        "collection_available": False,
        "network_calls": 0,
        "errors": [],
    }
    if "plan" in values:
        plan = cast(JudgeMeasurementPlan, values["plan"])
        validate_measurement_plan(plan)
        result.update(
            cases=len(plan["answer_bindings"]),
            independent_units=len(
                {item["unit_id"] for item in plan["sampling"]["case_units"]}
            ),
            planned_trials=plan["schedule"]["expected_trials"],
            maximum_attempts=plan["schedule"]["expected_trials"]
            * plan["schedule"]["max_attempts"],
            judge=plan["judge"],
            rubric={
                "sha256": plan["rubric"]["sha256"],
                "excerpt": plan["rubric"]["text"][:4096],
            },
            scale=plan["scale"],
        )
        if "policy" in values:
            policy = decode_analysis_policy(
                cast(JudgeAnalysisPolicyDocument, values["policy"]), plan=plan
            )
            result["metric"] = {
                "name": policy.metric_name,
                "direction": policy.direction,
                "allowed_degradation": str(policy.allowed_degradation),
                "minimum_units": policy.minimum_units,
                "maximum_interval_width": str(policy.maximum_interval_width),
            }
        if {"baseline_run", "subject_run"} <= values.keys():
            from invarlock.judge_measurements.native_capture import NATIVE_RUN_SOURCE

            if any(
                isinstance(values[name].get("source"), dict)
                and values[name]["source"].get("name") == NATIVE_RUN_SOURCE
                for name in ("baseline_run", "subject_run")
            ):
                raise JudgeWorkflowError(
                    "Native judge answers require their runtime capture; resume the native metric: judge request and workspace"
                )
            _validate_frozen_answer_bindings(
                values["plan"], values["baseline_run"], values["subject_run"]
            )
            result["baseline"] = values["baseline_run"]["run_id"]
            result["subject"] = values["subject_run"]["run_id"]
            if "measurements" in values:
                validate_measurements(
                    cast(JudgeMeasurements, values["measurements"]),
                    plan,
                    baseline_run=values["baseline_run"],
                    subject_run=values["subject_run"],
                )
                result["measurements"] = values["measurements"]["completeness"]
    if "collection" in values:
        if "plan" not in values:
            result["errors"].append(
                "Supply the plan before validating collection configuration"
            )
        else:
            plan = cast(JudgeMeasurementPlan, values["plan"])
            if request.integration == "runtime-provider-judge":
                from invarlock.judge_measurements.runtime_provider import (
                    validate_runtime_provider_collection,
                )

                budgets = validate_runtime_provider_collection(
                    values["collection"], plan
                )
                capacity = min(
                    budgets["max_calls"],
                    budgets["max_output_tokens"]
                    // plan["judge"]["config"]["max_output_tokens"],
                )
            elif request.integration == "openai-compatible-judge":
                from invarlock.judge_measurements.openai_compatible import (
                    validate_openai_compatible_collection,
                )

                budgets = validate_openai_compatible_collection(
                    values["collection"], plan
                )
                capacity = min(
                    budgets["max_calls"],
                    budgets["max_output_tokens"]
                    // plan["judge"]["config"]["max_output_tokens"],
                )
            else:
                budgets = _collection_budgets(values["collection"], plan)
                capacity = min(
                    budgets["max_calls"],
                    budgets["max_input_tokens"] // budgets["input_tokens_per_call"],
                    budgets["max_output_tokens"]
                    // plan["judge"]["config"]["max_output_tokens"],
                    budgets["max_cost_microusd"] // budgets["cost_microusd_per_call"],
                )
            result["budgets"] = budgets
            result["budget_capacity"] = {
                "maximum_admitted_calls": capacity,
                "planned_calls": plan["schedule"]["expected_trials"],
                "full_plan_reserved": capacity >= plan["schedule"]["expected_trials"],
            }
    if (
        request.mode == "judge_collect"
        and result["budget_capacity"] is not None
        and not result["budget_capacity"]["full_plan_reserved"]
    ):
        result["errors"].append(
            "Live collection budgets must reserve every planned call"
        )
    if missing:
        result["errors"].append(
            "Supply the missing request-relative inputs: " + ", ".join(missing)
        )
    if request.mode == "judge_collect":
        result["collection_integration"] = {
            "package": "invarlock",
            "api": "invarlock.judge_measurements.collect_runtime_provider"
            if request.integration == "runtime-provider-judge"
            else "invarlock.judge_measurements.collect_openai_compatible"
            if request.integration == "openai-compatible-judge"
            else "invarlock.judge_measurements.collect_configured",
            "execution": "installed_evaluate",
            "core_cli_execution": True,
        }
        result["workspace"] = str(request.workspace)
        if not result["errors"]:
            from invarlock.judge_measurements.native_workflow import (
                collection_preflight,
            )

            result["collection_environment"] = collection_preflight(
                values["collection"],
                **(
                    {
                        "integration": request.integration,
                        "model": request.model,
                        "request_root": request.root,
                        "plan": values["plan"],
                    }
                    if request.integration != "inspect-judge"
                    else {}
                ),
            )
            result["collection_available"] = True
        result["next_action"] = (
            "Run evaluate with a signing key to collect and publish judge evidence."
        )
    result["ok"] = result["ready"] = not result["errors"]
    return values, result


def preflight_judge_request(request: JudgeEvaluationRequest) -> JudgeWorkflowResult:
    """Validate retained inputs and show preparation without calls or publication."""
    try:
        _, payload = _prepare(request)
        return JudgeWorkflowResult(payload)
    except (OSError, ValueError) as exc:
        if isinstance(exc, JudgeWorkflowError):
            raise
        raise JudgeWorkflowError(str(exc)) from exc


def evaluate_judge_request(
    request: JudgeEvaluationRequest,
    *,
    signing_key: Path | None,
    unsigned: bool,
) -> JudgeWorkflowResult:
    """Publish imported evidence through its separately scoped evidence writer."""
    try:
        values, preflight = _prepare(request)
        if not preflight["ready"]:
            raise JudgeWorkflowError("Judge evaluation is not ready", payload=preflight)
        if unsigned == (signing_key is not None):
            raise JudgeWorkflowError(
                "Choose either --signing-key or explicit --unsigned for judge evidence"
            )
        from invarlock.judge_measurements.evidence import (
            _private_key,
            publish_judge_evidence,
        )

        # Authenticate the signing input before admitting a billable call.
        checked_key = _private_key(signing_key) if signing_key is not None else None
        collection_status = None
        if request.mode == "judge_collect":
            from invarlock.judge_measurements.evidence import object_sha256
            from invarlock.judge_measurements.native_workflow import (
                _require_service_network_authorization,
                _retain_identity,
                collect_frozen,
                locked_workspace,
                require_completed_collection,
            )

            assert request.workspace is not None and request.runner is not None
            _require_service_network_authorization(
                values["collection"], preflight["collection_environment"]
            )
            local_binding = {}
            if request.integration == "runtime-provider-judge":
                from invarlock.evaluation_transaction import _normalized_side

                assert request.model is not None
                local_binding = {
                    "integration": request.integration,
                    "model": _normalized_side(request.model),
                }
            elif request.integration == "openai-compatible-judge":
                local_binding = {"integration": request.integration}
            with locked_workspace(request.workspace) as unchanged:
                _retain_identity(
                    request.workspace / "identity.json",
                    {
                        "format": "invarlock/judge-collection-workspace-v1",
                        "inputs": {
                            key: object_sha256(value) for key, value in values.items()
                        },
                        "runner": dict(request.runner),
                        **local_binding,
                    },
                )
                unchanged()
                collection_stop: dict[str, str] = {}
                values["measurements"] = collect_frozen(
                    plan=values["plan"],
                    collection=values["collection"],
                    runner=dict(request.runner),
                    workspace=request.workspace,
                    baseline_run=values["baseline_run"],
                    subject_run=values["subject_run"],
                    status=collection_stop,
                    **cast(
                        dict[str, Any],
                        {
                            "integration": request.integration,
                            "model": request.model,
                            "request_root": request.root,
                        }
                        if request.integration != "inspect-judge"
                        else {},
                    ),
                )
                unchanged()
                collection_status = require_completed_collection(
                    values["measurements"],
                    request.workspace,
                    stop_reason=collection_stop.get("stop_reason"),
                )

        publication = publish_judge_evidence(
            request.evidence,
            plan=cast(JudgeMeasurementPlan, values["plan"]),
            measurements=cast(JudgeMeasurements, values["measurements"]),
            baseline_run=values["baseline_run"],
            subject_run=values["subject_run"],
            analysis_policy=cast(JudgeAnalysisPolicyDocument, values["policy"]),
            signing_key=checked_key,
            signer_identity=request.signer_identity if signing_key else None,
        )
        analysis = publication.analysis_result.to_dict()
        return JudgeWorkflowResult(
            {
                "format_version": "invarlock/judge-evaluation-result-v1",
                "kind": "judge",
                "ok": True,
                "evidence": str(publication.path),
                "authentication": "signed" if signing_key else "unsigned_local",
                "signer_identity": request.signer_identity if signing_key else None,
                "independent_verification": "not_performed",
                "decision": analysis["decision"],
                "analysis": analysis,
                **(
                    {"collection": collection_status}
                    if collection_status is not None
                    else {}
                ),
                "errors": [],
            }
        )
    except (OSError, ValueError) as exc:
        if isinstance(exc, JudgeWorkflowError):
            raise
        raise JudgeWorkflowError(str(exc)) from exc
