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
    EvaluationRequestError,
    _load_yaml,
    _reference_parts,
    _reject_include_directives,
    _resolve_output_reference,
)
from invarlock.evidence_pack_json import parse_json_bytes
from invarlock.judge_measurement_types import (
    JudgeAnalysisPolicyDocument,
    JudgeMeasurementPlan,
    JudgeMeasurements,
)
from invarlock.judge_measurements.analysis import decode_analysis_policy
from invarlock.judge_measurements.contracts import (
    JudgeMeasurementContractError,
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
        for source in inputs.values():
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
        )
    except (OSError, ValueError) as exc:
        if isinstance(exc, JudgeWorkflowError):
            raise
        raise JudgeWorkflowError(str(exc)) from exc


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
    except JudgeMeasurementContractError as exc:
        raise JudgeWorkflowError(str(exc)) from exc
    return budgets


def _prepare(request: JudgeEvaluationRequest) -> tuple[dict[str, Any], dict[str, Any]]:
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
    if missing:
        result["errors"].append(
            "Supply the missing request-relative inputs: " + ", ".join(missing)
        )
    if request.mode == "judge_collect":
        result["collection_integration"] = {
            "package": "invarlock-inspect-judge",
            "api": "invarlock_addins.inspect_judge.collect",
            "execution": "trusted_host_integration",
            "core_cli_execution": False,
        }
        result["next_action"] = (
            "Run the optional inspect-judge collect API with an explicitly "
            "constructed model, then import its retained measurements."
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
        if request.mode == "judge_collect":
            preflight["ok"] = preflight["ready"] = False
            preflight["errors"].append(
                "The core CLI does not execute provider calls. Run the optional inspect-judge collect API, then use a judge_import request."
            )
            raise JudgeWorkflowError(
                "Judge collection requires the optional trusted-host API",
                payload=preflight,
            )
        if not preflight["ready"]:
            raise JudgeWorkflowError("Judge evaluation is not ready", payload=preflight)
        if unsigned == (signing_key is not None):
            raise JudgeWorkflowError(
                "Choose either --signing-key or explicit --unsigned for judge evidence"
            )
        from invarlock.judge_measurements.evidence import publish_judge_evidence

        publication = publish_judge_evidence(
            request.evidence,
            plan=cast(JudgeMeasurementPlan, values["plan"]),
            measurements=cast(JudgeMeasurements, values["measurements"]),
            baseline_run=values["baseline_run"],
            subject_run=values["subject_run"],
            analysis_policy=cast(JudgeAnalysisPolicyDocument, values["policy"]),
            signing_key=signing_key,
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
                "errors": [],
            }
        )
    except (OSError, ValueError) as exc:
        if isinstance(exc, JudgeWorkflowError):
            raise
        raise JudgeWorkflowError(str(exc)) from exc
