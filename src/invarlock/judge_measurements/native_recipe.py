"""Predeclare native judging decisions and derive answer-dependent identities."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, cast

from jsonschema import Draft202012Validator

from invarlock.evaluation_records.cases import case_set_digest
from invarlock.evaluation_records.io import run_digest
from invarlock.evidence_pack_json import parse_json_bytes
from invarlock.judge_measurement_types import (
    JudgeAnalysisPolicyDocument,
    JudgeMeasurementPlan,
)
from invarlock.judge_measurements.analysis import decode_analysis_policy
from invarlock.judge_measurements.contracts import (
    _validate_frozen_answer_bindings,
    canonical_payload,
    measurement_plan_digest,
    render_judge_request,
    validate_measurement_plan,
)
from invarlock.public_contracts import (
    load_judge_analysis_policy_schema,
    load_judge_measurement_plan_schema,
)

NATIVE_POLICY_FORMAT = "invarlock/native-judge-policy-v1"
NATIVE_POLICY_MAX_BYTES = 4 * 1024 * 1024
_DERIVED_PLAN = {
    "case_set_sha256",
    "baseline_run_sha256",
    "subject_run_sha256",
    "answer_bindings",
}
_PLAN_FIELDS = {
    "format",
    "profile_id",
    "rubric",
    "prompt",
    "judge",
    "parser",
    "scale",
    "sampling",
    "schedule",
}


def _object(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} must contain exactly: {', '.join(sorted(fields))}")
    return cast(dict[str, Any], value)


def _recipe(value: object) -> dict[str, Any]:
    raw = canonical_payload(value)
    if len(raw) > NATIVE_POLICY_MAX_BYTES:
        raise ValueError("native judge policy exceeds its 4 MiB limit")
    result = _object(
        parse_json_bytes(raw, label="native judge policy"),
        {"format", "plan", "analysis", "collection", "runner"},
        "native judge policy",
    )
    if result["format"] != NATIVE_POLICY_FORMAT:
        raise ValueError("unsupported native judge policy format")
    plan = _object(result["plan"], _PLAN_FIELDS, "native judge plan template")
    _object(plan["rubric"], {"text"}, "native judge rubric")
    plan_schema = copy.deepcopy(load_judge_measurement_plan_schema())
    for name in _DERIVED_PLAN:
        plan_schema["properties"].pop(name)
        plan_schema["required"].remove(name)
    for parent, name in (("rubric", "sha256"), ("schedule", "expected_trials")):
        plan_schema["properties"][parent]["properties"].pop(name)
        plan_schema["properties"][parent]["required"].remove(name)
    policy_schema = copy.deepcopy(load_judge_analysis_policy_schema())
    policy_schema["properties"].pop("plan_sha256")
    policy_schema["required"].remove("plan_sha256")
    for schema, item, label in (
        (plan_schema, plan, "native judge plan template"),
        (policy_schema, result["analysis"], "native judge analysis template"),
    ):
        error = next(Draft202012Validator(schema).iter_errors(item), None)
        if error is not None:
            raise ValueError(f"{label} is invalid: {error.message[:240]}")
    runner = _object(
        result["runner"], {"scorer_id", "invocation_timeout_seconds"}, "judge runner"
    )
    scorer = runner["scorer_id"]
    if (
        not isinstance(scorer, str)
        or not 1 <= len(scorer) <= 128
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for c in scorer
        )
    ):
        raise ValueError("judge scorer_id must be a bounded identifier")
    timeout = runner["invocation_timeout_seconds"]
    if type(timeout) is not int or not 1 <= timeout <= 604800:
        raise ValueError(
            "judge invocation timeout must be between 1 and 604800 seconds"
        )
    return result


def finalize_native_plan(
    recipe: dict[str, Any], baseline_run: dict[str, Any], subject_run: dict[str, Any]
) -> tuple[JudgeMeasurementPlan, JudgeAnalysisPolicyDocument]:
    """Fill only derived hashes and counts; all measurement choices stay frozen."""
    recipe = _recipe(recipe)
    plan = recipe["plan"]
    # Validate complete run shape and uniqueness before constructing ID maps.
    baseline_digest, subject_digest = run_digest(baseline_run), run_digest(subject_run)
    plan["baseline_run_sha256"] = baseline_digest
    plan["subject_run_sha256"] = subject_digest
    plan["case_set_sha256"] = case_set_digest(
        {
            "format": "invarlock/evaluation-case-set-v1",
            "cases": [
                {key: row[key] for key in ("id", "input", "expected", "metadata")}
                for row in baseline_run["records"]
            ],
        }
    )
    rubric_text = cast(str, plan["rubric"]["text"])
    plan["rubric"]["sha256"] = hashlib.sha256(rubric_text.encode("utf-8")).hexdigest()
    repetitions = plan["schedule"]["repetitions"]
    if type(repetitions) is not int:
        raise ValueError("judge repetitions must use an integer value")
    plan["schedule"]["expected_trials"] = len(baseline_run["records"]) * 2 * repetitions
    sides = {
        "baseline": {row["id"]: row for row in baseline_run["records"]},
        "subject": {row["id"]: row for row in subject_run["records"]},
    }
    if sides["baseline"].keys() != sides["subject"].keys():
        raise ValueError("native judge runs must contain identical case membership")
    bindings = []
    for case_id in sorted(sides["baseline"]):
        binding = {"case_id": case_id}
        for side, rows in sides.items():
            row = rows[case_id]
            if not isinstance(row["output"], str) or row["error"] is not None:
                raise ValueError(
                    "native judging requires complete successful text answers"
                )
            request = render_judge_request(
                cast(JudgeMeasurementPlan, plan),
                input_text=row["input"],
                answer_text=row["output"],
            )
            binding[f"{side}_answer_sha256"] = hashlib.sha256(
                row["output"].encode("utf-8")
            ).hexdigest()
            binding[f"{side}_request_sha256"] = hashlib.sha256(request).hexdigest()
        bindings.append(binding)
    plan["answer_bindings"] = bindings
    finalized = cast(JudgeMeasurementPlan, plan)
    validate_measurement_plan(finalized)
    _validate_frozen_answer_bindings(finalized, baseline_run, subject_run)
    policy = cast(JudgeAnalysisPolicyDocument, recipe["analysis"])
    policy["plan_sha256"] = measurement_plan_digest(finalized)
    decode_analysis_policy(policy, plan=finalized)
    return finalized, policy


def prepare_native_judge(
    policy_bytes: bytes, schedule: dict[str, Any]
) -> dict[str, Any]:
    """Validate choices and reserve a complete plan before generating any answers."""
    if len(policy_bytes) > NATIVE_POLICY_MAX_BYTES:
        raise ValueError("native judge policy exceeds its 4 MiB limit")
    recipe = _recipe(parse_json_bytes(policy_bytes, label="native judge policy"))
    records = []
    for row in schedule["records"]:
        if len(row["input_parts"]) != 1 or any(
            part["kind"] != "text" for part in row["input_parts"]
        ):
            raise ValueError(
                "native judging requires exactly one text part per scheduled input"
            )
        records.append(
            {
                "id": row["record_id"],
                "input": "\n".join(part["text"] for part in row["input_parts"]),
                "expected": row["expected_output"],
                "metadata": {},
                "context": {},
                "output": "",
                "error": None,
                "scores": {},
            }
        )
    preview = {
        "format": "invarlock/evaluation-run-v1",
        "run_id": "preflight",
        "artifact_digest": "sha256:" + "0" * 64,
        "source": {"name": "invarlock-native-preflight", "version": "1"},
        "source_digest": None,
        "records": records,
        "score_provenance": {},
    }
    plan, policy = finalize_native_plan(recipe, preview, preview)
    from invarlock.judge_measurements.workflow import _collection_budgets

    budgets = _collection_budgets(recipe["collection"], plan)
    planned = plan["schedule"]["expected_trials"]
    capacity = min(
        budgets["max_calls"],
        budgets["max_input_tokens"] // budgets["input_tokens_per_call"],
        budgets["max_output_tokens"] // plan["judge"]["config"]["max_output_tokens"],
        budgets["max_cost_microusd"] // budgets["cost_microusd_per_call"],
    )
    if capacity < planned:
        raise ValueError("judge collection budgets must reserve every planned call")
    units = len({unit["unit_id"] for unit in plan["sampling"]["case_units"]})
    if units < policy["minimum_units"]:
        raise ValueError(
            "judge schedule has fewer independent units than its analysis minimum"
        )
    return {
        "recipe": recipe,
        "cases": len(records),
        "independent_units": units,
        "minimum_units": policy["minimum_units"],
        "planned_trials": planned,
        "budgets": budgets,
        "maximum_admitted_calls": capacity,
        "judge": plan["judge"],
        "metric_name": policy["metric_name"],
    }
