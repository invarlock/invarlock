"""Opt-in per-case references are distinct, bounded and bound into judge requests."""

from __future__ import annotations

import copy
import json

import pytest

from invarlock.evaluation_comparison.comparison import make_run
from invarlock.evaluation_records.cases import case_set_digest
from invarlock.evaluation_records.io import run_digest
from invarlock.judge_measurements import contracts
from invarlock.judge_measurements.captured_workflow import prepare_evaluator_judge
from tests.core.test_native_judge_transaction import _recipe
from tests.judge_measurements.test_contracts import _plan


def _reference_run(expected="gold reference"):
    return make_run(
        [
            {
                "id": case,
                "input": f"question {case}",
                "expected": expected,
                "output": "candidate answer",
            }
            for case in ("one", "two")
        ],
        source={"name": "existing-evaluator", "version": "1"},
        run_id="existing",
        artifact_digest="sha256:" + "a" * 64,
    )


def _reference_recipe():
    recipe = _recipe()
    recipe["plan"]["prompt"]["reference_mode"] = "per_case"
    return recipe


def test_reference_is_explicit_separate_and_changes_actual_request_digest():
    run = _reference_run()
    plan, _ = prepare_evaluator_judge(_reference_recipe(), run, run)
    changed = _reference_run("different gold")
    other_plan, _ = prepare_evaluator_judge(_reference_recipe(), changed, changed)
    assert plan["answer_bindings"] != other_plan["answer_bindings"]
    row = run["records"][0]
    request = json.loads(
        contracts.render_judge_request(
            plan,
            input_text=row["input"],
            answer_text=row["output"],
            reference_text=row["expected"],
        )
    )
    message = json.loads(request["messages"][-1]["content"])
    assert message["input"] == "question one"
    assert message["answer"] == "candidate answer"
    assert message["reference"] == "gold reference"
    assert row["input"] == "question one"


@pytest.mark.parametrize("mode", [None, "none"])
def test_legacy_or_explicit_none_keeps_historical_rendering_bytes(mode):
    plan = _plan()
    original = contracts.render_judge_request(
        plan, input_text="question", answer_text="answer"
    )
    if mode is not None:
        plan["prompt"]["reference_mode"] = mode
    actual = contracts.render_judge_request(
        plan,
        input_text="question",
        answer_text="answer",
        reference_text={"ignored": "legacy"},
    )
    assert actual == original
    assert "reference" not in json.loads(json.loads(actual)["messages"][-1]["content"])


@pytest.mark.parametrize("reference", [None, {}, [], 1, True])
def test_per_case_mode_rejects_missing_or_structured_reference_before_execution(
    reference,
):
    run = _reference_run(reference)
    with pytest.raises(ValueError, match="string reference"):
        prepare_evaluator_judge(_reference_recipe(), run, run)


def test_empty_reference_is_real_text_and_schema_rejects_unknown_mode():
    run = _reference_run("")
    plan, _ = prepare_evaluator_judge(_reference_recipe(), run, run)
    request = json.loads(
        contracts.render_judge_request(
            plan, input_text="q", answer_text="a", reference_text=""
        )
    )
    assert json.loads(request["messages"][-1]["content"])["reference"] == ""
    plan["prompt"]["reference_mode"] = "guess"
    with pytest.raises(ValueError, match="reference_mode"):
        contracts.validate_measurement_plan(plan)
    with pytest.raises(ValueError, match="unsupported judge reference mode"):
        contracts.render_judge_request(plan, input_text="q", answer_text="a")


def test_reference_is_bounded_independently_and_with_other_request_material():
    plan = _plan()
    plan["prompt"]["reference_mode"] = "per_case"
    with pytest.raises(ValueError, match="per-case judge reference.*exceeds"):
        contracts.render_judge_request(
            plan,
            input_text="q",
            answer_text="a",
            reference_text="x" * contracts.JUDGE_REQUEST_MAX_BYTES,
        )
    with pytest.raises(ValueError, match="normalized judge request.*exceeds"):
        contracts.render_judge_request(
            plan,
            input_text="q" * (contracts.JUDGE_REQUEST_MAX_BYTES // 2),
            answer_text="a",
            reference_text="r" * (contracts.JUDGE_REQUEST_MAX_BYTES // 2),
        )


def test_reference_does_not_leak_into_demonstration_inputs():
    plan = _plan()
    plan["prompt"]["reference_mode"] = "per_case"
    plan["prompt"]["demonstrations"] = [
        {"input": "demo task", "answer": "demo answer", "rating": "correct"}
    ]
    request = json.loads(
        contracts.render_judge_request(
            plan,
            input_text="actual task",
            answer_text="actual answer",
            reference_text="actual gold",
        )
    )
    users = [
        json.loads(message["content"])
        for message in request["messages"]
        if message["role"] == "user"
    ]
    assert "reference" not in users[0]
    assert users[-1]["reference"] == "actual gold"


def test_rebinding_run_hashes_does_not_hide_changed_or_dropped_reference_request():
    run = _reference_run()
    plan, _ = prepare_evaluator_judge(_reference_recipe(), run, run)
    changed = _reference_run("replacement gold")
    changed_plan = copy.deepcopy(plan)
    changed_plan["baseline_run_sha256"] = run_digest(changed)
    changed_plan["subject_run_sha256"] = run_digest(changed)
    changed_plan["case_set_sha256"] = case_set_digest(
        {
            "format": "invarlock/evaluation-case-set-v1",
            "cases": [
                {key: row[key] for key in ("id", "input", "expected", "metadata")}
                for row in changed["records"]
            ],
        }
    )
    with pytest.raises(ValueError, match="request binding does not match"):
        contracts._validate_frozen_answer_bindings(changed_plan, changed, changed)
    legacy_plan = copy.deepcopy(plan)
    legacy_plan["prompt"].pop("reference_mode")
    with pytest.raises(ValueError, match="request binding does not match"):
        contracts._validate_frozen_answer_bindings(legacy_plan, run, run)
