"""Native policy decisions are validated before answer collection and hash finalization."""

import copy

import pytest

from invarlock.judge_measurements.contracts import canonical_payload
from invarlock.judge_measurements.native_capture import validate_native_capture
from invarlock.judge_measurements.native_recipe import (
    finalize_native_plan,
    prepare_native_judge,
)
from tests.judge_measurements.test_native_capture import _capture


@pytest.fixture
def frozen(tmp_path):
    return _capture(tmp_path)


def test_finalization_is_deterministic_and_preserves_authored_choices(frozen):
    recipe = copy.deepcopy(frozen["recipe"])
    runs = validate_native_capture(frozen)
    plan, policy = finalize_native_plan(recipe, *runs)
    assert (plan, policy) == finalize_native_plan(recipe, *runs)
    assert recipe == frozen["recipe"]
    assert plan["judge"] == recipe["plan"]["judge"]
    assert plan["schedule"]["expected_trials"] == 4
    assert (
        prepare_native_judge(canonical_payload(recipe), frozen["schedule"])[
            "planned_trials"
        ]
        == 4
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(unknown=True),
        lambda r: r.update(format="other"),
        lambda r: r["plan"].update(answer_bindings=[]),
        lambda r: r["plan"]["rubric"].update(sha256="0" * 64),
        lambda r: r["plan"]["schedule"].update(expected_trials=4),
        lambda r: r["plan"]["schedule"].update(repetitions=True),
        lambda r: r["plan"]["prompt"].pop("system"),
        lambda r: r["analysis"].update(plan_sha256="0" * 64),
        lambda r: r["runner"].update(scorer_id="bad scorer"),
        lambda r: r["runner"].update(invocation_timeout_seconds=True),
        lambda r: r["collection"].update(max_calls=3),
        lambda r: r["collection"].update(max_output_tokens=100),
        lambda r: r["analysis"].update(minimum_units=3),
        lambda r: r["plan"]["sampling"]["case_units"].pop(),
    ],
)
def test_invalid_or_retroactively_filled_decisions_fail_preflight(frozen, mutate):
    recipe = copy.deepcopy(frozen["recipe"])
    mutate(recipe)
    with pytest.raises(ValueError):
        prepare_native_judge(canonical_payload(recipe), frozen["schedule"])


def test_text_profile_rejects_ambiguous_prompt_projection(frozen):
    row = frozen["schedule"]["records"][0]
    row["input_parts"].append(copy.deepcopy(row["input_parts"][0]))
    with pytest.raises(ValueError, match="exactly one text part"):
        prepare_native_judge(canonical_payload(frozen["recipe"]), frozen["schedule"])


def test_changed_answers_are_bound_and_failed_answers_cannot_be_judged(frozen):
    baseline, subject = validate_native_capture(frozen)
    original = finalize_native_plan(frozen["recipe"], baseline, subject)[0]
    subject["records"][0]["output"] = "changed answer"
    changed = finalize_native_plan(frozen["recipe"], baseline, subject)[0]
    assert changed["subject_run_sha256"] != original["subject_run_sha256"]
    assert (
        changed["answer_bindings"][0]["subject_answer_sha256"]
        != original["answer_bindings"][0]["subject_answer_sha256"]
    )
    subject["records"][0]["output"] = None
    with pytest.raises(ValueError):
        finalize_native_plan(frozen["recipe"], baseline, subject)


def test_policy_size_is_bounded_before_parsing(frozen):
    from invarlock.judge_measurements.native_recipe import NATIVE_POLICY_MAX_BYTES

    with pytest.raises(ValueError, match="4 MiB"):
        prepare_native_judge(b" " * (NATIVE_POLICY_MAX_BYTES + 1), frozen["schedule"])


def test_finalization_rejects_oversized_recipe_and_changed_membership(frozen):
    baseline, subject = validate_native_capture(frozen)
    oversized = copy.deepcopy(frozen["recipe"])
    oversized["plan"]["rubric"]["text"] = "a" * (4 * 1024 * 1024)
    with pytest.raises(ValueError, match="4 MiB"):
        finalize_native_plan(oversized, baseline, subject)
    subject["records"].pop()
    with pytest.raises(ValueError, match="identical case membership"):
        finalize_native_plan(frozen["recipe"], baseline, subject)


def test_fractional_integer_encoding_is_rejected(frozen):
    recipe = copy.deepcopy(frozen["recipe"])
    recipe["plan"]["schedule"]["repetitions"] = 1.0
    with pytest.raises(ValueError, match="integer value"):
        prepare_native_judge(canonical_payload(recipe), frozen["schedule"])
