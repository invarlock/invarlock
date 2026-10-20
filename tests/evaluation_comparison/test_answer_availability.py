"""Captured missingness preserves the schedule and legitimate zero scores."""

from copy import deepcopy

import pytest

from invarlock.core.scoring import UNICODE_VERSION, score
from invarlock.engine import compare_runs, evaluator_input_capabilities, make_run
from invarlock.evaluation_comparison.comparison import metric_summaries
from tests.evaluation_comparison.test_likelihood import policy as nll_policy
from tests.evaluation_comparison.test_likelihood import row as nll_row
from tests.evaluation_comparison.test_likelihood import run as nll_run


def policy(kind="exact_match"):
    configuration = (
        {"unicode_version": UNICODE_VERSION}
        if kind in ("normalized_match", "token_f1")
        else {"fields": ["/answer"]}
        if kind == "json_fields"
        else {}
    )
    return {
        "format": "invarlock/comparison-policy-v1",
        "metrics": [
            {
                "name": kind,
                "kind": kind,
                "configuration": configuration,
                "direction": "higher",
                "unit": "score",
                "aggregation": "mean",
                "minimum_count": 1,
                "maximum_regression": 1,
                "maximum_interval_width": 2,
            }
        ],
        "slices": [],
    }


def run(output, *, expected="answer", error=None):
    return make_run(
        [
            {
                "id": str(i),
                "input": "question",
                "expected": expected,
                "output": output,
                "error": error,
                "metadata": {"group": str(i)},
            }
            for i in range(2)
        ],
        source={"name": "fixture", "version": "1"},
        run_id="answers",
        artifact_digest="sha256:" + "1" * 64,
    )


@pytest.mark.parametrize("kind", ["exact_match", "normalized_match", "token_f1"])
@pytest.mark.parametrize("output", [None, 0, False, {}, []])
@pytest.mark.parametrize("side", ["baseline", "subject", "both"])
def test_text_metrics_require_text_on_both_sides(kind, output, side):
    baseline, subject = run("answer"), run("answer")
    if side in ("baseline", "both"):
        baseline = run(output)
    if side in ("subject", "both"):
        subject = run(output)
    before = deepcopy((baseline, subject))
    selected = policy(kind)
    selected["slices"] = [{"name": "first", "where": {"group": "0"}}]
    result = compare_runs(baseline, subject, selected)
    assert result["decision"] == "insufficient_evidence"
    for metric, ids in zip(result["metrics"], [["0", "1"], ["0"]], strict=True):
        assert metric["count"] == len(ids)
        assert metric["missing_ids"] == ids
        assert metric["subject_mean"] is None
        assert metric["interval"] is None
    assert all(s["usable_count"] == 0 for s in metric_summaries(result))
    for value in (baseline, subject):
        available = evaluator_input_capabilities(value)["exact_match"]
        assert available["usable_count"] == (
            2 if value["records"][0]["output"] == "answer" else 0
        )
    assert (baseline, subject) == before


@pytest.mark.parametrize(
    "kind",
    [
        "exact_match",
        "normalized_match",
        "token_f1",
        "numeric_tolerance",
        "json_exact",
        "json_fields",
    ],
)
@pytest.mark.parametrize("side", ["baseline", "subject"])
def test_explicit_errors_are_missing_for_every_deterministic_metric(kind, side):
    baseline, subject = run("answer"), run("answer")
    (baseline if side == "baseline" else subject)["records"][0]["error"] = "timeout"
    # References for all non-errored rows must still be valid.
    expected = (
        1
        if kind == "numeric_tolerance"
        else {"answer": 1}
        if kind.startswith("json")
        else "answer"
    )
    for value in (baseline, subject):
        for row in value["records"]:
            row["expected"] = expected
            row["output"] = expected
    result = compare_runs(baseline, subject, policy(kind))
    assert result["decision"] == "insufficient_evidence"
    assert result["metrics"][0]["missing_ids"] == ["0"]
    assert metric_summaries(result)[0]["usable_count"] == 1


@pytest.mark.parametrize(
    "kind,expected,output,value",
    [
        ("exact_match", "", "", 1),
        ("exact_match", "answer", "", 0),
        ("exact_match", "answer", "wrong", 0),
        ("normalized_match", "", "", 1),
        ("normalized_match", "answer", "", 0),
        ("token_f1", "", "", 1),
        ("token_f1", "answer", "", 0),
        ("numeric_tolerance", 1, "", 0),
        ("numeric_tolerance", 1, "not numeric", 0),
        ("numeric_tolerance", 1, 0, 0),
        ("json_exact", "null", None, 1),
        ("json_exact", {"answer": 1}, None, 0),
        ("json_exact", {"answer": None}, {"answer": None}, 1),
        ("json_exact", {"answer": 1}, "not JSON", 0),
        ("json_fields", {"answer": 1}, None, 0),
        ("json_fields", {"answer": None}, {"answer": None}, 1),
        ("json_fields", {"answer": 1}, {}, 0),
    ],
)
def test_present_answers_remain_scoreable(kind, expected, output, value):
    captured = run(output, expected=expected)
    selected = policy(kind)
    result = compare_runs(captured, captured, selected)
    assert result["decision"] == "pass"
    assert result["metrics"][0]["missing_ids"] == []
    assert result["metrics"][0]["subject_mean"] == value
    assert (
        score(kind, expected, output, selected["metrics"][0]["configuration"]) == value
    )


def test_numeric_null_is_missing():
    captured = run(None, expected=1)
    result = compare_runs(captured, captured, policy("numeric_tolerance"))
    assert result["decision"] == "insufficient_evidence"
    assert result["metrics"][0]["missing_ids"] == ["0", "1"]


def test_missing_answers_are_charged_before_scoring():
    from invarlock.evaluation_comparison.capacity import check_missing_id_capacity
    from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError

    captured = run(None)
    selected = [(row, row) for row in captured["records"]]
    with pytest.raises(EvaluationRecordsError, match="missing-ID arrays"):
        check_missing_id_capacity(
            [("overall", selected)], policy()["metrics"], byte_limit=2
        )


def test_complete_slice_still_scores_wrong_answers_when_overall_is_missing():
    captured = run("wrong")
    captured["records"][1]["output"] = None
    selected = policy()
    selected["slices"] = [{"name": "complete", "where": {"group": "0"}}]
    result = compare_runs(captured, captured, selected)
    assert result["metrics"][0]["decision"] == "insufficient_evidence"
    assert result["metrics"][0]["missing_ids"] == ["1"]
    assert result["metrics"][1]["decision"] == "pass"
    assert result["metrics"][1]["subject_mean"] == 0


def test_likelihood_only_rows_are_usable_only_for_metrics_with_required_facts():
    captured = nll_run([nll_row("a"), nll_row("b")])
    selected = nll_policy()
    selected["metrics"] += policy()["metrics"]
    result = compare_runs(captured, captured, selected)
    assert result["metrics"][0]["decision"] == "pass"
    assert result["metrics"][1]["decision"] == "insufficient_evidence"
    assert result["metrics"][1]["missing_ids"] == ["a", "b"]


def test_recorded_zero_does_not_require_an_answer():
    from invarlock.evaluation_records.templates import example_project

    baseline, subject, selected = example_project("judge")
    selected["metrics"] = selected["metrics"][:1]
    for captured in (baseline, subject):
        for row in captured["records"]:
            row["output"] = None
            row["scores"][selected["metrics"][0]["score_key"]] = 0
    selected["metrics"][0].pop("subject_minimum", None)
    selected["metrics"][0].update(maximum_regression=1, maximum_interval_width=2)
    result = compare_runs(baseline, subject, selected)
    assert result["decision"] == "pass"
    assert all(
        m["subject_mean"] == 0 and m["missing_ids"] == [] for m in result["metrics"]
    )
