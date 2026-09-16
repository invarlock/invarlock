"""Native row integrity must survive the example's evaluator-specific projection."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "examples/evaluator-qualification/maintained/batch_semantics.py"
PROVIDERS = (
    "promptfoo",
    "evidently",
    "langfuse",
    "azure-ai-evaluation",
    "pydantic-evals",
)


@pytest.fixture
def semantics():
    spec = importlib.util.spec_from_file_location("batch_semantics_test", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cases():
    return [
        {
            "record_id": "one",
            "input": "same prompt",
            "output": " red",
            "reference": " red",
        },
        {
            "record_id": "two",
            "input": "same prompt",
            "output": "blue",
            "reference": "red",
        },
    ]


def native(provider, cases):
    rows = []
    for position, case in enumerate(cases):
        correct = case["output"] == case["reference"]
        if provider == "promptfoo":
            rows.append(
                {
                    "testCase": {
                        "description": case["record_id"],
                        "vars": {"output": case["output"]},
                        "assert": [{"type": "equals", "value": case["reference"]}],
                        "options": {},
                    },
                    "vars": {"output": case["output"]},
                    "provider": {"id": "echo"},
                    "prompt": {
                        "raw": case["output"],
                        "label": "{{output}}",
                        "config": {},
                    },
                    "testIdx": position,
                    "promptIdx": 0,
                    "response": {"output": case["output"]},
                    "score": float(correct),
                    "success": correct,
                    "gradingResult": {
                        "pass": correct,
                        "score": float(correct),
                        "componentResults": [
                            {
                                "pass": correct,
                                "score": float(correct),
                                "assertion": {
                                    "type": "equals",
                                    "value": case["reference"],
                                },
                            }
                        ],
                    },
                }
            )
        elif provider == "evidently":
            rows.append(
                {
                    "record_id": case["record_id"],
                    "output": case["output"],
                    "reference": case["reference"],
                    "exact_match": correct,
                }
            )
        elif provider == "langfuse":
            rows.append(
                {
                    "item": {
                        "input": case["input"],
                        "expected_output": case["reference"],
                        "metadata": {
                            "record_id": case["record_id"],
                            "output": case["output"],
                        },
                    },
                    "output": case["output"],
                    "evaluations": [
                        {
                            "name": "exact_match",
                            "value": correct,
                            "data_type": "BOOLEAN",
                        }
                    ],
                }
            )
        elif provider == "azure-ai-evaluation":
            rows.append(
                {
                    "inputs.record_id": case["record_id"],
                    "inputs.response": case["output"],
                    "inputs.ground_truth": case["reference"],
                    "outputs.exact_match.exact_match": float(correct),
                }
            )
        else:
            rows.append(
                {
                    "name": case["record_id"],
                    "inputs": {
                        "record_id": case["record_id"],
                        "input": case["input"],
                        "output": case["output"],
                    },
                    "expected_output": case["reference"],
                    "output": case["output"],
                    "assertions": {"exact_match": {"value": correct}},
                    "evaluator_failures": [],
                    "scores": {},
                    "labels": {},
                }
            )
    if provider == "promptfoo":
        return {"results": {"results": rows}}
    if provider == "langfuse":
        return {"item_results": rows, "run_evaluations": []}
    if provider == "pydantic-evals":
        return {"cases": rows, "failures": [], "report_evaluator_failures": []}
    return {"rows": rows}


def rows_for(provider, document):
    if provider == "promptfoo":
        return document["results"]["results"]
    return document[
        {"langfuse": "item_results", "pydantic-evals": "cases"}.get(provider, "rows")
    ]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_projects_ordered_native_rows_with_repeated_prompts(semantics, provider, cases):
    source = native(provider, cases)
    before = copy.deepcopy(source)
    scores, details = semantics.project(provider, cases, source)
    assert scores == [1.0, 0.0]
    assert [detail["native_row"] for detail in details] == rows_for(provider, source)
    assert source == before


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "reordered", "extra", "aggregate", "non_object"],
)
def test_native_row_cardinality_and_order_are_not_repaired(
    semantics, provider, cases, mutation
):
    source = native(provider, cases)
    rows = rows_for(provider, source)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "reordered":
        rows.reverse()
    elif mutation == "extra":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "non_object":
        rows[0] = 1.0
    else:
        source = {"metrics": {"exact_match": 0.5}}
    with pytest.raises(ValueError, match="native"):
        semantics.project(provider, cases, source)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("field", ["record_id", "output", "reference"])
def test_native_source_changes_fail(semantics, provider, cases, field):
    changed = copy.deepcopy(cases)
    changed[0][field] = "changed"
    with pytest.raises(ValueError, match="native"):
        semantics.project(provider, cases, native(provider, changed))


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("bad_score", [False, "1", 0.5, float("nan"), None])
def test_native_scores_are_typed_and_must_agree(semantics, provider, cases, bad_score):
    source = native(provider, cases)
    row = rows_for(provider, source)[0]
    if provider == "promptfoo":
        row["score"] = bad_score
    elif provider == "evidently":
        row["exact_match"] = bad_score
    elif provider == "langfuse":
        row["evaluations"][0]["value"] = bad_score
    elif provider == "azure-ai-evaluation":
        row["outputs.exact_match.exact_match"] = bad_score
    else:
        row["assertions"]["exact_match"]["value"] = bad_score
    with pytest.raises(ValueError, match="native"):
        semantics.project(provider, cases, source)


@pytest.mark.parametrize(
    "mutation",
    [
        "assertion_type",
        "assertion_extra",
        "vars",
        "success",
        "grading_score",
        "component_score",
        "component_assertion",
        "component_extra",
    ],
)
def test_promptfoo_assertion_and_native_detail_drift(semantics, cases, mutation):
    source = native("promptfoo", cases)
    row = rows_for("promptfoo", source)[0]
    if mutation == "assertion_type":
        row["testCase"]["assert"][0]["type"] = "contains"
    elif mutation == "assertion_extra":
        row["testCase"]["assert"].append({"type": "javascript", "value": "true"})
    elif mutation == "vars":
        row["testCase"]["vars"]["output"] = "changed"
    elif mutation == "success":
        row["success"] = False
    elif mutation == "grading_score":
        row["gradingResult"]["score"] = 0.0
    elif mutation == "component_score":
        row["gradingResult"]["componentResults"][0]["score"] = 0.0
    elif mutation == "component_assertion":
        row["gradingResult"]["componentResults"][0]["assertion"]["value"] = "changed"
    else:
        row["gradingResult"]["componentResults"].append({})
    with pytest.raises(ValueError, match="native"):
        semantics.project("promptfoo", cases, source)


@pytest.mark.parametrize(
    "provider", ["langfuse", "pydantic-evals", "azure-ai-evaluation", "evidently"]
)
def test_wrong_or_extra_metrics_cannot_substitute(semantics, cases, provider):
    source = native(provider, cases)
    row = rows_for(provider, source)[0]
    if provider == "langfuse":
        row["evaluations"][0]["name"] = "contains"
    elif provider == "pydantic-evals":
        row["assertions"]["contains"] = row["assertions"].pop("exact_match")
    elif provider == "azure-ai-evaluation":
        row["outputs.exact_match.contains"] = row.pop("outputs.exact_match.exact_match")
    else:
        row["contains"] = row.pop("exact_match")
    with pytest.raises(ValueError, match="native"):
        semantics.project(provider, cases, source)


@pytest.mark.parametrize("evaluations", [[], [{}, {}], None])
def test_langfuse_requires_exactly_one_per_record_evaluation(
    semantics, cases, evaluations
):
    source = native("langfuse", cases)
    source["item_results"][0]["evaluations"] = evaluations
    with pytest.raises(ValueError, match="exactly one metric"):
        semantics.project("langfuse", cases, source)


@pytest.mark.parametrize("field", ["record_id", "input", "output", "reference"])
def test_only_single_strings_and_unique_ids_are_supported(semantics, cases, field):
    cases[0][field] = ["multiple", "targets"]
    with pytest.raises(ValueError, match="case"):
        semantics.validate_cases(cases)


def test_unknown_profile_duplicate_and_empty_ids_fail(semantics, cases):
    with pytest.raises(ValueError, match="unsupported"):
        semantics.project("other", cases, {})
    cases[1]["record_id"] = cases[0]["record_id"]
    with pytest.raises(ValueError, match="unique"):
        semantics.validate_cases(cases)
    cases[0]["record_id"] = ""
    with pytest.raises(ValueError, match="record_id"):
        semantics.validate_cases(cases)


@pytest.mark.parametrize(
    "token",
    [
        "{{output}}",
        "{% print 'x' %}",
        "{# comment #}",
        "file://reference.txt",
        "package:module:function",
    ],
)
@pytest.mark.parametrize("field", ["output", "reference"])
def test_promptfoo_templating_and_file_references_are_explicitly_unsupported(
    semantics, cases, token, field
):
    cases[0][field] = token
    with pytest.raises(ValueError, match="unsupported Promptfoo"):
        semantics.validate_domain("promptfoo", cases)
    semantics.validate_domain("pydantic-evals", cases)


def test_promptfoo_captured_final_newline_is_rejected_before_rendering(
    semantics, cases
):
    cases[0]["output"] = cases[0]["reference"] = "answer\n"
    with pytest.raises(ValueError, match="output-final newline"):
        semantics.validate_domain("promptfoo", cases)
    semantics.validate_domain("evidently", cases)
