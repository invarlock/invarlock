"""Strict native projections for separately versioned batch evaluator profiles."""

from __future__ import annotations

from typing import Any

PROVIDERS = (
    "promptfoo",
    "evidently",
    "langfuse",
    "azure-ai-evaluation",
    "pydantic-evals",
)


def validate_cases(cases: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for case in cases:
        if any(
            not isinstance(case.get(field), str)
            for field in ("record_id", "input", "output", "reference")
        ):
            raise ValueError("case fields must be single strings")
        record_id = case["record_id"]
        if not record_id:
            raise ValueError("case record_id must not be empty")
        if record_id in seen:
            raise ValueError("case record_id must be unique")
        seen.add(record_id)


def validate_domain(provider: str, cases: list[dict[str, Any]]) -> None:
    validate_cases(cases)
    if provider == "promptfoo":
        if any(case["output"].endswith("\n") for case in cases):
            raise ValueError("unsupported Promptfoo output-final newline rendering")
        if any(
            token in case[field]
            for case in cases
            for field in ("output", "reference")
            for token in ("{{", "{%", "{#", "file://", "package:")
        ):
            raise ValueError("unsupported Promptfoo template, file or package syntax")


def _equal(actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError("native identity, source, or scorer configuration changed")


def _score(value: object, expected: bool, *, boolean: bool = False) -> float:
    allowed = (bool,) if boolean else (int, float)
    if type(value) not in allowed or value != expected:
        raise ValueError("native score type or literal equality contradiction")
    return float(value)


def _promptfoo(case: dict[str, str], row: dict[str, Any], correct: bool) -> float:
    test = row["testCase"]
    assertion = {"type": "equals", "value": case["reference"]}
    _equal(test["description"], case["record_id"])
    _equal(test["vars"], {"output": case["output"]})
    _equal(test["assert"], [assertion])
    _equal(test["options"], {})
    _equal(row["vars"], test["vars"])
    _equal(row["provider"]["id"], "echo")
    _equal(row["prompt"], {"raw": case["output"], "label": "{{output}}", "config": {}})
    _equal(row["response"]["output"], case["output"])
    _score(row["success"], correct, boolean=True)
    grading = row["gradingResult"]
    _score(grading["pass"], correct, boolean=True)
    _score(grading["score"], correct)
    components = grading["componentResults"]
    if not isinstance(components, list) or len(components) != 1:
        raise ValueError("native assertion components must contain exactly one result")
    _equal(components[0]["assertion"], assertion)
    _score(components[0]["pass"], correct, boolean=True)
    _score(components[0]["score"], correct)
    return _score(row["score"], correct)


def _evidently(case: dict[str, str], row: dict[str, Any], correct: bool) -> float:
    _equal(set(row), {"record_id", "output", "reference", "exact_match"})
    for field in ("record_id", "output", "reference"):
        _equal(row[field], case[field])
    return _score(row["exact_match"], correct, boolean=True)


def _langfuse(case: dict[str, str], row: dict[str, Any], correct: bool) -> float:
    _equal(
        row["item"],
        {
            "input": case["input"],
            "expected_output": case["reference"],
            "metadata": {"output": case["output"], "record_id": case["record_id"]},
        },
    )
    _equal(row["output"], case["output"])
    evaluations = row["evaluations"]
    if not isinstance(evaluations, list) or len(evaluations) != 1:
        raise ValueError("native evaluations must contain exactly one metric")
    metric = evaluations[0]
    _equal(metric["name"], "exact_match")
    _equal(metric["data_type"], "BOOLEAN")
    return _score(metric["value"], correct, boolean=True)


def _azure(case: dict[str, str], row: dict[str, Any], correct: bool) -> float:
    _equal(
        set(row),
        {
            "inputs.record_id",
            "inputs.response",
            "inputs.ground_truth",
            "outputs.exact_match.exact_match",
        },
    )
    _equal(row["inputs.record_id"], case["record_id"])
    _equal(row["inputs.response"], case["output"])
    _equal(row["inputs.ground_truth"], case["reference"])
    return _score(row["outputs.exact_match.exact_match"], correct)


def _pydantic(case: dict[str, str], row: dict[str, Any], correct: bool) -> float:
    _equal(row["name"], case["record_id"])
    _equal(
        row["inputs"],
        {field: case[field] for field in ("record_id", "input", "output")},
    )
    _equal(row["expected_output"], case["reference"])
    _equal(row["output"], case["output"])
    _equal(row["evaluator_failures"], [])
    _equal(row["scores"], {})
    _equal(row["labels"], {})
    _equal(set(row["assertions"]), {"exact_match"})
    return _score(row["assertions"]["exact_match"]["value"], correct, boolean=True)


def project(
    provider: str, cases: list[dict[str, Any]], native: dict[str, Any]
) -> tuple[list[float], list[dict[str, Any]]]:
    """Reject ambiguous native batches; never repair their identities or order."""
    if provider not in PROVIDERS:
        raise ValueError("unsupported batch evaluator")
    validate_domain(provider, cases)
    try:
        if provider == "promptfoo":
            rows = native["results"]["results"]
        elif provider == "langfuse":
            _equal(native["run_evaluations"], [])
            rows = native["item_results"]
        elif provider == "pydantic-evals":
            _equal(native["failures"], [])
            _equal(native["report_evaluator_failures"], [])
            rows = native["cases"]
        else:
            rows = native["rows"]
        if not isinstance(rows, list) or len(rows) != len(cases):
            raise ValueError("native rows must cover exactly the scheduled records")
        projector = dict(
            zip(
                PROVIDERS,
                (_promptfoo, _evidently, _langfuse, _azure, _pydantic),
                strict=True,
            )
        )[provider]
        scores = []
        details = []
        for position, (case, row) in enumerate(zip(cases, rows, strict=True)):
            if not isinstance(row, dict):
                raise ValueError("native row must be an object")
            if provider == "promptfoo":
                _equal(row["testIdx"], position)
                _equal(row["promptIdx"], 0)
            scores.append(projector(case, row, case["output"] == case["reference"]))
            details.append({"native_row": row})
        return scores, details
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("native per-record result is missing or malformed") from exc
