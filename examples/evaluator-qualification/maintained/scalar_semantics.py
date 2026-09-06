"""Declared scalar scorer domains and strict native value/detail checks."""

from __future__ import annotations

from numbers import Real
from typing import Any

from maintained.batch_semantics import validate_cases

CONFIGURATIONS = {
    "lm-evaluation-harness": {
        "input_dtype": "object",
        "regexes_to_ignore": None,
        "ignore_case": False,
        "ignore_punctuation": False,
        "ignore_numbers": False,
    },
    "deepeval": {"threshold": 1, "verbose_mode": False},
    "ragas": {"metric": "ExactMatch"},
    "lighteval": {
        "strip_strings": False,
        "normalize_pred": None,
        "normalize_gold": None,
        "type_exact_match": "full",
        "references": 1,
        "predictions": 1,
    },
    "hugging-face-evaluate": {
        "metric": "exact_match",
        "regexes_to_ignore": None,
        "ignore_case": False,
        "ignore_punctuation": False,
        "ignore_numbers": False,
    },
    "autoevals": {"metric": "ExactMatch"},
    "openevals": {"key": "exact_match", "input_type": "string"},
    "openai-evals": {"match_function": "exact"},
    "arize-phoenix-evals": {"metric": "exact_match"},
    "opik": {"case_sensitive": True, "name": "equals_metric", "track": False},
    "trulens": {
        "name": "exact_match",
        "implementation": "runners.trulens_metric.exact_match",
    },
}


def validate_pair(provider: str, case: dict[str, Any]) -> None:
    if provider not in CONFIGURATIONS:
        raise ValueError("unsupported scalar evaluator")
    validate_cases([case])
    output, reference = case["output"], case["reference"]
    if provider == "deepeval" and (
        output == "" or (output == reference) != (output.strip() == reference.strip())
    ):
        raise ValueError("unsupported DeepEval empty output or whitespace collision")
    if provider == "lighteval" and output == reference == "":
        raise ValueError("unsupported LightEval empty-pair semantics")
    if provider == "hugging-face-evaluate" and (output == reference) != (
        output.rstrip("\0") == reference.rstrip("\0")
    ):
        raise ValueError("unsupported Evaluate trailing-NUL collision")


def _numeric(value: object, expected: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value != expected:
        raise ValueError("native numeric score type or literal equality contradiction")
    return float(value)


def validate_result(
    provider: str, case: dict[str, Any], result: dict[str, Any]
) -> float:
    validate_pair(provider, case)
    expected = case["output"] == case["reference"]
    return validate_native(provider, result, expected)


def validate_native(provider: str, result: dict[str, Any], expected: bool) -> float:
    try:
        if provider == "openevals":
            if (
                result["key"] != "exact_match"
                or type(result["score"]) is not bool
                or result["score"] != expected
            ):
                raise ValueError("native OpenEvals key or Boolean score contradiction")
            return float(result["score"])
        score = _numeric(result["score"], expected)
        if provider == "deepeval":
            _numeric(result["metric_score"], expected)
            if (
                type(result["successful"]) is not bool
                or result["successful"] != expected
                or isinstance(result["threshold"], bool)
                or result["threshold"] != 1
                or result["error"] is not None
            ):
                raise ValueError("native DeepEval status or threshold contradiction")
        elif provider in ("autoevals", "opik"):
            expected_name = "ExactMatch" if provider == "autoevals" else "equals_metric"
            if result["name"] != expected_name:
                raise ValueError("native metric name changed")
            if provider == "autoevals" and result["error"] is not None:
                raise ValueError("native AutoEvals score carries an error")
        return score
    except (KeyError, TypeError) as exc:
        raise ValueError("native scalar result is missing or malformed") from exc
