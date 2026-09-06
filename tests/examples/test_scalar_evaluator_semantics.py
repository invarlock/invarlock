from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/evaluator-qualification"
PROVIDERS = [
    "lm-evaluation-harness",
    "deepeval",
    "ragas",
    "lighteval",
    "hugging-face-evaluate",
    "autoevals",
    "openevals",
    "openai-evals",
    "arize-phoenix-evals",
    "opik",
    "trulens",
]


@pytest.fixture
def semantics(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    spec = importlib.util.spec_from_file_location(
        "scalar_semantics_test", EXAMPLE / "maintained/scalar_semantics.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair(output="same", reference="same"):
    return {
        "record_id": "one",
        "input": "question",
        "output": output,
        "reference": reference,
    }


def result(provider, correct=True):
    value = float(correct)
    if provider == "openevals":
        return {"score": correct, "key": "exact_match"}
    if provider == "deepeval":
        return {
            "score": value,
            "metric_score": value,
            "successful": correct,
            "threshold": 1,
            "error": None,
        }
    if provider in {"autoevals", "opik"}:
        return {
            "score": value,
            "name": "ExactMatch" if provider == "autoevals" else "equals_metric",
            "error": None,
        }
    return {"score": value}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_typed_native_scalar_matches_literal_output(semantics, provider):
    assert semantics.validate_result(provider, pair(), result(provider)) == 1.0
    assert (
        semantics.validate_result(provider, pair("a", "b"), result(provider, False))
        == 0.0
    )


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("bad", ["1", None, float("nan"), 0.5, False])
def test_native_scalar_contradiction_or_coercion_cannot_gain_authority(
    semantics, provider, bad
):
    native = result(provider)
    native["score"] = bad
    with pytest.raises(ValueError, match="native"):
        semantics.validate_result(provider, pair(), native)


@pytest.mark.parametrize(
    "provider,field,value",
    [
        ("deepeval", "metric_score", 0.0),
        ("deepeval", "successful", False),
        ("deepeval", "threshold", 0.0),
        ("deepeval", "threshold", True),
        ("deepeval", "error", "failed"),
        ("autoevals", "error", "failed"),
        ("autoevals", "name", "similarity"),
        ("opik", "name", "judge"),
        ("openevals", "key", "contains"),
    ],
)
def test_native_detail_and_configuration_contradictions_fail(
    semantics, provider, field, value
):
    native = result(provider)
    native[field] = value
    with pytest.raises(ValueError, match="native"):
        semantics.validate_result(provider, pair(), native)


@pytest.mark.parametrize(
    "provider,output,reference",
    [
        ("deepeval", " answer", "answer"),
        ("deepeval", "answer", " answer"),
        ("deepeval", "", ""),
        ("lighteval", "", ""),
        ("hugging-face-evaluate", "a\0", "a"),
        ("hugging-face-evaluate", "\0", ""),
    ],
)
def test_upstream_pair_collisions_and_empty_special_cases_are_rejected(
    semantics, provider, output, reference
):
    with pytest.raises(ValueError, match="unsupported"):
        semantics.validate_pair(provider, pair(output, reference))


@pytest.mark.parametrize(
    "provider,output,reference",
    [
        ("deepeval", " answer", " answer"),
        ("deepeval", " ", " "),
        ("lighteval", "", "different"),
        ("lm-evaluation-harness", "a\0", "a"),
        ("hugging-face-evaluate", "a\0", "a\0"),
        ("opik", "Answer", "answer"),
    ],
)
def test_supported_current_domain_preserves_equal_boundaries_and_literal_misses(
    semantics, provider, output, reference
):
    semantics.validate_pair(provider, pair(output, reference))


@pytest.mark.parametrize("field", ["input", "output", "reference", "record_id"])
def test_non_string_or_multiple_targets_are_not_repaired(semantics, field):
    case = pair()
    case[field] = ["first", "second"]
    with pytest.raises(ValueError, match="case"):
        semantics.validate_pair("lm-evaluation-harness", case)


def test_missing_native_shape_and_unknown_evaluator_fail(semantics):
    with pytest.raises(ValueError, match="native"):
        semantics.validate_result("opik", pair(), {})
    with pytest.raises(ValueError, match="unsupported"):
        semantics.validate_pair("unknown", pair())
