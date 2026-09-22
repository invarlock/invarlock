"""Serialize retained measurements through the pinned scalar evaluator SDKs.

This helper exercises SDK data containers and serialization, never model calls
or new grading. Some SDKs expose only metric result types: their original case
arguments remain explicit wrapper fields. Evaluate uses its public ``compute``
API to roundtrip nullable prediction/reference batches through Arrow; the local
metric returns those columns unchanged and computes no score.
"""

from __future__ import annotations

import importlib.metadata
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from invarlock.evaluation_records.scalar_integrations import (
    export_records,
    serialize_results,
)

VERSIONS = {
    "deepeval": "4.1.3",
    "ragas": "0.4.3",
    "lighteval": "0.13.0",
    "hugging-face-evaluate": "0.4.6",
    "autoevals": "0.3.0",
    "openevals": "0.2.0",
    "arize-phoenix-evals": "3.3.0",
    "opik": "2.2.7",
}


def roundtrip(evaluator: str, payload: Any, *, tmp_path: Path) -> list[dict]:
    """Return JSON after actual SDK construction and dedicated serialization.

    The caller supplies the pinned SDK environment and network isolation. The
    input is the native-shaped retained payload from ``native_shapes.payload``.
    SDK defaults may add context, but original fields and every canonical case
    fact must survive. In particular, likelihood-only cases keep a null output.
    """
    if evaluator not in VERSIONS:
        raise ValueError(f"unsupported scalar SDK: {evaluator}")
    package = "evaluate" if evaluator == "hugging-face-evaluate" else evaluator
    installed = importlib.metadata.version(package)
    if installed != VERSIONS[evaluator]:
        raise ValueError(
            f"requires {package}=={VERSIONS[evaluator]}, found {installed}"
        )
    original = export_records(evaluator, payload)
    native = deepcopy(payload)
    tmp_path.mkdir(parents=True, exist_ok=True)

    if evaluator == "hugging-face-evaluate":
        _evaluate_roundtrip(native, tmp_path)
    else:
        for entry in native:
            if evaluator == "deepeval":
                from deepeval.test_case import LLMTestCase

                entry["test_case"] = LLMTestCase(**entry["test_case"])
            elif evaluator == "ragas":
                from ragas import SingleTurnSample
                from ragas.metrics.result import MetricResult

                entry["sample"] = SingleTurnSample(**entry["sample"])
                if "metric_result" in entry:
                    entry["metric_result"] = MetricResult(**entry["metric_result"])
            elif evaluator == "lighteval":
                from lighteval.models.model_output import ModelResponse
                from lighteval.tasks.requests import Doc

                entry["doc"] = Doc(**entry["doc"])
                entry["model_response"] = ModelResponse(**entry["model_response"])
            elif evaluator == "autoevals":
                from autoevals.score import Score

                _metric_objects(entry, Score)
            elif evaluator == "openevals":
                from openevals.types import EvaluatorResult

                # The SDK's public result type is a TypedDict, not a DTO class.
                _metric_objects(entry, EvaluatorResult)
            elif evaluator == "arize-phoenix-evals":
                from phoenix.evals.evaluators import Score

                _metric_objects(entry, Score)
            elif evaluator == "opik":
                from opik.evaluation.metrics.score_result import ScoreResult

                _metric_objects(entry, ScoreResult)

    serialized = serialize_results(evaluator, native)
    # Exercise the exact JSON boundary consumed by the separate recipient.
    result = json.loads(json.dumps(serialized, allow_nan=False, ensure_ascii=False))
    restored = export_records(evaluator, result)
    if len(original) != len(restored):
        raise AssertionError("SDK serialization changed the number of cases")
    for before, after, supplied, retained in zip(
        original, restored, payload, result, strict=True
    ):
        before_facts = {key: val for key, val in before.items() if key != "context"}
        after_facts = {key: val for key, val in after.items() if key != "context"}
        if _canonical(before_facts) != _canonical(after_facts):
            raise AssertionError(
                f"SDK serialization changed case facts: {before['id']}"
            )
        _assert_retained(supplied, retained, path=before["id"])
    return result


def _metric_objects(entry: dict, constructor: Any) -> None:
    if "metric_result" not in entry:
        return
    value = entry["metric_result"]
    entry["metric_result"] = (
        [constructor(**metric) for metric in value]
        if isinstance(value, list)
        else constructor(**value)
    )


def _evaluate_roundtrip(entries: list[dict], tmp_path: Path) -> None:
    import datasets
    import evaluate

    class RetainedColumns(evaluate.Metric):
        def __init__(self, column_features, **kwargs):
            self._column_features = column_features
            super().__init__(**kwargs)

        def _info(self):
            return evaluate.MetricInfo(
                description="Serialize retained nullable columns without grading",
                citation="",
                inputs_description="Previously captured prediction/reference pairs",
                features=self._column_features,
            )

        def _compute(self, predictions, references):
            return {"predictions": predictions, "references": references}

    # Evaluate 0.4.6 rejects a null first element under its string feature,
    # including otherwise valid nullable Arrow columns. Partition actual cases
    # by column nullability, use a native null feature where appropriate, then
    # restore the original case order. No sentinel or generated text is added.
    groups: dict[tuple[bool, bool], list[dict]] = {}
    for entry in entries:
        key = (
            entry["predictions"][0] is None,
            entry.get("references", [None])[0] is None,
        )
        groups.setdefault(key, []).append(entry)
    for group_index, ((prediction_null, reference_null), group) in enumerate(
        groups.items()
    ):
        features = datasets.Features(
            {
                "predictions": datasets.Value("null" if prediction_null else "string"),
                "references": datasets.Value("null" if reference_null else "string"),
            }
        )
        metric = RetainedColumns(
            features, cache_dir=str(tmp_path / f"evaluate-columns-{group_index}")
        )
        columns = metric.compute(
            predictions=[entry["predictions"][0] for entry in group],
            references=[entry.get("references", [None])[0] for entry in group],
        )
        if len(columns["predictions"]) != len(group) or len(
            columns["references"]
        ) != len(group):
            raise AssertionError("Evaluate serialization changed the number of cases")
        for index, entry in enumerate(group):
            entry["predictions"] = [columns["predictions"][index]]
            if "references" in entry:
                entry["references"] = [columns["references"][index]]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False)


def _assert_retained(original: Any, result: Any, *, path: str) -> None:
    """Permit SDK defaults, but reject discarded or coerced source fields."""
    if isinstance(original, dict):
        if not isinstance(result, dict):
            raise AssertionError(f"SDK serialization changed object at {path}")
        for key, value in original.items():
            if key not in result:
                raise AssertionError(f"SDK serialization dropped {path}.{key}")
            _assert_retained(value, result[key], path=f"{path}.{key}")
    elif isinstance(original, list):
        if not isinstance(result, list) or len(original) != len(result):
            raise AssertionError(f"SDK serialization changed list at {path}")
        for index, (before, after) in enumerate(zip(original, result, strict=True)):
            _assert_retained(before, after, path=f"{path}[{index}]")
    elif _canonical(original) != _canonical(result):
        raise AssertionError(f"SDK serialization changed value at {path}")
