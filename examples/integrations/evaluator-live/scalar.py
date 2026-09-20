"""Run a fresh supplied task and local SDK metrics, then capture native JSON.

``run(evaluator, cases, task, workdir)`` calls the synchronous ``task(case)``
exactly once for each independently identified case. Its returned dictionary
must contain ``output`` (text or null), and may contain ``error`` and ``metadata``.
Case metadata and task metadata are merged without overwriting conflicting facts.

The SDK's public exact-match API evaluates available text answers/references.
Likelihood-only, failed, and reference-free cases keep their original facts and
an explicit auxiliary-metric skip in context. Upstream metric failures also stay
in context; they do not become model failures or InvarLock decisions. The caller
owns model/API authorization, source identities, and final InvarLock scoring.
This module imports optional SDKs only when ``run`` is called.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

VERSIONS = {
    "deepeval": "4.1.3",
    "ragas": "0.4.3",
    "hugging-face-evaluate": "0.4.6",
    "autoevals": "0.3.0",
    "openevals": "0.2.0",
    "arize-phoenix-evals": "3.3.0",
    "opik": "2.2.7",
}
METRIC_APIS = {
    "deepeval": "deepeval.metrics.ExactMatchMetric.measure",
    "ragas": "ragas.metrics.collections.ExactMatch.ascore",
    "hugging-face-evaluate": "evaluate.Metric.compute (local exact-match implementation)",
    "autoevals": "autoevals.ExactMatch.__call__",
    "openevals": "openevals.exact.exact_match",
    "arize-phoenix-evals": "phoenix.evals.metrics.exact_match",
    "opik": "opik.evaluation.metrics.Equals.score",
}


def run(
    evaluator: str,
    cases: list[dict],
    task: Callable[[dict], dict],
    workdir: str | Path,
) -> list[dict]:
    """Execute fresh task calls and SDK metrics; return evaluator-native-json rows."""
    if evaluator not in VERSIONS:
        raise ValueError(f"unsupported live scalar evaluator: {evaluator}")
    _validate_cases(cases)
    package = "evaluate" if evaluator == "hugging-face-evaluate" else evaluator
    installed = importlib.metadata.version(package)
    if installed != VERSIONS[evaluator]:
        raise ValueError(
            f"requires {package}=={VERSIONS[evaluator]}, found {installed}"
        )
    metric = _metric(evaluator, Path(workdir))
    results = []
    for case in cases:
        try:
            observed = task(deepcopy(case))
        except Exception as exc:
            observed = {"output": None, "error": f"{type(exc).__name__}: {exc}"}
        _validate_observation(observed)
        metadata = deepcopy(case.get("metadata", {}))
        for key, value in observed.get("metadata", {}).items():
            if key in metadata and _json(metadata[key]) != _json(value):
                raise ValueError(f"task metadata conflicts with case metadata: {key}")
            metadata[key] = deepcopy(value)
        if (
            observed["output"] is None
            and observed.get("error") is None
            and not isinstance(metadata.get("invarlock_likelihood"), dict)
        ):
            raise ValueError(
                "null output requires an explicit task error or likelihood facts"
            )
        entry = _entry(evaluator, case, observed, metadata)
        execution = entry["capture_execution"] = {
            "task_invocations": 1,
            "metric_api": METRIC_APIS[evaluator],
        }
        if observed.get("error") is not None:
            skip = "task reported an error"
        elif observed["output"] is None:
            skip = "no generated text; likelihood facts remain available separately"
        elif case.get("expected") is None:
            skip = "no text reference"
        else:
            skip = None
        if skip is not None:
            execution.update(metric_status="skipped", reason=skip)
        else:
            try:
                entry["metric_result"] = metric(entry)
            except Exception as exc:
                execution.update(
                    metric_status="failed", error=f"{type(exc).__name__}: {exc}"
                )
            else:
                execution["metric_status"] = "completed"
        # Serialize each result immediately, before a reused SDK metric mutates
        # its own score attributes on the next case.
        results.append(_serialize(evaluator, entry))
    return results


def _serialize(evaluator, entry):
    """Map public fields explicitly; the evaluator environment needs no core SDK."""
    result = dict(entry)
    if evaluator == "deepeval":
        result["test_case"] = {
            key: getattr(entry["test_case"], key)
            for key in ("input", "actual_output", "expected_output")
        }
    elif evaluator == "ragas":
        result["sample"] = {
            key: getattr(entry["sample"], key)
            for key in ("user_input", "response", "reference")
        }
    metric = entry.get("metric_result")
    if metric is not None and not isinstance(metric, dict):
        fields = {
            "deepeval": (
                "score",
                "name",
                "reason",
                "error",
                "threshold",
                "success",
                "strict_mode",
                "evaluation_model",
                "evaluation_cost",
                "verbose_logs",
            ),
            "ragas": ("value", "reason", "traces"),
            "autoevals": ("name", "score", "metadata", "error"),
            "arize-phoenix-evals": (
                "name",
                "score",
                "label",
                "explanation",
                "metadata",
                "kind",
                "direction",
            ),
            "opik": (
                "name",
                "value",
                "reason",
                "category_name",
                "metadata",
                "scoring_failed",
            ),
        }[evaluator]
        result["metric_result"] = {
            key: getattr(metric, key) for key in fields if hasattr(metric, key)
        }
    return json.loads(_json(result))


def _entry(evaluator, case, observed, metadata):
    entry = {
        "id": case["id"],
        "metadata": metadata,
        "error": observed.get("error"),
        "source_case": deepcopy(case),
        "task_result": deepcopy(observed),
    }
    prompt, output, expected = case["input"], observed["output"], case.get("expected")
    if evaluator == "deepeval":
        from deepeval.test_case import LLMTestCase

        entry["test_case"] = LLMTestCase(
            input=prompt, actual_output=output, expected_output=expected
        )
    elif evaluator == "ragas":
        from ragas import SingleTurnSample

        entry["sample"] = SingleTurnSample(
            user_input=prompt, response=output, reference=expected
        )
    elif evaluator == "hugging-face-evaluate":
        entry.update(input=prompt, predictions=[output], references=[expected])
    elif evaluator == "autoevals":
        entry.update(input=prompt, output=output, expected=expected)
    elif evaluator == "openevals":
        entry.update(inputs=prompt, outputs=output, reference_outputs=expected)
    elif evaluator == "arize-phoenix-evals":
        entry["record"] = {"input": prompt, "output": output, "expected": expected}
    elif evaluator == "opik":
        entry["dataset_item"] = {
            "input": prompt,
            "output": output,
            "reference": expected,
        }
    return entry


def _metric(evaluator, workdir):
    if evaluator == "deepeval":
        from deepeval.metrics import ExactMatchMetric

        def measure(entry):
            metric = ExactMatchMetric()
            metric.measure(entry["test_case"])
            return metric

        return measure
    if evaluator == "ragas":
        from ragas.metrics.collections import ExactMatch

        metric = ExactMatch()
        return lambda entry: asyncio.run(
            metric.ascore(
                response=entry["sample"].response, reference=entry["sample"].reference
            )
        )
    if evaluator == "autoevals":
        from autoevals import ExactMatch

        metric = ExactMatch()
        return lambda entry: metric(output=entry["output"], expected=entry["expected"])
    if evaluator == "openevals":
        from openevals.exact import exact_match

        return lambda entry: exact_match(
            outputs=entry["outputs"], reference_outputs=entry["reference_outputs"]
        )
    if evaluator == "arize-phoenix-evals":
        from phoenix.evals.metrics import exact_match

        return lambda entry: exact_match(
            entry["record"]["output"], entry["record"]["expected"]
        )
    if evaluator == "opik":
        from opik.evaluation.metrics import Equals

        metric = Equals(case_sensitive=True, track=False)
        return lambda entry: metric.score(
            output=entry["dataset_item"]["output"],
            reference=entry["dataset_item"]["reference"],
        )
    import datasets
    import evaluate

    class LocalExactMatch(evaluate.Metric):
        def _info(self):
            return evaluate.MetricInfo(
                description="Exact comparison of fresh task outputs and references",
                citation="",
                inputs_description="Singleton text prediction and reference",
                features=datasets.Features(
                    {
                        "predictions": datasets.Value("string"),
                        "references": datasets.Value("string"),
                    }
                ),
            )

        def _compute(self, predictions, references):
            return {
                "exact_match": sum(
                    output == expected
                    for output, expected in zip(predictions, references, strict=True)
                )
                / len(predictions)
            }

    workdir.mkdir(parents=True, exist_ok=True)
    metric = LocalExactMatch(cache_dir=str(workdir / "evaluate-metric"))
    return lambda entry: metric.compute(
        predictions=entry["predictions"], references=entry["references"]
    )


def _validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be a dictionary")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("case IDs must be unique nonempty strings")
        seen.add(identifier)
        if not isinstance(case.get("input"), str):
            raise ValueError("live scalar cases require text input")
        if case.get("expected") is not None and not isinstance(case["expected"], str):
            raise ValueError("live scalar references must be text or null")
        if not isinstance(case.get("metadata", {}), dict):
            raise ValueError("case metadata must be a dictionary")
        _json(case)


def _validate_observation(observed):
    if not isinstance(observed, dict) or "output" not in observed:
        raise ValueError("task must return a dictionary containing output")
    if observed["output"] is not None and not isinstance(observed["output"], str):
        raise ValueError("task output must be text or null")
    if observed.get("error") is not None and not isinstance(observed["error"], str):
        raise ValueError("task error must be text or null")
    if not isinstance(observed.get("metadata", {}), dict):
        raise ValueError("task metadata must be a dictionary")
    _json(observed)


def _json(value: Any) -> str:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for item in value.values():
            _json(item)
    elif isinstance(value, list):
        for item in value:
            _json(item)
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False)
