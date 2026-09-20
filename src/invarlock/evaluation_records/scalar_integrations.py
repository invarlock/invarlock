"""Per-case SDK exports for scalar evaluator ecosystems, without SDK imports.

``export_records(name, entries)`` accepts a nonempty list of dictionaries. Each
entry has a caller-owned stable ``id`` and the following source-specific fields:

* deepeval: ``test_case`` (LLMTestCase), ``metric_result`` (measured metric or
  MetricsData: score, optional name/reason/error/threshold).
* ragas: ``sample`` (SingleTurnSample), ``metric_result`` (MetricResult).
* lighteval: ``doc`` (Doc), ``model_response`` (ModelResponse), ``metric_result``
  (the per-sample scalar or metric dictionary, never the aggregate result).
* hugging-face-evaluate: ``input``, ``predictions`` and ``references`` (each a
  singleton batch supplied to compute), and ``metric_result`` (compute result).
* autoevals: ``input``, ``output``, ``expected``, ``metric_result`` (Score).
* openevals: ``inputs``, ``outputs``, ``reference_outputs``, ``metric_result``
  (EvaluatorResult or a nonempty list of these).
* arize-phoenix-evals: ``record`` (input/output/expected evaluation arguments),
  ``metric_result`` (Score or a nonempty list of Scores).
* opik: ``dataset_item`` (input/output/reference data passed to the metric),
  ``metric_result`` (ScoreResult or a nonempty list of ScoreResults).

``metric_result`` is optional in every wrapper: capture original case facts
without running an upstream metric when InvarLock will score them. Supplied
metrics retain their native shape and are validated. Reference fields may be
omitted for reference-free judge tasks; LightEval's generative profile uses
``choices=[]`` and the unused integer ``gold_index=0``. Missing references make
exact-match/NLL facts unavailable.
JSON outputs remain structured; text-only scorer capability is checked centrally.

SDK data objects or dictionaries using the same Python field names are accepted.
Only the explicitly enumerated data attributes of SDK objects are retained;
JSON dictionaries retain all fields. Unsupported nested objects need an explicit
JSON mapping by the caller; no repr, str, model_dump or arbitrary serializer runs.
All entries may include ``metadata`` and ``error``. Only explicit
``metadata.invarlock_likelihood`` becomes a likelihood observation. Upstream
metrics stay in context, with no generation or qualification authority. Optional
``metadata.invarlock_scores`` explicitly selects finite numeric observations;
score provenance and authority remain the responsibility of the central capture.
The entire normalized wrapper is retained as ``context.upstream_record`` and
can be passed back to this function for an identical offline import.

Shapes correspond to the pinned qualification matrix: DeepEval 4.1.3,
Ragas 0.4.3, LightEval 0.13.0, Evaluate 0.4.6, AutoEvals 0.3.0,
OpenEvals 0.2.0, Phoenix Evals 3.3.0 and Opik 2.2.7.
"""

from __future__ import annotations

import math
from typing import Any, NoReturn, cast

from invarlock.evaluation_record_contracts.contracts import (
    MAX_RECORDS,
    EvaluationRecordsError,
)
from invarlock.evaluation_records.capture_facts import merge_capture_scores

EVALUATORS = (
    "deepeval",
    "ragas",
    "lighteval",
    "hugging-face-evaluate",
    "autoevals",
    "openevals",
    "arize-phoenix-evals",
    "opik",
)

_CASE_FIELDS = {
    "test_case": (
        "input",
        "actual_output",
        "expected_output",
        "context",
        "retrieval_context",
        "metadata",
        "name",
        "tags",
        "comments",
        "token_cost",
        "completion_time",
        "multimodal",
        "tools_called",
        "expected_tools",
        "mcp_servers",
        "mcp_tools_called",
        "mcp_resources_called",
        "mcp_prompts_called",
        "custom_column_key_values",
    ),
    "sample": (
        "user_input",
        "response",
        "reference",
        "retrieved_contexts",
        "reference_contexts",
        "retrieved_context_ids",
        "reference_context_ids",
        "multi_responses",
        "rubrics",
        "persona_name",
        "query_style",
        "query_length",
    ),
    "doc": (
        "query",
        "choices",
        "gold_index",
        "instruction",
        "specific",
        "id",
        "task_name",
        "original_query",
        "unconditioned_query",
        "generation_size",
        "stop_sequences",
        "num_samples",
        "use_logits",
        "fewshot_sorting_class",
        "fewshot_samples",
        "sampling_methods",
        "images",
        "generation_grammar",
    ),
    "model_response": (
        "input",
        "input_tokens",
        "text",
        "output_tokens",
        "text_post_processed",
        "reasonings",
        "logprobs",
        "argmax_logits_eq_gold",
        "logits",
        "unconditioned_logprobs",
        "truncated_tokens_count",
        "padded_tokens_count",
    ),
}
_METRIC_FIELDS = {
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
    "openevals": ("key", "score", "comment", "metadata", "source_run_id"),
    "arize-phoenix-evals": (
        "name",
        "score",
        "label",
        "explanation",
        "metadata",
        "kind",
        "direction",
    ),
    "opik": ("name", "value", "reason", "category_name", "metadata", "scoring_failed"),
}


def _fail(message: str) -> NoReturn:
    raise EvaluationRecordsError(message)


def _json(value: Any, path: str, depth: int = 0) -> Any:
    if depth > 32:
        _fail(f"{path} exceeds JSON nesting limit")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            _fail(f"{path} requires string JSON keys")
        return {
            key: _json(item, f"{path}.{key}", depth + 1) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_json(item, f"{path}[]", depth + 1) for item in value]
    _fail(f"{path} requires finite JSON data; explicitly map unsupported SDK values")


def _object(value: Any, fields: tuple[str, ...], path: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], _json(value, path))
    selected = {
        field: getattr(value, field) for field in fields if hasattr(value, field)
    }
    if not selected:
        _fail(f"{path} requires a native data object or field dictionary")
    return cast(dict[str, Any], _json(selected, path))


def _required(value: dict[str, Any], key: str) -> Any:
    if key not in value:
        _fail(f"native case is missing required {key!r}")
    return value[key]


def _one(value: Any, name: str) -> Any:
    if not isinstance(value, list) or len(value) != 1:
        _fail(f"{name} requires exactly one per-case value; select a single response")
    return value[0]


def _metric(evaluator: str, value: Any) -> Any:
    if evaluator in ("lighteval", "hugging-face-evaluate"):
        native = _json(value, "metric_result")
        if evaluator == "lighteval" and type(native) in (float, int):
            return native
        if not isinstance(native, dict) or not native:
            _fail("metric_result requires a nonempty per-case metric dictionary")
        return native
    if isinstance(value, list):
        if evaluator not in ("openevals", "arize-phoenix-evals", "opik") or not value:
            _fail("metric_result has an unsupported or empty result list")
        if any(isinstance(item, list) for item in value):
            _fail("metric_result requires a flat list of native metric results")
        return [_metric(evaluator, item) for item in value]
    native = _object(value, _METRIC_FIELDS[evaluator], "metric_result")
    score_field = "value" if evaluator in ("ragas", "opik") else "score"
    _required(native, score_field)
    if evaluator in ("autoevals", "opik", "openevals"):
        name = _required(native, "key" if evaluator == "openevals" else "name")
        if not isinstance(name, str) or not name:
            _fail("metric_result requires a nonempty metric name")
    return native


def _extract(evaluator: str, entry: dict[str, Any]) -> tuple[Any, Any, Any]:
    if evaluator == "deepeval":
        case = entry["test_case"]
        return (
            _required(case, "input"),
            case.get("expected_output"),
            _required(case, "actual_output"),
        )
    if evaluator == "ragas":
        case = entry["sample"]
        if case.get("multi_responses"):
            _fail("Ragas multi_responses is ambiguous; select one response explicitly")
        return (
            _required(case, "user_input"),
            case.get("reference"),
            _required(case, "response"),
        )
    if evaluator == "lighteval":
        doc, response = entry["doc"], entry["model_response"]
        choices, indices = _required(doc, "choices"), _required(doc, "gold_index")
        # LightEval 0.13.0 documents this exact Doc shape for generative tasks.
        # Zero is an unused placeholder here, not an index into an answer list.
        if choices == [] and type(indices) is int and indices == 0:
            return (
                _required(doc, "query"),
                None,
                _one(_required(response, "text"), "LightEval ModelResponse.text"),
            )
        indices = indices if isinstance(indices, list) else [indices]
        if (
            not isinstance(choices, list)
            or not choices
            or not indices
            or any(not isinstance(choice, str) for choice in choices)
            or any(type(i) is not int or not 0 <= i < len(choices) for i in indices)
            or len(indices) != len(set(indices))
        ):
            _fail("LightEval gold_index must select valid, unique text choices")
        golds = [choices[i] for i in indices]
        return (
            _required(doc, "query"),
            golds[0] if len(golds) == 1 else golds,
            _one(_required(response, "text"), "LightEval ModelResponse.text"),
        )
    if evaluator == "hugging-face-evaluate":
        return (
            _required(entry, "input"),
            _one(entry["references"], "Evaluate references")
            if "references" in entry
            else None,
            _one(_required(entry, "predictions"), "Evaluate predictions"),
        )
    if evaluator == "openevals":
        return (
            _required(entry, "inputs"),
            entry.get("reference_outputs"),
            _required(entry, "outputs"),
        )
    if evaluator == "arize-phoenix-evals":
        case = entry["record"]
        return (
            _required(case, "input"),
            case.get("expected"),
            _required(case, "output"),
        )
    if evaluator == "opik":
        case = entry["dataset_item"]
        return (
            _required(case, "input"),
            case.get("reference"),
            _required(case, "output"),
        )
    return (
        _required(entry, "input"),
        entry.get("expected"),
        _required(entry, "output"),
    )


def export_records(evaluator: str, results: object) -> list[dict[str, Any]]:
    """Extract retained native cases; callers independently bind their schedule."""
    if evaluator not in EVALUATORS:
        _fail(f"unsupported scalar evaluator: {evaluator!r}")
    if (
        not isinstance(results, list)
        or not 1 <= len(results) <= MAX_RECORDS
        or any(not isinstance(entry, dict) for entry in results)
    ):
        _fail(
            "scalar evaluator export requires a nonempty bounded list of native entries"
        )
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    object_keys = {
        "deepeval": ("test_case",),
        "ragas": ("sample",),
        "lighteval": ("doc", "model_response"),
        "arize-phoenix-evals": ("record",),
        "opik": ("dataset_item",),
    }.get(evaluator, ())
    for original in results:
        entry = dict(original)
        for key in object_keys:
            value = _required(entry, key)
            if key in _CASE_FIELDS:
                entry[key] = _object(value, _CASE_FIELDS[key], key)
            elif not isinstance(value, dict):
                _fail(f"{key} must contain the native evaluation argument dictionary")
        if "metric_result" in entry:
            entry["metric_result"] = _metric(evaluator, entry["metric_result"])
        entry = _json(entry, "entry")
        identifier = _required(entry, "id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            _fail("native cases require unique, nonempty stable string ids")
        ids.add(identifier)
        case_input, expected, output = _extract(evaluator, entry)
        if case_input is None:
            _fail("native case requires a per-case input")
        error = entry.get("error")
        if error is not None and (not isinstance(error, str) or not error):
            _fail("native error must be a nonempty string or null")
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, dict):
            _fail("native entry metadata must be an object")
        if output is None and error is None and "invarlock_likelihood" not in metadata:
            _fail(
                "native case requires an actual output or null output with an explicit "
                "error or likelihood observation"
            )
        scores = merge_capture_scores({}, metadata)
        record = {
            "id": identifier,
            "input": case_input,
            "expected": expected,
            "output": output,
            "scores": scores,
            "metadata": {
                key: value
                for key, value in metadata.items()
                if isinstance(value, str) and not key.startswith("invarlock_")
            },
            "error": error,
            "context": {"upstream_record": entry},
        }
        if "invarlock_likelihood" in metadata:
            record["likelihood"] = metadata["invarlock_likelihood"]
        records.append(record)
    return records


def serialize_results(evaluator: str, results: object) -> list[dict[str, Any]]:
    """Detach SDK entries to the same native wrappers used by offline imports."""
    return [
        row["context"]["upstream_record"] for row in export_records(evaluator, results)
    ]
