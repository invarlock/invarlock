"""Dedicated evaluator mappings behind one complete, offline export boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

from invarlock.evaluation_record_contracts.contracts import (
    MAX_INPUT_BYTES,
    EvaluationRecordsError,
    validate,
)

EVALUATORS = (
    "lm-evaluation-harness",
    "inspect-ai",
    "promptfoo",
    "deepeval",
    "ragas",
    "lighteval",
    "hugging-face-evaluate",
    "pydantic-evals",
    "autoevals",
    "openevals",
    "mlflow",
    "garak",
    "openai-evals",
    "arize-phoenix-evals",
    "langfuse",
    "opik",
    "azure-ai-evaluation",
    "evidently",
    "trulens",
)
_SCALAR = {
    "deepeval",
    "ragas",
    "lighteval",
    "hugging-face-evaluate",
    "autoevals",
    "openevals",
    "arize-phoenix-evals",
    "opik",
}
_BATCH = {
    "pydantic-evals",
    "azure-ai-evaluation",
    "evidently",
    "mlflow",
    "garak",
    "openai-evals",
    "trulens",
}
FORMAT = "invarlock/evaluator-export-v1"


def _serialize(evaluator: str, result: Any, version: str) -> Any:
    if evaluator in _SCALAR:
        from invarlock.evaluation_records.scalar_integrations import serialize_results

        return serialize_results(evaluator, result)
    if evaluator in _BATCH:
        from invarlock.evaluation_records.batch_integrations import serialize_results

        return serialize_results(evaluator, result)
    if evaluator == "langfuse":
        from invarlock.evaluation_records.langfuse import serialize_experiment_result

        return serialize_experiment_result(result, version)
    if evaluator == "inspect-ai" and not isinstance(result, dict):
        from importlib.metadata import version as installed_version

        eval_log_type = import_module("inspect_ai.log").EvalLog

        if not isinstance(result, eval_log_type):
            raise EvaluationRecordsError("Inspect requires an EvalLog or JSON export")
        if installed_version("inspect-ai") != version:
            raise EvaluationRecordsError(
                "Inspect source version differs from installed SDK"
            )
        return result.model_dump(mode="json")
    return result


def _plain_json(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise EvaluationRecordsError("evaluator export exceeds JSON nesting limit")
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise EvaluationRecordsError("evaluator export requires string JSON keys")
        for item in value.values():
            _plain_json(item, depth + 1)
    elif type(value) is list:
        for item in value:
            _plain_json(item, depth + 1)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise EvaluationRecordsError(
            "use the evaluator's explicit native field serializer"
        )


def _records(
    evaluator: str, payload: Any, version: str, run_id: str
) -> list[dict[str, Any]]:
    from invarlock.evaluation_records.adapters import (
        _harness,
        _inspect,
        _promptfoo,
        _rows,
    )

    if evaluator == "inspect-ai":
        return _inspect(payload, allow_native_score_values=True)
    if evaluator == "lm-evaluation-harness":
        return _harness(_rows(payload, "Harness samples"), allow_stable_ids=True)
    if evaluator == "promptfoo":
        return _promptfoo(_rows(payload, "Promptfoo results"))
    if evaluator == "langfuse":
        from invarlock.evaluation_records.langfuse import parse_langfuse_export

        return parse_langfuse_export(
            payload, source={"name": evaluator, "version": version}, run_id=run_id
        )
    if evaluator in _SCALAR:
        from invarlock.evaluation_records.scalar_integrations import export_records

        return export_records(evaluator, payload)
    if evaluator in _BATCH:
        from invarlock.evaluation_records.batch_integrations import export_records

        return export_records(evaluator, payload)
    raise EvaluationRecordsError(
        "unsupported evaluator; choose " + ", ".join(EVALUATORS)
    )


def parse_evaluator_export(
    value: Any, *, source: Mapping[str, str] | None, run_id: str | None
) -> list[dict[str, Any]]:
    """Reconstruct source-specific mappings; the source name grants no authority."""
    validate(value, "evaluator_export")
    version = value["source_version"]
    if not version.strip():
        raise EvaluationRecordsError("evaluator source version requires bounded text")
    if (
        source is None
        or dict(source) != {"name": value["evaluator"], "version": version}
        or run_id != value["run_id"]
    ):
        raise EvaluationRecordsError(
            "evaluator export source or run identity differs from the request"
        )
    try:
        return _records(value["evaluator"], value["payload"], version, value["run_id"])
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
        raise EvaluationRecordsError(
            f"invalid {value['evaluator']} capture: {exc}"
        ) from exc


def export_evaluator_result(
    evaluator: str,
    result: Any,
    destination: str | Path,
    *,
    expected_ids: list[str],
    source_version: str,
    run_id: str,
    artifact_digest: str | None = None,
    service_identity: Mapping[str, Any] | None = None,
    input_projection: Mapping[str, Any] | None = None,
    score_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture one supported native result and return its validated canonical run.

    The SDK-side result must follow the named integration's documented profile.
    Retained metrics are context; the selected InvarLock scorer owns evaluation.
    No evaluator SDK is required to import, verify or report the exported file.
    """
    from invarlock.evaluation_records.adapters import write_evaluator_export

    if not isinstance(evaluator, str) or evaluator not in EVALUATORS:
        raise EvaluationRecordsError("unsupported evaluator")
    try:
        payload = _serialize(evaluator, result, source_version)
    except (ImportError, TypeError, ValueError, AttributeError, RecursionError) as exc:
        if isinstance(exc, EvaluationRecordsError):
            raise
        raise EvaluationRecordsError(
            f"cannot capture {evaluator} result: {exc}"
        ) from exc
    envelope = {
        "format": FORMAT,
        "evaluator": evaluator,
        "source_version": source_version,
        "run_id": run_id,
        "payload": payload,
    }
    _plain_json(envelope)
    try:
        # No default=str: SDK objects require the dedicated field serializers.
        raw = json.dumps(envelope, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise EvaluationRecordsError(
            "evaluator result must contain finite JSON-compatible native fields"
        ) from exc
    if len(raw) > MAX_INPUT_BYTES:
        raise EvaluationRecordsError("evaluator export exceeds its byte limit")
    return write_evaluator_export(
        raw,
        destination,
        expected_ids=expected_ids,
        adapter="evaluator-json",
        source={"name": evaluator, "version": source_version},
        run_id=run_id,
        artifact_digest=artifact_digest,
        service_identity=service_identity,
        input_projection=input_projection,
        score_provenance=score_provenance,
    )
