"""Bounded, explicit JSON export of Langfuse Python experiment results.

This is InvarLock's capture envelope, not a Langfuse dashboard export. It
requires no SDK and never synthesizes model likelihood or judge evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from importlib import import_module
from typing import Any

from invarlock.evaluation_record_contracts.contracts import (
    MAX_RECORDS,
    EvaluationRecordsError,
)
from invarlock.evaluation_records.capture_facts import merge_capture_scores

_HOSTED_TEXT_FIELDS = {
    "source_trace_id",
    "source_observation_id",
    "dataset_id",
    "dataset_name",
    "created_at",
    "updated_at",
}
_HOSTED_FIELDS = _HOSTED_TEXT_FIELDS | {"status", "media_references"}


def serialize_experiment_result(result: Any, version: str) -> dict[str, Any]:
    """Read public SDK fields at capture time; recipients need only the JSON."""
    if isinstance(result, dict):
        return result
    from importlib.metadata import version as installed_version

    dataset_item_type = import_module("langfuse.api").DatasetItem
    experiment_type = import_module("langfuse.experiment").ExperimentResult

    if not isinstance(result, experiment_type):
        raise EvaluationRecordsError("Langfuse requires an ExperimentResult or export")
    if installed_version("langfuse") != version:
        raise EvaluationRecordsError(
            "Langfuse source version differs from installed SDK"
        )

    def evaluation(value: Any) -> dict[str, Any]:
        return {
            key: getattr(value, key)
            for key in (
                "name",
                "value",
                "comment",
                "metadata",
                "data_type",
                "config_id",
            )
        }

    def item(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if not isinstance(value, dataset_item_type):
            raise EvaluationRecordsError("Langfuse item requires a dict or DatasetItem")
        captured = {
            key: getattr(value, key)
            for key in (
                "id",
                "input",
                "expected_output",
                "metadata",
                "source_trace_id",
                "source_observation_id",
                "dataset_id",
                "dataset_name",
            )
        }
        captured["status"] = value.status.value
        for key in ("created_at", "updated_at"):
            captured[key] = getattr(value, key).isoformat()
        captured["media_references"] = [
            media.model_dump(mode="json") for media in value.media_references
        ]
        return captured

    if not 1 <= len(result.item_results) <= MAX_RECORDS:
        raise EvaluationRecordsError("Langfuse requires bounded nonempty item results")
    return {
        "format": "invarlock/langfuse-export-v1",
        "sdk_version": version,
        "result": {
            **{
                key: getattr(result, key)
                for key in (
                    "name",
                    "run_name",
                    "description",
                    "experiment_id",
                    "dataset_run_id",
                    "dataset_run_url",
                )
            },
            "item_results": [
                {
                    "item": item(row.item),
                    "output": row.output,
                    "evaluations": [evaluation(value) for value in row.evaluations],
                    "trace_id": row.trace_id,
                    "dataset_run_id": row.dataset_run_id,
                }
                for row in result.item_results
            ],
            "run_evaluations": [evaluation(value) for value in result.run_evaluations],
        },
    }


def _object(
    value: Any, required: set[str], optional: set[str], label: str
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - required - optional
    ):
        raise EvaluationRecordsError(
            f"Langfuse {label} has missing or unsupported fields"
        )
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise EvaluationRecordsError(f"Langfuse {label} requires nonempty bounded text")
    return value


def _optional_text(value: Any, label: str) -> None:
    if value is not None:
        _text(value, label)


def _evaluations(value: Any) -> dict[str, float]:
    if not isinstance(value, list) or len(value) > MAX_RECORDS:
        raise EvaluationRecordsError("Langfuse evaluations require a bounded array")
    scores: dict[str, float] = {}
    names: set[str] = set()
    for entry in value:
        evaluation = _object(
            entry,
            {"name", "value"},
            {"data_type", "comment", "metadata", "config_id"},
            "evaluation",
        )
        name = _text(evaluation["name"], "score name")
        if name in names:
            raise EvaluationRecordsError(
                "Langfuse duplicate evaluation names are ambiguous"
            )
        names.add(name)
        kind, score = evaluation.get("data_type"), evaluation["value"]
        if evaluation.get("metadata") is not None and not isinstance(
            evaluation["metadata"], dict
        ):
            raise EvaluationRecordsError(
                "Langfuse evaluation metadata must be an object or null"
            )
        for field in ("comment", "config_id"):
            if evaluation.get(field) is not None and not isinstance(
                evaluation[field], str
            ):
                raise EvaluationRecordsError(
                    f"Langfuse evaluation {field} must be text or null"
                )
        if kind == "CATEGORICAL" and isinstance(score, str):
            # Preserve categories in context, never invent a numeric mapping.
            continue
        if kind == "BOOLEAN" and (
            type(score) is bool or type(score) in (int, float) and score in (0, 1)
        ):
            scores[name] = float(score)
        elif (
            kind in (None, "NUMERIC")
            and type(score) in (int, float)
            and math.isfinite(score)
        ):
            scores[name] = float(score)
        else:
            raise EvaluationRecordsError(
                "Langfuse evaluation value contradicts its declared data_type"
            )
    return scores


def parse_langfuse_export(
    value: Any, *, source: Mapping[str, str] | None, run_id: str | None
) -> list[dict[str, Any]]:
    """Map the explicit experiment-result envelope through shared case capture."""
    envelope = _object(value, {"format", "sdk_version", "result"}, set(), "export")
    if envelope["format"] != "invarlock/langfuse-export-v1":
        raise EvaluationRecordsError(
            "langfuse-json requires invarlock/langfuse-export-v1"
        )
    version = _text(envelope["sdk_version"], "SDK version")
    if source is None or dict(source) != {"name": "langfuse", "version": version}:
        raise EvaluationRecordsError(
            "Langfuse source name/version must match the export SDK version"
        )
    result = _object(
        envelope["result"],
        {"name", "run_name", "experiment_id", "item_results", "run_evaluations"},
        {"description", "dataset_run_id", "dataset_run_url"},
        "result",
    )
    for key in ("name", "run_name", "experiment_id"):
        _text(result[key], key)
    if run_id != result["run_name"]:
        raise EvaluationRecordsError("Langfuse run_id must match exported run_name")
    for key in ("dataset_run_id", "dataset_run_url"):
        _optional_text(result.get(key), key)
    if result.get("description") is not None and not isinstance(
        result["description"], str
    ):
        raise EvaluationRecordsError("Langfuse description must be text or null")
    dataset_run_id = result.get("dataset_run_id")
    if dataset_run_id is not None and dataset_run_id != result["experiment_id"]:
        raise EvaluationRecordsError("Langfuse dataset run and experiment IDs differ")
    _evaluations(result["run_evaluations"])
    rows = result["item_results"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_RECORDS:
        raise EvaluationRecordsError(
            "Langfuse item_results require a nonempty bounded array"
        )
    records = []
    for row in rows:
        native = _object(
            row,
            {"item", "output", "evaluations"},
            {"trace_id", "dataset_run_id"},
            "item result",
        )
        item = _object(
            native["item"],
            {"input"},
            {"expected_output", "metadata", "id"} | _HOSTED_FIELDS,
            "item",
        )
        if item.keys() & _HOSTED_FIELDS:
            _text(item.get("id"), "hosted native ID")
            if "status" in item and item["status"] not in ("ACTIVE", "ARCHIVED"):
                raise EvaluationRecordsError(
                    "Langfuse dataset item status is unsupported"
                )
            for field in _HOSTED_TEXT_FIELDS:
                _optional_text(item.get(field), field)
            if "media_references" in item:
                media = item["media_references"]
                if (
                    not isinstance(media, list)
                    or len(media) > MAX_RECORDS
                    or any(not isinstance(entry, dict) for entry in media)
                ):
                    raise EvaluationRecordsError(
                        "Langfuse media references require a bounded object array"
                    )
                # Media descriptions remain opaque provenance. Import never resolves them.
        metadata = item.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise EvaluationRecordsError(
                "Langfuse item metadata must be an object or null"
            )
        local_id = metadata.get("invarlock_id")
        item_id = item.get("id")
        if local_id is not None:
            _text(local_id, "invarlock_id")
        if item_id is not None:
            _text(item_id, "item ID")
        if local_id is not None and item_id is not None and local_id != item_id:
            raise EvaluationRecordsError("Langfuse item ID conflicts with invarlock_id")
        record_id = _text(
            item_id if item_id is not None else local_id, "stable item ID"
        )
        _optional_text(native.get("trace_id"), "trace ID")
        if native.get("dataset_run_id") != dataset_run_id:
            raise EvaluationRecordsError(
                "Langfuse item dataset run ID differs from experiment"
            )
        if dataset_run_id is not None and item_id is None:
            raise EvaluationRecordsError("Langfuse dataset item requires its native ID")
        error = metadata.get("invarlock_error")
        _optional_text(error, "invarlock_error")
        record = {
            "id": record_id,
            "input": item["input"],
            "expected": item.get("expected_output"),
            "output": native["output"],
            "error": error,
            "scores": merge_capture_scores(
                _evaluations(native["evaluations"]), metadata
            ),
            "metadata": {
                key: val
                for key, val in metadata.items()
                if isinstance(val, str) and not key.startswith("invarlock_")
            },
            "context": {
                "langfuse": {
                    "name": result["name"],
                    "run_name": result["run_name"],
                    "experiment_id": result["experiment_id"],
                    "dataset_run_id": dataset_run_id,
                    "item_result": native,
                }
            },
        }
        if "invarlock_likelihood" in metadata:
            record["likelihood"] = metadata["invarlock_likelihood"]
        records.append(record)
    return records
