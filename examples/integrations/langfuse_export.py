"""Retain a complete Langfuse experiment for offline InvarLock comparison.

Call ``write_experiment_export(result, path, expected_ids=dataset_ids)`` after
``Langfuse.run_experiment``. The independent ID list detects omitted task results.
Langfuse is needed only when producing the export, not when importing it.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

from invarlock.evaluation_record_contracts.contracts import (
    MAX_INPUT_BYTES,
    MAX_RECORDS,
)

SDK_VERSION = "4.14.1"
_RESULT_FIELDS = (
    "name",
    "run_name",
    "description",
    "experiment_id",
    "dataset_run_id",
    "dataset_run_url",
)
_EVALUATION_FIELDS = ("name", "value", "comment", "metadata", "data_type", "config_id")


def _check_json(value: Any) -> None:
    if value is None or type(value) in (str, bool, int, float):
        return
    if isinstance(value, list):
        for item in value:
            _check_json(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _check_json(item)
        return
    raise ValueError("experiment fields must contain JSON values and string keys")


def _evaluation(value: Any) -> dict[str, Any]:
    return {key: getattr(value, key) for key in _EVALUATION_FIELDS}


def _item(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    from langfuse.api import DatasetItem

    if not isinstance(value, DatasetItem):
        raise ValueError("experiment items must be local dictionaries or DatasetItem")
    # Langfuse's serializer emits API aliases even with by_alias=False. Keep
    # the SDK's public Python field names explicit in this capture envelope.
    result = {
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
    result["status"] = value.status.value
    for key in ("created_at", "updated_at"):
        result[key] = getattr(value, key).isoformat()
    result["media_references"] = [
        item.model_dump(mode="json") for item in value.media_references
    ]
    return result


def _identity(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    local_id = metadata.get("invarlock_id") if isinstance(metadata, dict) else None
    hosted_id = item.get("id")
    if local_id is not None and hosted_id is not None and local_id != hosted_id:
        raise ValueError("local and hosted item identities contradict each other")
    identity = hosted_id if hosted_id is not None else local_id
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("every item requires a stable invarlock_id or hosted id")
    return identity


def serialize_experiment(result: Any, *, expected_ids: list[str]) -> dict[str, Any]:
    """Export public SDK fields without accepting dropped or duplicate cases."""
    version = importlib.metadata.version("langfuse")
    if version != SDK_VERSION:
        raise ValueError(f"Langfuse {SDK_VERSION} is required; found {version}")
    from langfuse.experiment import ExperimentResult

    if not isinstance(result, ExperimentResult):
        raise ValueError("result must be an actual Langfuse ExperimentResult")
    if (
        not isinstance(expected_ids, list)
        or not expected_ids
        or len(expected_ids) > MAX_RECORDS
        or any(
            not isinstance(value, str) or not value.strip() for value in expected_ids
        )
        or len(set(expected_ids)) != len(expected_ids)
    ):
        raise ValueError("expected_ids must be a bounded nonempty list of unique IDs")
    if len(result.item_results) != len(expected_ids):
        raise ValueError("experiment result does not contain every expected case")
    rows, seen = [], set()
    for row in result.item_results:
        item = _item(row.item)
        identity = _identity(item)
        if identity in seen:
            raise ValueError("experiment contains duplicate item identities")
        seen.add(identity)
        rows.append(
            {
                "item": item,
                "output": row.output,
                "evaluations": [_evaluation(value) for value in row.evaluations],
                "trace_id": row.trace_id,
                "dataset_run_id": row.dataset_run_id,
            }
        )
    if seen != set(expected_ids):
        raise ValueError("experiment item identities differ from expected_ids")
    exported = {
        "format": "invarlock/langfuse-export-v1",
        "sdk_version": version,
        "result": {
            **{key: getattr(result, key) for key in _RESULT_FIELDS},
            "item_results": rows,
            "run_evaluations": [_evaluation(value) for value in result.run_evaluations],
        },
    }
    _check_json(exported)
    payload = json.dumps(exported, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(payload) + 1 > MAX_INPUT_BYTES:
        raise ValueError(f"experiment export exceeds the {MAX_INPUT_BYTES}-byte limit")
    return json.loads(payload)


def write_experiment_export(
    result: Any, destination: str | Path, *, expected_ids: list[str]
) -> dict[str, Any]:
    """Write a new export without replacing an existing capture."""
    exported = serialize_experiment(result, expected_ids=expected_ids)
    payload = json.dumps(exported, ensure_ascii=False, allow_nan=False).encode("utf-8")
    with Path(destination).open("xb") as stream:
        stream.write(payload)
        stream.write(b"\n")
    return exported
