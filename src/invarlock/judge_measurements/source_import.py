"""Import complete retained judge calls from an evaluator-owned collector."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, cast

from invarlock.evidence_pack_json import StrictJsonError, parse_json_bytes
from invarlock.judge_measurement_types import JudgeMeasurementPlan, JudgeMeasurements
from invarlock.judge_measurements.contracts import (
    MEASUREMENTS_FORMAT,
    MEASUREMENTS_MAX_BYTES,
    SOURCE_FORMAT,
    JudgeMeasurementContractError,
    measurement_plan_digest,
    validate_measurement_plan,
    validate_measurements,
)


def import_judge_sources(
    sources: Mapping[str, bytes],
    *,
    plan: JudgeMeasurementPlan,
    baseline_run: dict[str, Any],
    subject_run: dict[str, Any],
) -> JudgeMeasurements:
    """Assemble unchanged canonical retained-call shards and replay every trial.

    Each shard has format ``invarlock/retained-judge-json-v1`` and a trials
    array. Attempts retain requests, responses, errors and source positions.
    Scalar scores and aggregate evaluator summaries cannot form this evidence.
    No provider SDK, network operation or inference runs during import.
    """
    if not isinstance(plan, dict):
        raise JudgeMeasurementContractError(
            "The judge measurement plan must be an object"
        )
    validate_measurement_plan(plan)
    if not isinstance(sources, Mapping) or not 1 <= len(sources) <= 1000:
        raise JudgeMeasurementContractError(
            "Supply 1..1000 retained judge source shards"
        )
    retained = []
    trials: list[dict[str, Any]] = []
    total_bytes = 0
    for source_id, raw in sources.items():
        if not isinstance(raw, bytes) or not 1 <= len(raw) <= 16 * 1024 * 1024:
            raise JudgeMeasurementContractError(
                "Each retained judge source must contain 1 byte..16 MiB"
            )
        total_bytes += len(raw)
        if total_bytes > MEASUREMENTS_MAX_BYTES:
            raise JudgeMeasurementContractError(
                "Retained judge sources exceed the measurement byte limit"
            )
        try:
            value = parse_json_bytes(raw, label="retained judge source")
        except StrictJsonError as exc:
            raise JudgeMeasurementContractError(str(exc)) from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"format", "trials"}
            or value["format"] != SOURCE_FORMAT
            or not isinstance(value["trials"], list)
        ):
            raise JudgeMeasurementContractError(
                "A retained source must contain complete judge trials, not scores or summaries"
            )
        if len(trials) + len(value["trials"]) > 200000 or any(
            not isinstance(trial, dict) for trial in value["trials"]
        ):
            raise JudgeMeasurementContractError(
                "Retained judge trial inventory is invalid"
            )
        trials.extend(value["trials"])
        retained.append(
            {
                "source_id": source_id,
                "profile": "retained-judge-json-v1",
                "encoding": "utf-8",
                "media_type": "application/json",
                "byte_size": len(raw),
                "content": raw.decode("utf-8"),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    expected = plan["schedule"]["expected_trials"]
    completed = sum(trial.get("status") == "complete" for trial in trials)
    result = cast(
        JudgeMeasurements,
        {
            "format": MEASUREMENTS_FORMAT,
            "profile_id": plan["profile_id"],
            "plan_sha256": measurement_plan_digest(plan),
            "source_profile": "retained-judge-json-v1",
            "sources": retained,
            "trials": trials,
            "completeness": {
                "expected_trials": expected,
                "recorded_trials": len(trials),
                "completed_trials": completed,
                "status": "complete" if expected == completed else "incomplete",
            },
        },
    )
    validate_measurements(
        result, plan, baseline_run=baseline_run, subject_run=subject_run
    )
    return result
