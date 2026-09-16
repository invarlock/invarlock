"""Opaque hosted service descriptors authenticate declarations, never weights."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    _canonical_chunks,
    _validator,
    digest,
)


def validate_service_identity(identity: Mapping[str, Any]) -> None:
    """Validate the closed descriptor, declared configuration and UTC window."""
    if not isinstance(identity, Mapping):
        raise EvaluationRecordsError("service_identity must be an object")
    identity = dict(identity)
    run_schema = _validator("run").schema
    assert isinstance(run_schema, Mapping)
    schema = run_schema["properties"]["service_identity"]
    from jsonschema import Draft202012Validator

    error = next(Draft202012Validator(schema).iter_errors(identity), None)
    if error is not None:
        raise EvaluationRecordsError(f"invalid service_identity: {error.message}")
    size = 0
    for chunk in _canonical_chunks(identity):
        size += len(chunk)
        if size > 1024 * 1024:
            raise EvaluationRecordsError(
                "service_identity exceeds the 1 MiB byte limit"
            )
    if identity["configuration_digest"] != digest(identity["configuration"]):
        raise EvaluationRecordsError("service_identity configuration digest differs")
    window = identity["observation_window"]
    try:
        start = datetime.fromisoformat(window["started_at"])
        end = datetime.fromisoformat(window["ended_at"])
    except ValueError as exc:
        raise EvaluationRecordsError(
            "service_identity requires valid UTC timestamps"
        ) from exc
    if start > end:
        raise EvaluationRecordsError("service_identity observation window is reversed")


def evaluated_subject_digest(run: Mapping[str, Any]) -> str:
    """Return artifact identity or hosted descriptor identity with no fallback."""
    if "artifact_digest" not in run:
        raise EvaluationRecordsError(
            "run requires explicit artifact_digest, null for hosted"
        )
    if "service_identity" in run:
        validate_service_identity(run["service_identity"])
        if run.get("artifact_digest") is not None:
            raise EvaluationRecordsError(
                "service identity cannot claim an artifact digest"
            )
        return digest(run["service_identity"])
    artifact = run.get("artifact_digest")
    if (
        not isinstance(artifact, str)
        or re.fullmatch(r"sha256:[a-f0-9]{64}", artifact) is None
    ):
        raise EvaluationRecordsError("run requires artifact or hosted service identity")
    return artifact
