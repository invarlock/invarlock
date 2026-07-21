"""Independent verification for runtime behavioral scoring observations.

Runtime providers report per-record facts.  This module authenticates those facts
against a verifier-owned evaluation batch and recomputes the supported behavioral
metric without consuming any provider-supplied aggregate value.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAliasType, TypeGuard, cast

from jsonschema import Draft202012Validator

from invarlock.public_contracts import load_runtime_scoring_observation_schema

from .types import EvaluationBatch, evaluation_input_parts_sha256

RuntimeBehavioralMetric = TypeAliasType(  # noqa: UP040
    "RuntimeBehavioralMetric",
    Literal[
        "exact_match",
        "normalized_nll_per_utf8_byte",
    ],
)

_SUPPORTED_METRICS = frozenset({"exact_match", "normalized_nll_per_utf8_byte"})
_PROVIDER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _is_supported_metric(value: str) -> TypeGuard[RuntimeBehavioralMetric]:
    return value in _SUPPORTED_METRICS


class RuntimeBehavioralObservationError(ValueError):
    """Raised when a scoring observation cannot support a strict metric."""


@dataclass(frozen=True)
class RuntimeBehavioralMetricResult:
    """A verifier-owned aggregate derived from authenticated per-record facts."""

    metric: RuntimeBehavioralMetric
    value: float
    correct_records: int | None
    total_records: int
    aggregate_source_sha256: str


def runtime_scoring_records_sha256(
    records: Sequence[Mapping[str, object]],
) -> str:
    """Hash the exact ordered scoring records using canonical JSON.

    The returned bare lowercase digest is the only accepted meaning of a scoring
    observation's ``aggregate_source_sha256`` field.  Providers can bind the facts
    they emitted, but the verifier remains the sole owner of aggregate metrics.
    """

    encoded = json.dumps(
        list(records),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _schema_error(payload: Mapping[str, object]) -> str | None:
    validator = Draft202012Validator(load_runtime_scoring_observation_schema())
    errors = sorted(
        validator.iter_errors(dict(payload)),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return None
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    return f"scoring observation schema violation at {path}: {error.message}"


def _require_expected_bindings(
    *,
    expected_provider_name: str,
    expected_artifact_identity_sha256: str,
    expected_batch: EvaluationBatch,
) -> None:
    if _PROVIDER_NAME.fullmatch(expected_provider_name) is None:
        raise RuntimeBehavioralObservationError(
            "expected provider name is not canonical"
        )
    if _SHA256.fullmatch(expected_artifact_identity_sha256) is None:
        raise RuntimeBehavioralObservationError(
            "expected artifact identity is not a lowercase sha256 digest"
        )
    if not isinstance(expected_batch, EvaluationBatch):
        raise RuntimeBehavioralObservationError(
            "expected_batch must be an EvaluationBatch"
        )
    for record in expected_batch.records:
        input_sha256 = (
            evaluation_input_parts_sha256(record.input_parts)
            if record.input_parts
            else hashlib.sha256(record.input_text.encode("utf-8")).hexdigest()
        )
        if record.input_sha256 != input_sha256:
            raise RuntimeBehavioralObservationError(
                f"expected record {record.record_id!r} input_sha256 does not match "
                "its authenticated input material"
            )
        if record.expected_output is None:
            raise RuntimeBehavioralObservationError(
                f"expected record {record.record_id!r} requires expected_output"
            )


def _require_observation_bindings(
    payload: Mapping[str, object],
    *,
    expected_provider_name: str,
    expected_artifact_identity_sha256: str,
    expected_batch: EvaluationBatch,
) -> list[Mapping[str, object]]:
    if payload["provider_name"] != expected_provider_name:
        raise RuntimeBehavioralObservationError(
            "scoring observation provider does not match the expected provider"
        )
    if payload["artifact_identity_sha256"] != expected_artifact_identity_sha256:
        raise RuntimeBehavioralObservationError(
            "scoring observation artifact identity does not match the expected artifact"
        )
    if payload["schedule_sha256"] != expected_batch.schedule_sha256:
        raise RuntimeBehavioralObservationError(
            "scoring observation schedule does not match the expected schedule"
        )

    records = cast(list[Mapping[str, object]], payload["records"])
    record_ids = [cast(str, record["record_id"]) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeBehavioralObservationError(
            "scoring observation contains duplicate record IDs"
        )
    expected_pairing = tuple(
        (record.record_id, record.input_sha256) for record in expected_batch.records
    )
    observed_pairing = tuple(
        (cast(str, record["record_id"]), cast(str, record["input_sha256"]))
        for record in records
    )
    if observed_pairing != expected_pairing:
        raise RuntimeBehavioralObservationError(
            "scoring observation pairing does not exactly match the expected batch"
        )
    return records


def _require_authentic_record_facts(
    records: Sequence[Mapping[str, object]],
    *,
    aggregate_source_sha256: object,
) -> None:
    for record in records:
        record_id = cast(str, record["record_id"])
        status = cast(str, record["status"])
        if status != "ok":
            error_code = cast(str, record["error_code"])
            raise RuntimeBehavioralObservationError(
                f"scoring record {record_id!r} failed with {error_code}"
            )
        logprob_sum = record["logprob_sum"]
        if logprob_sum is not None and not math.isfinite(
            float(cast(int | float, logprob_sum))
        ):
            raise RuntimeBehavioralObservationError(
                f"scoring record {record_id!r} contains a non-finite fact"
            )
        output_text = record["output_text"]
        if output_text is not None:
            if not isinstance(output_text, str):
                raise RuntimeBehavioralObservationError(
                    f"scoring record {record_id!r} has invalid output text"
                )
            observed_output_sha256 = record["output_sha256"]
            expected_output_sha256 = hashlib.sha256(
                output_text.encode("utf-8")
            ).hexdigest()
            if observed_output_sha256 != expected_output_sha256:
                raise RuntimeBehavioralObservationError(
                    f"scoring record {record_id!r} output_sha256 does not match "
                    "output_text"
                )

    try:
        records_sha256 = runtime_scoring_records_sha256(records)
    except (TypeError, ValueError) as exc:
        raise RuntimeBehavioralObservationError(
            "scoring observation contains non-canonical or non-finite record facts"
        ) from exc
    if aggregate_source_sha256 != records_sha256:
        raise RuntimeBehavioralObservationError(
            "aggregate_source_sha256 does not match the canonical scoring records"
        )


def _replay_exact_match(
    records: Sequence[Mapping[str, object]],
    *,
    expected_batch: EvaluationBatch,
) -> tuple[float, int]:
    outputs: list[str] = []
    for record in records:
        output_text = record["output_text"]
        if not isinstance(output_text, str):
            raise RuntimeBehavioralObservationError(
                f"scoring record {record['record_id']!r} has no output for "
                "exact-match replay"
            )
        outputs.append(output_text)
    expected_outputs = tuple(
        cast(str, record.expected_output) for record in expected_batch.records
    )
    correct_records = sum(
        output == expected
        for output, expected in zip(outputs, expected_outputs, strict=True)
    )
    return correct_records / len(expected_outputs), correct_records


def _replay_normalized_nll(
    records: Sequence[Mapping[str, object]],
    *,
    expected_batch: EvaluationBatch,
) -> float:
    values: list[float] = []
    for record, expected in zip(records, expected_batch.records, strict=True):
        record_id = cast(str, record["record_id"])
        logprob_sum = record["logprob_sum"]
        token_count = record["token_count"]
        utf8_byte_count = record["utf8_byte_count"]
        expected_output = cast(str, expected.expected_output)
        expected_bytes = len(expected_output.encode("utf-8"))
        if (
            isinstance(logprob_sum, bool)
            or not isinstance(logprob_sum, (int, float))
            or isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
            or isinstance(utf8_byte_count, bool)
            or not isinstance(utf8_byte_count, int)
            or utf8_byte_count <= 0
        ):
            raise RuntimeBehavioralObservationError(
                f"scoring record {record_id!r} lacks normalized NLL facts"
            )
        if utf8_byte_count != expected_bytes:
            raise RuntimeBehavioralObservationError(
                f"scoring record {record_id!r} utf8_byte_count does not match "
                "expected_output"
            )
        value = -float(logprob_sum) / utf8_byte_count
        if not math.isfinite(value) or value < 0:
            raise RuntimeBehavioralObservationError(
                f"scoring record {record_id!r} has invalid normalized NLL"
            )
        values.append(value)
    return math.fsum(values) / len(values)


def verify_runtime_behavioral_observation(
    payload: Mapping[str, object],
    *,
    expected_provider_name: str,
    expected_artifact_identity_sha256: str,
    expected_batch: EvaluationBatch,
    metric: str,
) -> RuntimeBehavioralMetricResult:
    """Authenticate an observation and recompute one supported behavioral metric.

    Strict verification requires exact provider, artifact, schedule, and ordered
    ``(record_id, input_sha256)`` bindings.  Error records, malformed hashes,
    non-finite measurements, missing answers, and provider aggregate fields fail
    closed.  The result is computed only from expected outputs and authenticated
    output text.
    """

    if not _is_supported_metric(metric):
        raise RuntimeBehavioralObservationError(
            f"unsupported metric for behavioral replay: {metric!r}"
        )
    if not isinstance(payload, Mapping):
        raise RuntimeBehavioralObservationError(
            "scoring observation payload must be an object"
        )
    schema_error = _schema_error(payload)
    if schema_error is not None:
        raise RuntimeBehavioralObservationError(schema_error)
    _require_expected_bindings(
        expected_provider_name=expected_provider_name,
        expected_artifact_identity_sha256=expected_artifact_identity_sha256,
        expected_batch=expected_batch,
    )
    if expected_batch.metric != metric:
        raise RuntimeBehavioralObservationError(
            "behavioral replay metric does not match the expected batch"
        )
    records = _require_observation_bindings(
        payload,
        expected_provider_name=expected_provider_name,
        expected_artifact_identity_sha256=expected_artifact_identity_sha256,
        expected_batch=expected_batch,
    )
    _require_authentic_record_facts(
        records,
        aggregate_source_sha256=payload["aggregate_source_sha256"],
    )
    total_records = len(expected_batch.records)
    if metric == "exact_match":
        value, correct_records = _replay_exact_match(
            records, expected_batch=expected_batch
        )
    else:
        value = _replay_normalized_nll(records, expected_batch=expected_batch)
        correct_records = None
    return RuntimeBehavioralMetricResult(
        metric=metric,
        value=value,
        correct_records=correct_records,
        total_records=total_records,
        aggregate_source_sha256=cast(str, payload["aggregate_source_sha256"]),
    )


__all__ = [
    "RuntimeBehavioralMetric",
    "RuntimeBehavioralMetricResult",
    "RuntimeBehavioralObservationError",
    "runtime_scoring_records_sha256",
    "verify_runtime_behavioral_observation",
]
