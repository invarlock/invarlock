from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence

import pytest

from invarlock.core.runtime_provider import EvaluationBatch, EvaluationRecord
from invarlock.reporting.validation.runtime_behavioral_observation import (
    RuntimeBehavioralObservationError,
    verify_runtime_behavioral_observation,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _records_sha256(records: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _batch() -> EvaluationBatch:
    return EvaluationBatch(
        schedule_sha256=_sha256("schedule"),
        records=(
            EvaluationRecord(
                record_id="sample-1",
                input_text="What is the capital of France?",
                input_sha256=_sha256("What is the capital of France?"),
                expected_output="Paris",
            ),
            EvaluationRecord(
                record_id="sample-2",
                input_text="Choose A or B.",
                input_sha256=_sha256("Choose A or B."),
                expected_output="A",
            ),
        ),
    )


def _record(
    expected: EvaluationRecord,
    *,
    output: str,
    logprob_sum: float | None = -1.0,
) -> dict[str, object]:
    return {
        "record_id": expected.record_id,
        "input_sha256": expected.input_sha256,
        "status": "ok",
        "output_text": output,
        "output_sha256": _sha256(output),
        "logprob_sum": logprob_sum,
        "token_count": 1 if logprob_sum is not None else None,
        "utf8_byte_count": 1 if logprob_sum is not None else None,
        "error_code": None,
    }


def _payload() -> dict[str, object]:
    batch = _batch()
    records = [
        _record(batch.records[0], output="  PARIS\n"),
        _record(batch.records[1], output="B"),
    ]
    return {
        "format_version": "invarlock/runtime-scoring-observation-v1",
        "provider_name": "llama_cpp",
        "artifact_identity_sha256": _sha256("artifact"),
        "schedule_sha256": batch.schedule_sha256,
        "records": records,
        "aggregate_source_sha256": _records_sha256(records),
    }


def test_verifier_recomputes_exact_match_from_record_facts() -> None:
    result = verify_runtime_behavioral_observation(
        _payload(),
        expected_provider_name="llama_cpp",
        expected_artifact_identity_sha256=_sha256("artifact"),
        expected_batch=_batch(),
        metric="exact_match",
    )

    assert result.metric == "exact_match"
    assert result.value == 0.5
    assert result.correct_records == 1
    assert result.total_records == 2
    assert result.aggregate_source_sha256 == _payload()["aggregate_source_sha256"]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("provider_name", "other_provider", "provider"),
        ("artifact_identity_sha256", "f" * 64, "artifact"),
        ("schedule_sha256", "e" * 64, "schedule"),
    ],
)
def test_verifier_rejects_binding_drift(
    field: str, replacement: object, message: str
) -> None:
    payload = _payload()
    payload[field] = replacement

    with pytest.raises(RuntimeBehavioralObservationError, match=message):
        verify_runtime_behavioral_observation(
            payload,
            expected_provider_name="llama_cpp",
            expected_artifact_identity_sha256=_sha256("artifact"),
            expected_batch=_batch(),
            metric="exact_match",
        )


def test_verifier_requires_exact_ordered_record_pairing() -> None:
    payload = _payload()
    raw_records = payload["records"]
    assert isinstance(raw_records, list)
    records = list(reversed(raw_records))
    payload["records"] = records
    payload["aggregate_source_sha256"] = _records_sha256(records)

    with pytest.raises(RuntimeBehavioralObservationError, match="pairing"):
        verify_runtime_behavioral_observation(
            payload,
            expected_provider_name="llama_cpp",
            expected_artifact_identity_sha256=_sha256("artifact"),
            expected_batch=_batch(),
            metric="exact_match",
        )


def test_verifier_rejects_duplicate_record_ids_before_aggregation() -> None:
    payload = _payload()
    records = copy.deepcopy(payload["records"])
    assert isinstance(records, list)
    assert isinstance(records[1], dict)
    assert isinstance(records[0], dict)
    records[1]["record_id"] = records[0]["record_id"]
    payload["records"] = records
    payload["aggregate_source_sha256"] = _records_sha256(records)

    with pytest.raises(RuntimeBehavioralObservationError, match="duplicate"):
        verify_runtime_behavioral_observation(
            payload,
            expected_provider_name="llama_cpp",
            expected_artifact_identity_sha256=_sha256("artifact"),
            expected_batch=_batch(),
            metric="exact_match",
        )


def test_verifier_rejects_backend_error_records() -> None:
    payload = _payload()
    records = copy.deepcopy(payload["records"])
    assert isinstance(records, list)
    assert isinstance(records[0], dict)
    records[0].update(
        {
            "status": "error",
            "output_text": None,
            "output_sha256": None,
            "logprob_sum": None,
            "token_count": None,
            "utf8_byte_count": None,
            "error_code": "backend_error",
        }
    )
    payload["records"] = records
    payload["aggregate_source_sha256"] = _records_sha256(records)

    with pytest.raises(RuntimeBehavioralObservationError, match="backend_error"):
        verify_runtime_behavioral_observation(
            payload,
            expected_provider_name="llama_cpp",
            expected_artifact_identity_sha256=_sha256("artifact"),
            expected_batch=_batch(),
            metric="exact_match",
        )


def test_verifier_rejects_output_hash_tampering() -> None:
    payload = _payload()
    records = copy.deepcopy(payload["records"])
    assert isinstance(records, list)
    assert isinstance(records[0], dict)
    records[0]["output_sha256"] = "d" * 64
    payload["records"] = records
    payload["aggregate_source_sha256"] = _records_sha256(records)

    with pytest.raises(RuntimeBehavioralObservationError, match="output_sha256"):
        verify_runtime_behavioral_observation(
            payload,
            expected_provider_name="llama_cpp",
            expected_artifact_identity_sha256=_sha256("artifact"),
            expected_batch=_batch(),
            metric="exact_match",
        )


def test_verifier_rejects_aggregate_source_tampering() -> None:
    payload = _payload()
    payload["aggregate_source_sha256"] = "d" * 64

    with pytest.raises(RuntimeBehavioralObservationError, match="aggregate_source"):
        verify_runtime_behavioral_observation(
            payload,
            expected_provider_name="llama_cpp",
            expected_artifact_identity_sha256=_sha256("artifact"),
            expected_batch=_batch(),
            metric="exact_match",
        )


def test_verifier_rejects_non_finite_record_facts() -> None:
    payload = _payload()
    records = copy.deepcopy(payload["records"])
    assert isinstance(records, list)
    assert isinstance(records[0], dict)
    records[0]["logprob_sum"] = float("nan")
    payload["records"] = records

    with pytest.raises(RuntimeBehavioralObservationError, match="finite"):
        verify_runtime_behavioral_observation(
            payload,
            expected_provider_name="llama_cpp",
            expected_artifact_identity_sha256=_sha256("artifact"),
            expected_batch=_batch(),
            metric="exact_match",
        )


def test_verifier_rejects_provider_supplied_aggregate_values() -> None:
    payload = _payload()
    payload["exact_match"] = 1.0

    with pytest.raises(RuntimeBehavioralObservationError, match="schema"):
        verify_runtime_behavioral_observation(
            payload,
            expected_provider_name="llama_cpp",
            expected_artifact_identity_sha256=_sha256("artifact"),
            expected_batch=_batch(),
            metric="exact_match",
        )


def test_verifier_rejects_unbound_expected_inputs_and_answers() -> None:
    batch = _batch()
    bad_hash_batch = EvaluationBatch(
        schedule_sha256=batch.schedule_sha256,
        records=(
            EvaluationRecord(
                record_id=batch.records[0].record_id,
                input_text=batch.records[0].input_text,
                input_sha256="a" * 64,
                expected_output=batch.records[0].expected_output,
            ),
            batch.records[1],
        ),
    )
    missing_answer_batch = EvaluationBatch(
        schedule_sha256=batch.schedule_sha256,
        records=(
            copy.deepcopy(batch.records[0]),
            EvaluationRecord(
                record_id=batch.records[1].record_id,
                input_text=batch.records[1].input_text,
                input_sha256=batch.records[1].input_sha256,
                expected_output=None,
            ),
        ),
    )

    with pytest.raises(RuntimeBehavioralObservationError, match="input_sha256"):
        verify_runtime_behavioral_observation(
            _payload(),
            expected_provider_name="llama_cpp",
            expected_artifact_identity_sha256=_sha256("artifact"),
            expected_batch=bad_hash_batch,
            metric="exact_match",
        )
    with pytest.raises(RuntimeBehavioralObservationError, match="expected_output"):
        verify_runtime_behavioral_observation(
            _payload(),
            expected_provider_name="llama_cpp",
            expected_artifact_identity_sha256=_sha256("artifact"),
            expected_batch=missing_answer_batch,
            metric="exact_match",
        )


@pytest.mark.parametrize("metric", ["multiple_choice_accuracy", "provider_accuracy"])
def test_verifier_rejects_unsupported_metrics(metric: str) -> None:
    with pytest.raises(RuntimeBehavioralObservationError, match="unsupported metric"):
        verify_runtime_behavioral_observation(
            _payload(),
            expected_provider_name="llama_cpp",
            expected_artifact_identity_sha256=_sha256("artifact"),
            expected_batch=_batch(),
            metric=metric,
        )
