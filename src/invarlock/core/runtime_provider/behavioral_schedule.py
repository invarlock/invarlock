"""Closed, torch-free behavioral evaluation schedule contract."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes

from ..dataset_identity import (
    HOSTED_DATASET_PROVIDERS,
    canonical_dataset_revision,
)
from .types import EvaluationBatch, EvaluationRecord

RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT = "invarlock/runtime-behavioral-schedule-v1"
MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES = 16 * 1024 * 1024
MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS = 10_000
MAX_RUNTIME_BEHAVIORAL_TEXT_CHARACTERS = 65_536
MAX_RUNTIME_BEHAVIORAL_RECORD_ID_CHARACTERS = 256
MAX_RUNTIME_BEHAVIORAL_DATASET_COORDINATE_CHARACTERS = 512
MAX_RUNTIME_BEHAVIORAL_SPLIT_CHARACTERS = 128

_ROOT_FIELDS = frozenset({"format_version", "dataset_identity", "records"})
_DATASET_IDENTITY_FIELDS = frozenset(
    {"provider", "dataset_name", "config_name", "revision", "split"}
)
_RECORD_FIELDS = frozenset(
    {"record_id", "input_text", "input_sha256", "expected_output"}
)
_RECORD_MATERIAL_FIELDS = frozenset({"record_id", "input_text", "expected_output"})
_PROVIDER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _require_exact_fields(
    payload: Mapping[str, object],
    *,
    expected: frozenset[str],
    field_name: str,
) -> None:
    actual = set(payload)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if unknown:
        details.append(f"unknown {', '.join(unknown)}")
    raise ValueError(
        f"{field_name} must contain exactly the public fields ({'; '.join(details)})"
    )


def _require_trimmed_text(
    value: object,
    *,
    field_name: str,
    max_characters: int,
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    if len(value) > max_characters:
        raise ValueError(f"{field_name} must not exceed {max_characters} characters")
    return value


def _require_safe_logical_text(
    value: object,
    *,
    field_name: str,
    max_characters: int,
) -> str:
    text = _require_trimmed_text(
        value,
        field_name=field_name,
        max_characters=max_characters,
    )
    normalized = text.replace("\\", "/")
    if normalized.startswith("/") or any(
        part in {"", ".", ".."} for part in normalized.split("/")
    ):
        raise ValueError(f"{field_name} must not be an absolute or traversal path")
    if ":/" in normalized or (len(normalized) >= 2 and normalized[1] == ":"):
        raise ValueError(f"{field_name} must not be an absolute or traversal path")
    return text


def _optional_safe_logical_text(
    value: object,
    *,
    field_name: str,
    max_characters: int,
) -> str | None:
    if value is None:
        return None
    return _require_safe_logical_text(
        value,
        field_name=field_name,
        max_characters=max_characters,
    )


@dataclass(frozen=True)
class RuntimeBehavioralDatasetIdentity:
    """Exact portable coordinates for the schedule's source dataset."""

    provider: str
    dataset_name: str | None
    config_name: str | None
    revision: str | None
    split: str

    def __post_init__(self) -> None:
        provider = _require_trimmed_text(
            self.provider,
            field_name="dataset_identity.provider",
            max_characters=64,
        )
        if _PROVIDER_NAME.fullmatch(provider) is None:
            raise ValueError(
                "dataset_identity.provider must be a canonical provider name"
            )
        _optional_safe_logical_text(
            self.dataset_name,
            field_name="dataset_identity.dataset_name",
            max_characters=MAX_RUNTIME_BEHAVIORAL_DATASET_COORDINATE_CHARACTERS,
        )
        _optional_safe_logical_text(
            self.config_name,
            field_name="dataset_identity.config_name",
            max_characters=MAX_RUNTIME_BEHAVIORAL_DATASET_COORDINATE_CHARACTERS,
        )
        if (
            self.revision is not None
            and canonical_dataset_revision(self.revision) is None
        ):
            raise ValueError(
                "dataset_identity.revision must be an immutable lowercase revision"
            )
        _require_safe_logical_text(
            self.split,
            field_name="dataset_identity.split",
            max_characters=MAX_RUNTIME_BEHAVIORAL_SPLIT_CHARACTERS,
        )
        if provider in HOSTED_DATASET_PROVIDERS and (
            self.dataset_name is None
            or self.config_name is None
            or self.revision is None
        ):
            raise ValueError(
                "hosted dataset_identity requires dataset_name, config_name, and revision"
            )

    def to_payload(self) -> dict[str, str | None]:
        return {
            "provider": self.provider,
            "dataset_name": self.dataset_name,
            "config_name": self.config_name,
            "revision": self.revision,
            "split": self.split,
        }


@dataclass(frozen=True)
class RuntimeBehavioralSchedule:
    """Validated ordered schedule plus its recomputed canonical digest."""

    dataset_identity: RuntimeBehavioralDatasetIdentity
    records: tuple[EvaluationRecord, ...]
    schedule_sha256: str
    format_version: str = field(default=RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT, init=False)

    def to_payload(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "dataset_identity": self.dataset_identity.to_payload(),
            "records": [
                {
                    "record_id": record.record_id,
                    "input_text": record.input_text,
                    "input_sha256": record.input_sha256,
                    "expected_output": record.expected_output,
                }
                for record in self.records
            ],
        }

    def evaluation_batch(self) -> EvaluationBatch:
        """Return the provider-neutral batch bound to this schedule digest."""

        return EvaluationBatch(
            schedule_sha256=self.schedule_sha256,
            records=self.records,
        )


def canonical_runtime_behavioral_schedule_json(
    schedule: RuntimeBehavioralSchedule,
) -> bytes:
    """Serialize the exact public schedule material with ordered records."""

    return json.dumps(
        schedule.to_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_payload_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _build_dataset_identity(value: object) -> RuntimeBehavioralDatasetIdentity:
    if not isinstance(value, Mapping):
        raise ValueError("dataset_identity must be an object")
    _require_exact_fields(
        value,
        expected=_DATASET_IDENTITY_FIELDS,
        field_name="dataset_identity",
    )
    return RuntimeBehavioralDatasetIdentity(
        provider=value["provider"],
        dataset_name=value["dataset_name"],
        config_name=value["config_name"],
        revision=value["revision"],
        split=value["split"],
    )


def _build_record(value: object, *, index: int) -> EvaluationRecord:
    field_name = f"records[{index}]"
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    _require_exact_fields(value, expected=_RECORD_FIELDS, field_name=field_name)
    record_id = _require_safe_logical_text(
        value["record_id"],
        field_name=f"{field_name}.record_id",
        max_characters=MAX_RUNTIME_BEHAVIORAL_RECORD_ID_CHARACTERS,
    )
    input_text = value["input_text"]
    expected_output = value["expected_output"]
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError(f"{field_name}.input_text must be non-empty text")
    if len(input_text) > MAX_RUNTIME_BEHAVIORAL_TEXT_CHARACTERS:
        raise ValueError(
            f"{field_name}.input_text must not exceed "
            f"{MAX_RUNTIME_BEHAVIORAL_TEXT_CHARACTERS} characters"
        )
    if not isinstance(expected_output, str) or not expected_output.strip():
        raise ValueError(f"{field_name}.expected_output must be non-empty text")
    if len(expected_output) > MAX_RUNTIME_BEHAVIORAL_TEXT_CHARACTERS:
        raise ValueError(
            f"{field_name}.expected_output must not exceed "
            f"{MAX_RUNTIME_BEHAVIORAL_TEXT_CHARACTERS} characters"
        )
    input_sha256 = value["input_sha256"]
    expected_sha256 = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
    if input_sha256 != expected_sha256:
        raise ValueError(f"{field_name}.input_sha256 does not match input_text")
    return EvaluationRecord(
        record_id=record_id,
        input_text=input_text,
        input_sha256=expected_sha256,
        expected_output=expected_output,
    )


def build_runtime_behavioral_schedule(
    payload: Mapping[str, object],
) -> RuntimeBehavioralSchedule:
    """Validate exact schedule material and recompute its canonical SHA-256."""

    if not isinstance(payload, Mapping):
        raise ValueError("runtime behavioral schedule must be an object")
    _require_exact_fields(payload, expected=_ROOT_FIELDS, field_name="schedule")
    if payload["format_version"] != RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT:
        raise ValueError("runtime behavioral schedule format_version is unsupported")
    dataset_identity = _build_dataset_identity(payload["dataset_identity"])
    raw_records = payload["records"]
    if (
        not isinstance(raw_records, Sequence)
        or isinstance(raw_records, (str, bytes, bytearray))
        or not raw_records
    ):
        raise ValueError("records must be a non-empty array")
    if len(raw_records) > MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS:
        raise ValueError(
            f"records must not exceed {MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS} entries"
        )
    records = tuple(
        _build_record(value, index=index) for index, value in enumerate(raw_records)
    )
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("record IDs must be unique within a schedule")

    canonical_payload: dict[str, object] = {
        "format_version": RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT,
        "dataset_identity": dataset_identity.to_payload(),
        "records": [
            {
                "record_id": record.record_id,
                "input_text": record.input_text,
                "input_sha256": record.input_sha256,
                "expected_output": record.expected_output,
            }
            for record in records
        ],
    }
    canonical_payload_json = _canonical_payload_json(canonical_payload)
    if len(canonical_payload_json) > MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES:
        raise ValueError(
            "runtime behavioral schedule exceeds the "
            f"{MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES}-byte size limit"
        )
    schedule_sha256 = hashlib.sha256(canonical_payload_json).hexdigest()
    return RuntimeBehavioralSchedule(
        dataset_identity=dataset_identity,
        records=records,
        schedule_sha256=schedule_sha256,
    )


def build_runtime_behavioral_schedule_from_material(
    *,
    dataset_identity: Mapping[str, object],
    records: Sequence[object],
) -> RuntimeBehavioralSchedule:
    """Build canonical schedule material while deriving every input digest."""

    materialized_records: list[dict[str, object]] = []
    for index, value in enumerate(records):
        field_name = f"records[{index}]"
        if not isinstance(value, Mapping):
            raise ValueError(f"{field_name} must be an object")
        _require_exact_fields(
            value,
            expected=_RECORD_MATERIAL_FIELDS,
            field_name=field_name,
        )
        input_text = value["input_text"]
        if not isinstance(input_text, str):
            raise ValueError(f"{field_name}.input_text must be text")
        materialized_records.append(
            {
                "record_id": value["record_id"],
                "input_text": input_text,
                "input_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
                "expected_output": value["expected_output"],
            }
        )
    return build_runtime_behavioral_schedule(
        {
            "format_version": RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT,
            "dataset_identity": dict(dataset_identity),
            "records": materialized_records,
        }
    )


def parse_runtime_behavioral_schedule_json(text: str) -> RuntimeBehavioralSchedule:
    """Parse strict JSON and build a closed runtime behavioral schedule."""

    if not isinstance(text, str):
        raise ValueError("runtime behavioral schedule JSON must be text")
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES:
        raise ValueError(
            "runtime behavioral schedule exceeds the "
            f"{MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES}-byte size limit"
        )
    payload = parse_json_bytes(encoded, label="runtime behavioral schedule")
    if not isinstance(payload, Mapping):
        raise ValueError("runtime behavioral schedule must be an object")
    return build_runtime_behavioral_schedule(payload)


def load_runtime_behavioral_schedule(
    path: str | Path,
) -> RuntimeBehavioralSchedule:
    """Load one UTF-8 schedule file under the strict public contract."""

    payload = read_regular_file_bytes(
        Path(path),
        label="runtime behavioral schedule",
        max_bytes=MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES,
    )
    decoded = parse_json_bytes(payload, label="runtime behavioral schedule")
    if not isinstance(decoded, Mapping):
        raise ValueError("runtime behavioral schedule must be an object")
    return build_runtime_behavioral_schedule(decoded)


__all__ = [
    "MAX_RUNTIME_BEHAVIORAL_DATASET_COORDINATE_CHARACTERS",
    "MAX_RUNTIME_BEHAVIORAL_RECORD_ID_CHARACTERS",
    "MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES",
    "MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS",
    "MAX_RUNTIME_BEHAVIORAL_SPLIT_CHARACTERS",
    "MAX_RUNTIME_BEHAVIORAL_TEXT_CHARACTERS",
    "RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT",
    "RuntimeBehavioralDatasetIdentity",
    "RuntimeBehavioralSchedule",
    "build_runtime_behavioral_schedule",
    "build_runtime_behavioral_schedule_from_material",
    "canonical_runtime_behavioral_schedule_json",
    "load_runtime_behavioral_schedule",
    "parse_runtime_behavioral_schedule_json",
]
