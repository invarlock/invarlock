"""Deterministic preparation of a paired schedule from pinned local records."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes

from .runtime_provider.behavioral_schedule import (
    MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES,
    MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS,
    RuntimeBehavioralDatasetIdentity,
    RuntimeBehavioralSchedule,
    build_runtime_behavioral_schedule_from_material,
)
from .runtime_provider.types import RuntimeTask

LOCAL_RECORDS_JSONL_FORMAT = "jsonl"
LOCAL_RECORDS_DATASET_CONFIG = "evaluation-records-jsonl-v1"
LOCAL_DATASET_PREPARATION_FORMAT = "invarlock/local-jsonl-preparation-v1"
MAX_LOCAL_RECORD_FIELD_NAME_CHARACTERS = 128

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,127}$")
_CONTENT_ROLE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class LocalDatasetRequest:
    """Pinned local source and the explicit mapping used to prepare a schedule."""

    path: Path
    sha256: str
    name: str
    split: str
    input_field: str
    expected_output_field: str
    id_field: str | None = None
    content_role: str | None = None
    content_id_field: str | None = None
    content_sha256_field: str | None = None
    content_byte_length_field: str | None = None
    content_media_type_field: str | None = None
    limit: int | None = None
    format: str = LOCAL_RECORDS_JSONL_FORMAT

    def __post_init__(self) -> None:
        if self.format != LOCAL_RECORDS_JSONL_FORMAT:
            raise ValueError(f"dataset format must be {LOCAL_RECORDS_JSONL_FORMAT!r}")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("dataset sha256 must be a lowercase SHA-256 digest")
        mapped_fields = {
            "input_field": self.input_field,
            "expected_output_field": self.expected_output_field,
        }
        if self.id_field is not None:
            mapped_fields["id_field"] = self.id_field
        content_fields = {
            "content_id_field": self.content_id_field,
            "content_sha256_field": self.content_sha256_field,
            "content_byte_length_field": self.content_byte_length_field,
            "content_media_type_field": self.content_media_type_field,
        }
        has_any_content_mapping = any(
            value is not None for value in content_fields.values()
        )
        has_content_mapping = all(
            value is not None for value in content_fields.values()
        )
        if has_any_content_mapping and not has_content_mapping:
            raise ValueError("dataset content field mappings must be provided together")
        if has_content_mapping != (self.content_role is not None):
            raise ValueError(
                "dataset content_role and content field mappings must be provided "
                "together"
            )
        if self.content_role is not None and (
            _CONTENT_ROLE.fullmatch(self.content_role) is None
        ):
            raise ValueError("dataset content_role must be a canonical input role")
        mapped_fields.update(
            {
                label: value
                for label, value in content_fields.items()
                if value is not None
            }
        )
        for label, field_name in mapped_fields.items():
            if _FIELD_NAME.fullmatch(field_name) is None:
                raise ValueError(
                    f"dataset {label} must be a simple top-level JSON field name"
                )
        if len(set(mapped_fields.values())) != len(mapped_fields):
            raise ValueError("dataset field mappings must reference distinct fields")
        if self.limit is not None and not (
            1 <= self.limit <= MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS
        ):
            raise ValueError(
                "dataset limit must be between 1 and "
                f"{MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS}"
            )
        RuntimeBehavioralDatasetIdentity(
            provider="local",
            dataset_name=self.name,
            config_name=LOCAL_RECORDS_DATASET_CONFIG,
            revision=self.sha256,
            split=self.split,
        )


def local_dataset_preparation_payload(
    source: LocalDatasetRequest,
    schedule: RuntimeBehavioralSchedule,
) -> dict[str, object]:
    """Return the path-free preparation intent bound into run-mode evidence."""

    payload: dict[str, object] = {
        "format_version": LOCAL_DATASET_PREPARATION_FORMAT,
        "source_sha256": source.sha256,
        "source_format": source.format,
        "name": source.name,
        "split": source.split,
        "input_field": source.input_field,
        "expected_output_field": source.expected_output_field,
        "id_field": source.id_field,
        "content_role": source.content_role,
        "content_id_field": source.content_id_field,
        "content_sha256_field": source.content_sha256_field,
        "content_byte_length_field": source.content_byte_length_field,
        "content_media_type_field": source.content_media_type_field,
        "limit": source.limit,
        "selected_record_count": len(schedule.records),
    }
    validate_local_dataset_preparation(payload, schedule)
    return payload


def validate_local_dataset_preparation(
    payload: Mapping[str, object],
    schedule: RuntimeBehavioralSchedule,
) -> None:
    """Cross-check signed run preparation intent against its canonical schedule."""

    expected_fields = {
        "format_version",
        "source_sha256",
        "source_format",
        "name",
        "split",
        "input_field",
        "expected_output_field",
        "id_field",
        "content_role",
        "content_id_field",
        "content_sha256_field",
        "content_byte_length_field",
        "content_media_type_field",
        "limit",
        "selected_record_count",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("run dataset preparation fields are invalid")
    if payload.get("format_version") != LOCAL_DATASET_PREPARATION_FORMAT:
        raise ValueError("run dataset preparation format_version is invalid")
    if payload.get("source_format") != LOCAL_RECORDS_JSONL_FORMAT:
        raise ValueError("run dataset preparation source_format is invalid")
    source_sha256 = payload.get("source_sha256")
    if not isinstance(source_sha256, str) or _SHA256.fullmatch(source_sha256) is None:
        raise ValueError("run dataset preparation source_sha256 is invalid")

    mapped_fields: list[str] = []
    content_field_names = (
        "content_id_field",
        "content_sha256_field",
        "content_byte_length_field",
        "content_media_type_field",
    )
    if any(payload.get(name) is not None for name in content_field_names) and not all(
        payload.get(name) is not None for name in content_field_names
    ):
        raise ValueError(
            "run dataset preparation content field mappings must be provided together"
        )
    has_content_mapping = all(
        payload.get(name) is not None for name in content_field_names
    )
    content_role = payload.get("content_role")
    if has_content_mapping != (content_role is not None):
        raise ValueError(
            "run dataset preparation content_role and content field mappings must "
            "be provided together"
        )
    if content_role is not None and (
        not isinstance(content_role, str)
        or _CONTENT_ROLE.fullmatch(content_role) is None
    ):
        raise ValueError("run dataset preparation content_role is invalid")
    for field_name in (
        "input_field",
        "expected_output_field",
        "id_field",
        *content_field_names,
    ):
        value = payload.get(field_name)
        if field_name in {"id_field", *content_field_names} and value is None:
            continue
        if not isinstance(value, str) or _FIELD_NAME.fullmatch(value) is None:
            raise ValueError(f"run dataset preparation {field_name} is invalid")
        mapped_fields.append(value)
    if len(mapped_fields) != len(set(mapped_fields)):
        raise ValueError("run dataset preparation field mappings must be distinct")

    identity = schedule.dataset_identity
    expected_identity = {
        "provider": "local",
        "dataset_name": payload.get("name"),
        "config_name": LOCAL_RECORDS_DATASET_CONFIG,
        "revision": source_sha256,
        "split": payload.get("split"),
    }
    if identity.to_payload() != expected_identity:
        raise ValueError(
            "run dataset preparation does not match the canonical schedule identity"
        )

    selected_count = payload.get("selected_record_count")
    if (
        isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or selected_count != len(schedule.records)
    ):
        raise ValueError(
            "run dataset preparation selected_record_count does not match the schedule"
        )
    limit = payload.get("limit")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit != selected_count
    ):
        raise ValueError("run dataset preparation limit does not match the selection")
    if payload.get("id_field") is None:
        expected_ids = [f"record/{index:08d}" for index in range(selected_count)]
        if [record.record_id for record in schedule.records] != expected_ids:
            raise ValueError(
                "run dataset preparation positional IDs do not match the schedule"
            )
    records_have_content = [
        any(part.kind == "content" for part in record.input_parts)
        for record in schedule.records
    ]
    if has_content_mapping != all(records_have_content):
        raise ValueError(
            "run dataset preparation content mappings do not match the schedule"
        )
    if schedule.task == "vision_text_generation" and not has_content_mapping:
        raise ValueError(
            "vision_text_generation requires run dataset content field mappings"
        )
    if schedule.task == "vision_text_generation" and content_role != "image":
        raise ValueError("vision_text_generation requires content_role 'image'")
    if schedule.task in {"text_causal", "text_seq2seq", "masked_language"} and (
        has_content_mapping
    ):
        raise ValueError(
            f"{schedule.task} does not accept run dataset content field mappings"
        )


def _mapped_text(
    value: Mapping[str, object],
    field_name: str,
    *,
    line_number: int,
) -> str:
    mapped = value.get(field_name)
    if not isinstance(mapped, str):
        raise ValueError(
            f"dataset line {line_number} field {field_name!r} must be text"
        )
    return mapped


def _record_id(
    value: Mapping[str, object],
    source: LocalDatasetRequest,
    *,
    source_index: int,
    line_number: int,
) -> str:
    if source.id_field is None:
        return f"record/{source_index:08d}"
    return _mapped_text(value, source.id_field, line_number=line_number)


def _mapped_positive_int(
    value: Mapping[str, object],
    field_name: str,
    *,
    line_number: int,
) -> int:
    mapped = value.get(field_name)
    if isinstance(mapped, bool) or not isinstance(mapped, int) or mapped <= 0:
        raise ValueError(
            f"dataset line {line_number} field {field_name!r} "
            "must be a positive integer"
        )
    return mapped


def prepare_local_evaluation_schedule_bytes(
    source: LocalDatasetRequest,
    payload: bytes,
    *,
    task: RuntimeTask = "text_causal",
) -> RuntimeBehavioralSchedule:
    """Authenticate local JSONL bytes and derive one canonical ordered schedule.

    Input order is preserved. When no ID field is mapped, IDs are derived from
    the zero-based source position, so the same pinned bytes and preparation
    request always produce the same IDs and schedule digest.
    """

    if not isinstance(payload, bytes):
        raise TypeError("local evaluation dataset payload must be bytes")
    if len(payload) > MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES:
        raise ValueError(
            "local evaluation dataset exceeds the "
            f"{MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES}-byte size limit"
        )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != source.sha256:
        raise ValueError(
            "local evaluation dataset sha256 mismatch: "
            f"expected {source.sha256}, got {actual_sha256}"
        )
    has_content = source.content_id_field is not None
    if task == "vision_text_generation" and not has_content:
        raise ValueError(
            "vision_text_generation requires dataset content field mappings"
        )
    if task == "vision_text_generation" and source.content_role != "image":
        raise ValueError("vision_text_generation requires content_role 'image'")
    if task in {"text_causal", "text_seq2seq", "masked_language"} and has_content:
        raise ValueError(f"{task} does not accept dataset content field mappings")
    lines = payload.splitlines()
    if not lines:
        raise ValueError("local evaluation dataset must contain at least one record")
    if source.limit is not None and len(lines) < source.limit:
        raise ValueError(
            "local evaluation dataset contains fewer records than the requested limit"
        )

    selected_count = len(lines) if source.limit is None else source.limit
    if selected_count > MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS:
        raise ValueError(
            "local evaluation dataset selection exceeds the "
            f"{MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS}-record limit"
        )

    records: list[dict[str, object]] = []
    for source_index, raw_line in enumerate(lines[:selected_count]):
        line_number = source_index + 1
        if not raw_line.strip():
            raise ValueError(f"local evaluation dataset line {line_number} is blank")
        decoded = parse_json_bytes(
            raw_line,
            label=f"local evaluation dataset line {line_number}",
        )
        if not isinstance(decoded, Mapping):
            raise ValueError(
                f"local evaluation dataset line {line_number} must be an object"
            )
        prompt = _mapped_text(
            decoded,
            source.input_field,
            line_number=line_number,
        )
        record: dict[str, object] = {
            "record_id": _record_id(
                decoded,
                source,
                source_index=source_index,
                line_number=line_number,
            ),
            "input_text": prompt,
            "expected_output": _mapped_text(
                decoded,
                source.expected_output_field,
                line_number=line_number,
            ),
        }
        if has_content:
            assert source.content_role is not None
            assert source.content_id_field is not None
            assert source.content_sha256_field is not None
            assert source.content_byte_length_field is not None
            assert source.content_media_type_field is not None
            content_id = _mapped_text(
                decoded, source.content_id_field, line_number=line_number
            )
            content_sha256 = _mapped_text(
                decoded, source.content_sha256_field, line_number=line_number
            )
            content_media_type = _mapped_text(
                decoded, source.content_media_type_field, line_number=line_number
            )
            record.pop("input_text")
            record["input_parts"] = [
                {
                    "kind": "content",
                    "role": source.content_role,
                    "content_id": content_id,
                    "media_type": content_media_type,
                    "byte_length": _mapped_positive_int(
                        decoded,
                        source.content_byte_length_field,
                        line_number=line_number,
                    ),
                    "sha256": content_sha256,
                },
                {
                    "kind": "text",
                    "role": "prompt",
                    "text": prompt,
                    "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                },
            ]
        records.append(record)

    return build_runtime_behavioral_schedule_from_material(
        dataset_identity={
            "provider": "local",
            "dataset_name": source.name,
            "config_name": LOCAL_RECORDS_DATASET_CONFIG,
            "revision": source.sha256,
            "split": source.split,
        },
        records=records,
        task=task,
    )


def prepare_local_evaluation_schedule(
    source: LocalDatasetRequest,
    *,
    task: RuntimeTask = "text_causal",
) -> RuntimeBehavioralSchedule:
    """Read and prepare a pinned local dataset outside a request transaction."""

    payload = read_regular_file_bytes(
        source.path,
        label="local evaluation dataset",
        max_bytes=MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES,
    )
    return prepare_local_evaluation_schedule_bytes(source, payload, task=task)


__all__ = [
    "LOCAL_DATASET_PREPARATION_FORMAT",
    "LOCAL_RECORDS_DATASET_CONFIG",
    "LOCAL_RECORDS_JSONL_FORMAT",
    "MAX_LOCAL_RECORD_FIELD_NAME_CHARACTERS",
    "LocalDatasetRequest",
    "local_dataset_preparation_payload",
    "prepare_local_evaluation_schedule",
    "prepare_local_evaluation_schedule_bytes",
    "validate_local_dataset_preparation",
]
