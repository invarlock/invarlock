from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from invarlock.core.runtime_provider.behavioral_schedule import (
    build_runtime_behavioral_schedule,
)
from invarlock.core.schedule_preparation import (
    LOCAL_RECORDS_DATASET_CONFIG,
    LocalDatasetRequest,
    local_dataset_preparation_payload,
    prepare_local_evaluation_schedule,
    prepare_local_evaluation_schedule_bytes,
    validate_local_dataset_preparation,
)
from invarlock.evidence_pack_contract import dataset_preparation_binding_errors


def _jsonl(*records: Mapping[str, object]) -> bytes:
    return b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for record in records
    )


def _source(path: Path, payload: bytes, **overrides: object) -> LocalDatasetRequest:
    values: dict[str, object] = {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "name": "release-regression",
        "split": "validation",
        "input_field": "prompt",
        "expected_output_field": "continuation",
        "id_field": "example_id",
    }
    values.update(overrides)
    return LocalDatasetRequest(**values)  # type: ignore[arg-type]


def test_prepare_local_schedule_authenticates_maps_and_preserves_order(
    tmp_path: Path,
) -> None:
    payload = _jsonl(
        {
            "example_id": "case/2",
            "prompt": "Second prompt",
            "continuation": "Second answer",
            "ignored_metadata": {"source": "fixture"},
        },
        {
            "example_id": "case/1",
            "prompt": "First prompt",
            "continuation": "First answer",
            "ignored_metadata": {"source": "fixture"},
        },
    )
    path = tmp_path / "records.jsonl"
    path.write_bytes(payload)

    schedule = prepare_local_evaluation_schedule(_source(path, payload))

    assert [record.record_id for record in schedule.records] == ["case/2", "case/1"]
    assert [record.input_text for record in schedule.records] == [
        "Second prompt",
        "First prompt",
    ]
    assert schedule.dataset_identity.to_payload() == {
        "provider": "local",
        "dataset_name": "release-regression",
        "config_name": LOCAL_RECORDS_DATASET_CONFIG,
        "revision": hashlib.sha256(payload).hexdigest(),
        "split": "validation",
    }


def test_generated_ids_are_stable_source_positions(tmp_path: Path) -> None:
    payload = _jsonl(
        {"prompt": "One", "continuation": "A"},
        {"prompt": "Two", "continuation": "B"},
    )
    source = _source(tmp_path / "records.jsonl", payload, id_field=None)

    first = prepare_local_evaluation_schedule_bytes(source, payload)
    second = prepare_local_evaluation_schedule_bytes(source, payload)

    assert [record.record_id for record in first.records] == [
        "record/00000000",
        "record/00000001",
    ]
    assert second.schedule_sha256 == first.schedule_sha256


def test_source_order_changes_schedule_digest(tmp_path: Path) -> None:
    one = {"example_id": "one", "prompt": "One", "continuation": "A"}
    two = {"example_id": "two", "prompt": "Two", "continuation": "B"}
    original = _jsonl(one, two)
    reordered = _jsonl(two, one)

    original_schedule = prepare_local_evaluation_schedule_bytes(
        _source(tmp_path / "original.jsonl", original), original
    )
    reordered_schedule = prepare_local_evaluation_schedule_bytes(
        _source(tmp_path / "reordered.jsonl", reordered), reordered
    )

    assert reordered_schedule.schedule_sha256 != original_schedule.schedule_sha256


def test_limit_selects_exact_prefix_and_is_bound_to_schedule(tmp_path: Path) -> None:
    payload = _jsonl(
        {"example_id": "one", "prompt": "One", "continuation": "A"},
        {"example_id": "two", "prompt": "Two", "continuation": "B"},
        {"example_id": "three", "prompt": "Three", "continuation": "C"},
    )
    limited = prepare_local_evaluation_schedule_bytes(
        _source(tmp_path / "records.jsonl", payload, limit=2), payload
    )
    complete = prepare_local_evaluation_schedule_bytes(
        _source(tmp_path / "records.jsonl", payload), payload
    )

    assert [record.record_id for record in limited.records] == ["one", "two"]
    assert limited.schedule_sha256 != complete.schedule_sha256


def test_path_free_preparation_descriptor_binds_mapping_and_selection(
    tmp_path: Path,
) -> None:
    payload = _jsonl(
        {"example_id": "one", "prompt": "One", "continuation": "A"},
        {"example_id": "two", "prompt": "Two", "continuation": "B"},
    )
    source = _source(tmp_path / "private-source.jsonl", payload, limit=2)
    schedule = prepare_local_evaluation_schedule_bytes(source, payload)

    descriptor = local_dataset_preparation_payload(source, schedule)

    assert "path" not in descriptor
    assert descriptor == {
        "format_version": "invarlock/local-jsonl-preparation-v1",
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_format": "jsonl",
        "name": "release-regression",
        "split": "validation",
        "input_field": "prompt",
        "expected_output_field": "continuation",
        "id_field": "example_id",
        "content_role": None,
        "content_id_field": None,
        "content_sha256_field": None,
        "content_byte_length_field": None,
        "content_media_type_field": None,
        "limit": 2,
        "selected_record_count": 2,
    }
    assert (
        dataset_preparation_binding_errors(
            {
                "comparison": {"dataset": descriptor},
                "execution": {"mode": "run"},
            },
            schedule,
        )
        == []
    )


def test_vision_schedule_preparation_emits_path_free_authenticated_parts(
    tmp_path: Path,
) -> None:
    payload = _jsonl(
        {
            "example_id": "vision-1",
            "prompt": "Describe the image.",
            "continuation": "A square.",
            "content_id": "image_1",
            "content_sha256": "a" * 64,
            "content_bytes": 123,
            "content_media_type": "image/png",
        }
    )
    source = LocalDatasetRequest(
        path=tmp_path / "records.jsonl",
        sha256=hashlib.sha256(payload).hexdigest(),
        name="vision-evaluation",
        split="validation",
        input_field="prompt",
        expected_output_field="continuation",
        id_field="example_id",
        content_role="image",
        content_id_field="content_id",
        content_sha256_field="content_sha256",
        content_byte_length_field="content_bytes",
        content_media_type_field="content_media_type",
    )

    schedule = prepare_local_evaluation_schedule_bytes(
        source, payload, task="vision_text_generation"
    )
    descriptor = local_dataset_preparation_payload(source, schedule)

    assert schedule.task == "vision_text_generation"
    assert [part.kind for part in schedule.records[0].input_parts] == [
        "content",
        "text",
    ]
    assert schedule.records[0].input_parts[0].content_id == "image_1"
    assert schedule.records[0].input_parts[0].role == "image"
    assert descriptor["content_role"] == "image"
    assert descriptor["content_id_field"] == "content_id"
    assert "path" not in str(schedule.to_payload()).lower()


@pytest.mark.parametrize("task", ["text_causal", "text_seq2seq", "masked_language"])
def test_text_tasks_reject_content_field_mappings(tmp_path: Path, task: str) -> None:
    payload = _jsonl(
        {
            "prompt": "Prompt",
            "continuation": "Answer",
            "content_id": "image_1",
            "content_sha256": "a" * 64,
            "content_bytes": 1,
            "content_media_type": "image/png",
        }
    )
    source = LocalDatasetRequest(
        path=tmp_path / "records.jsonl",
        sha256=hashlib.sha256(payload).hexdigest(),
        name="invalid-text-content",
        split="validation",
        input_field="prompt",
        expected_output_field="continuation",
        content_role="image",
        content_id_field="content_id",
        content_sha256_field="content_sha256",
        content_byte_length_field="content_bytes",
        content_media_type_field="content_media_type",
    )
    with pytest.raises(ValueError, match="does not accept dataset content"):
        prepare_local_evaluation_schedule_bytes(source, payload, task=task)


def test_future_canonical_text_task_preserves_task_without_new_execution(
    tmp_path: Path,
) -> None:
    payload = _jsonl(
        {
            "example_id": "future-1",
            "prompt": "Authenticate this task-neutral text input.",
            "continuation": "accepted",
        }
    )

    schedule = prepare_local_evaluation_schedule_bytes(
        _source(tmp_path / "records.jsonl", payload),
        payload,
        task="audio_transcription_review",
    )

    assert schedule.task == "audio_transcription_review"
    assert schedule.evaluation_batch().task == "audio_transcription_review"
    assert [part.to_payload() for part in schedule.records[0].input_parts] == [
        {
            "kind": "text",
            "role": "prompt",
            "text": "Authenticate this task-neutral text input.",
            "sha256": hashlib.sha256(
                b"Authenticate this task-neutral text input."
            ).hexdigest(),
        }
    ]


def test_future_canonical_content_task_authenticates_declared_role(
    tmp_path: Path,
) -> None:
    payload = _jsonl(
        {
            "example_id": "audio-1",
            "prompt": "Transcribe the audio.",
            "continuation": "hello",
            "content_id": "clip_1",
            "content_sha256": "d" * 64,
            "content_bytes": 4096,
            "content_media_type": "audio/wav",
        }
    )
    source = LocalDatasetRequest(
        path=tmp_path / "records.jsonl",
        sha256=hashlib.sha256(payload).hexdigest(),
        name="audio-evaluation",
        split="validation",
        input_field="prompt",
        expected_output_field="continuation",
        id_field="example_id",
        content_role="audio",
        content_id_field="content_id",
        content_sha256_field="content_sha256",
        content_byte_length_field="content_bytes",
        content_media_type_field="content_media_type",
    )

    schedule = prepare_local_evaluation_schedule_bytes(
        source,
        payload,
        task="audio_text_generation",
    )

    assert schedule.task == "audio_text_generation"
    assert schedule.records[0].input_parts[0].to_payload() == {
        "kind": "content",
        "role": "audio",
        "content_id": "clip_1",
        "media_type": "audio/wav",
        "byte_length": 4096,
        "sha256": "d" * 64,
    }
    assert local_dataset_preparation_payload(source, schedule)["content_role"] == (
        "audio"
    )
    relabeled = prepare_local_evaluation_schedule_bytes(
        dataclasses.replace(source, content_role="signal"),
        payload,
        task="audio_text_generation",
    )
    assert relabeled.schedule_sha256 != schedule.schedule_sha256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("content_id", 7, "must be text"),
        ("content_sha256", "not-a-digest", "lowercase sha256"),
        ("content_bytes", True, "positive integer"),
        ("content_bytes", 0, "positive integer"),
        ("content_media_type", "IMAGE/PNG", "media_type must be canonical"),
    ],
)
def test_vision_content_material_fails_closed_before_schedule_publication(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    record: dict[str, object] = {
        "example_id": "vision-1",
        "prompt": "Describe the image.",
        "continuation": "A square.",
        "content_id": "image_1",
        "content_sha256": "a" * 64,
        "content_bytes": 123,
        "content_media_type": "image/png",
    }
    record[field] = value
    payload = _jsonl(record)
    source = LocalDatasetRequest(
        path=tmp_path / "records.jsonl",
        sha256=hashlib.sha256(payload).hexdigest(),
        name="vision-evaluation",
        split="validation",
        input_field="prompt",
        expected_output_field="continuation",
        id_field="example_id",
        content_role="image",
        content_id_field="content_id",
        content_sha256_field="content_sha256",
        content_byte_length_field="content_bytes",
        content_media_type_field="content_media_type",
    )

    with pytest.raises(ValueError, match=message):
        prepare_local_evaluation_schedule_bytes(
            source,
            payload,
            task="vision_text_generation",
        )


def test_vision_task_requires_complete_content_mapping(tmp_path: Path) -> None:
    payload = _jsonl({"prompt": "Prompt", "continuation": "Answer"})
    source = _source(tmp_path / "records.jsonl", payload, id_field=None)
    with pytest.raises(ValueError, match="requires dataset content field mappings"):
        prepare_local_evaluation_schedule_bytes(
            source, payload, task="vision_text_generation"
        )

    with pytest.raises(ValueError, match="must be provided together"):
        dataclasses.replace(source, content_id_field="content_id")


def test_content_role_and_mappings_are_required_together(tmp_path: Path) -> None:
    payload = _jsonl({"prompt": "Prompt", "continuation": "Answer"})
    source = _source(tmp_path / "records.jsonl", payload, id_field=None)

    with pytest.raises(ValueError, match="content_role.*provided together"):
        dataclasses.replace(source, content_role="image")
    with pytest.raises(ValueError, match="content_role.*provided together"):
        dataclasses.replace(
            source,
            content_id_field="content_id",
            content_sha256_field="content_sha256",
            content_byte_length_field="content_bytes",
            content_media_type_field="content_media_type",
        )
    with pytest.raises(ValueError, match="canonical input role"):
        dataclasses.replace(
            source,
            content_role="Audio/Clip",
            content_id_field="content_id",
            content_sha256_field="content_sha256",
            content_byte_length_field="content_bytes",
            content_media_type_field="content_media_type",
        )


def test_vision_task_requires_image_content_role(tmp_path: Path) -> None:
    payload = _jsonl(
        {
            "prompt": "Describe.",
            "continuation": "Answer",
            "content_id": "content_1",
            "content_sha256": "a" * 64,
            "content_bytes": 1,
            "content_media_type": "image/png",
        }
    )
    source = LocalDatasetRequest(
        path=tmp_path / "records.jsonl",
        sha256=hashlib.sha256(payload).hexdigest(),
        name="vision-evaluation",
        split="validation",
        input_field="prompt",
        expected_output_field="continuation",
        content_role="audio",
        content_id_field="content_id",
        content_sha256_field="content_sha256",
        content_byte_length_field="content_bytes",
        content_media_type_field="content_media_type",
    )

    with pytest.raises(ValueError, match="requires content_role 'image'"):
        prepare_local_evaluation_schedule_bytes(
            source,
            payload,
            task="vision_text_generation",
        )


def test_vision_preparation_descriptor_rejects_role_drift(tmp_path: Path) -> None:
    payload = _jsonl(
        {
            "prompt": "Describe.",
            "continuation": "Answer",
            "content_id": "content_1",
            "content_sha256": "a" * 64,
            "content_bytes": 1,
            "content_media_type": "image/png",
        }
    )
    source = LocalDatasetRequest(
        path=tmp_path / "records.jsonl",
        sha256=hashlib.sha256(payload).hexdigest(),
        name="vision-evaluation",
        split="validation",
        input_field="prompt",
        expected_output_field="continuation",
        content_role="image",
        content_id_field="content_id",
        content_sha256_field="content_sha256",
        content_byte_length_field="content_bytes",
        content_media_type_field="content_media_type",
    )
    schedule = prepare_local_evaluation_schedule_bytes(
        source,
        payload,
        task="vision_text_generation",
    )
    descriptor = local_dataset_preparation_payload(source, schedule)
    descriptor["content_role"] = "audio"

    with pytest.raises(ValueError, match="requires content_role 'image'"):
        validate_local_dataset_preparation(descriptor, schedule)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {"content_id_field": "content_id"},
            "content field mappings must be provided together",
        ),
        (
            {
                "content_id_field": "content_id",
                "content_sha256_field": "content_sha256",
                "content_byte_length_field": "content_bytes",
                "content_media_type_field": "content_media_type",
            },
            "content_role.*provided together",
        ),
        (
            {
                "content_role": "IMAGE",
                "content_id_field": "content_id",
                "content_sha256_field": "content_sha256",
                "content_byte_length_field": "content_bytes",
                "content_media_type_field": "content_media_type",
            },
            "content_role is invalid",
        ),
        ({"input_field": "nested.prompt"}, "input_field is invalid"),
    ],
)
def test_preparation_descriptor_rejects_untrusted_content_contract_shape(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    payload = _jsonl({"prompt": "Prompt", "continuation": "Answer"})
    source = _source(tmp_path / "records.jsonl", payload, id_field=None)
    schedule = prepare_local_evaluation_schedule_bytes(source, payload)
    descriptor = local_dataset_preparation_payload(source, schedule)
    descriptor.update(mutation)

    with pytest.raises(ValueError, match=message):
        validate_local_dataset_preparation(descriptor, schedule)


def test_preparation_descriptor_rejects_content_mapping_without_content_parts(
    tmp_path: Path,
) -> None:
    payload = _jsonl({"prompt": "Prompt", "continuation": "Answer"})
    source = _source(tmp_path / "records.jsonl", payload, id_field=None)
    schedule = prepare_local_evaluation_schedule_bytes(source, payload)
    descriptor = local_dataset_preparation_payload(source, schedule)
    descriptor.update(
        {
            "content_role": "image",
            "content_id_field": "content_id",
            "content_sha256_field": "content_sha256",
            "content_byte_length_field": "content_bytes",
            "content_media_type_field": "content_media_type",
        }
    )

    with pytest.raises(ValueError, match="content mappings do not match"):
        validate_local_dataset_preparation(descriptor, schedule)


def test_preparation_descriptor_enforces_task_specific_content_semantics(
    tmp_path: Path,
) -> None:
    text_payload = _jsonl({"prompt": "Prompt", "continuation": "Answer"})
    text_source = _source(tmp_path / "text.jsonl", text_payload, id_field=None)
    text_schedule = prepare_local_evaluation_schedule_bytes(text_source, text_payload)
    vision_payload = text_schedule.to_payload()
    vision_payload["task"] = "vision_text_generation"
    vision_without_content = build_runtime_behavioral_schedule(vision_payload)
    text_descriptor = local_dataset_preparation_payload(text_source, text_schedule)

    with pytest.raises(ValueError, match="requires run dataset content"):
        validate_local_dataset_preparation(text_descriptor, vision_without_content)

    content_payload = _jsonl(
        {
            "prompt": "Transcribe.",
            "continuation": "hello",
            "content_id": "clip_1",
            "content_sha256": "d" * 64,
            "content_bytes": 256,
            "content_media_type": "audio/wav",
        }
    )
    content_source = LocalDatasetRequest(
        path=tmp_path / "content.jsonl",
        sha256=hashlib.sha256(content_payload).hexdigest(),
        name="content-evaluation",
        split="validation",
        input_field="prompt",
        expected_output_field="continuation",
        content_role="audio",
        content_id_field="content_id",
        content_sha256_field="content_sha256",
        content_byte_length_field="content_bytes",
        content_media_type_field="content_media_type",
    )
    future_schedule = prepare_local_evaluation_schedule_bytes(
        content_source,
        content_payload,
        task="audio_text_generation",
    )
    text_with_content_payload = future_schedule.to_payload()
    text_with_content_payload["task"] = "text_causal"
    text_with_content = build_runtime_behavioral_schedule(text_with_content_payload)
    content_descriptor = local_dataset_preparation_payload(
        content_source,
        future_schedule,
    )

    with pytest.raises(ValueError, match="text_causal does not accept"):
        validate_local_dataset_preparation(content_descriptor, text_with_content)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"extra": True}, "fields are invalid"),
        ({"format_version": "wrong"}, "format_version is invalid"),
        ({"source_format": "csv"}, "source_format is invalid"),
        ({"source_sha256": "bad"}, "source_sha256 is invalid"),
        ({"id_field": "prompt"}, "field mappings must be distinct"),
        ({"name": "other"}, "does not match the canonical schedule identity"),
        ({"selected_record_count": True}, "selected_record_count"),
        ({"limit": 1}, "limit does not match"),
    ],
)
def test_preparation_descriptor_rejects_drift_from_authenticated_schedule(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    payload = _jsonl(
        {"example_id": "one", "prompt": "One", "continuation": "A"},
        {"example_id": "two", "prompt": "Two", "continuation": "B"},
    )
    source = _source(tmp_path / "records.jsonl", payload, limit=2)
    schedule = prepare_local_evaluation_schedule_bytes(source, payload)
    descriptor = local_dataset_preparation_payload(source, schedule)
    descriptor.update(mutation)

    with pytest.raises(ValueError, match=message):
        validate_local_dataset_preparation(descriptor, schedule)


def test_preparation_binding_rejects_missing_mode_specific_intent(
    tmp_path: Path,
) -> None:
    payload = _jsonl({"prompt": "One", "continuation": "A"})
    source = _source(tmp_path / "records.jsonl", payload, id_field=None)
    schedule = prepare_local_evaluation_schedule_bytes(source, payload)
    descriptor = local_dataset_preparation_payload(source, schedule)

    assert "binding is invalid" in dataset_preparation_binding_errors({}, schedule)[0]
    assert (
        dataset_preparation_binding_errors(
            {
                "comparison": {"dataset": "schedule/runtime-behavioral-schedule.json"},
                "execution": {"mode": "import"},
            },
            schedule,
        )
        == []
    )
    assert (
        "canonical schedule"
        in dataset_preparation_binding_errors(
            {
                "comparison": {"dataset": "other.json"},
                "execution": {"mode": "import"},
            },
            schedule,
        )[0]
    )
    assert (
        "path-free preparation descriptor"
        in dataset_preparation_binding_errors(
            {
                "comparison": {"dataset": "wrong"},
                "execution": {"mode": "run"},
            },
            schedule,
        )[0]
    )

    drifted = dict(descriptor)
    drifted["selected_record_count"] = 2
    assert (
        "selected_record_count"
        in dataset_preparation_binding_errors(
            {
                "comparison": {"dataset": drifted},
                "execution": {"mode": "run"},
            },
            schedule,
        )[0]
    )


def test_dataset_digest_mismatch_fails_before_parsing(tmp_path: Path) -> None:
    payload = _jsonl({"example_id": "one", "prompt": "One", "continuation": "A"})
    source = _source(tmp_path / "records.jsonl", payload, sha256="0" * 64)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        prepare_local_evaluation_schedule_bytes(source, payload)


def test_requested_limit_cannot_silently_shrink(tmp_path: Path) -> None:
    payload = _jsonl({"example_id": "one", "prompt": "One", "continuation": "A"})

    with pytest.raises(ValueError, match="fewer records"):
        prepare_local_evaluation_schedule_bytes(
            _source(tmp_path / "records.jsonl", payload, limit=2), payload
        )


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([{"example_id": "one", "prompt": 3, "continuation": "A"}], "must be text"),
        ([{"example_id": "one", "prompt": "One"}], "must be text"),
        (
            [
                [],
            ],
            "must be an object",
        ),
    ],
)
def test_invalid_selected_record_fails_closed(
    tmp_path: Path, records: list[object], message: str
) -> None:
    payload = b"".join(
        json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"
        for record in records
    )

    with pytest.raises(ValueError, match=message):
        prepare_local_evaluation_schedule_bytes(
            _source(tmp_path / "records.jsonl", payload), payload
        )


def test_duplicate_mapped_ids_fail_closed(tmp_path: Path) -> None:
    payload = _jsonl(
        {"example_id": "same", "prompt": "One", "continuation": "A"},
        {"example_id": "same", "prompt": "Two", "continuation": "B"},
    )

    with pytest.raises(ValueError, match="record IDs must be unique"):
        prepare_local_evaluation_schedule_bytes(
            _source(tmp_path / "records.jsonl", payload), payload
        )


def test_field_mappings_must_be_distinct_and_top_level(tmp_path: Path) -> None:
    payload = _jsonl({"value": "One"})

    with pytest.raises(ValueError, match="distinct fields"):
        _source(
            tmp_path / "records.jsonl",
            payload,
            input_field="value",
            expected_output_field="value",
            id_field=None,
        )
    with pytest.raises(ValueError, match="top-level JSON field"):
        _source(
            tmp_path / "records.jsonl",
            payload,
            input_field="nested.prompt",
            expected_output_field="answer",
            id_field=None,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"format": "csv"}, "dataset format"),
        ({"sha256": "A" * 64}, "lowercase SHA-256"),
        ({"limit": 0}, "dataset limit"),
    ],
)
def test_local_dataset_request_rejects_invalid_source_contract_fields(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    payload = _jsonl({"example_id": "one", "prompt": "One", "continuation": "A"})

    with pytest.raises(ValueError, match=message):
        _source(tmp_path / "records.jsonl", payload, **override)


def test_schedule_preparation_requires_bytes_at_the_owned_boundary(
    tmp_path: Path,
) -> None:
    payload = b"{}\n"

    with pytest.raises(TypeError, match="payload must be bytes"):
        prepare_local_evaluation_schedule_bytes(
            _source(tmp_path / "records.jsonl", payload),
            "{}\n",  # type: ignore[arg-type]
        )


def test_schedule_preparation_rejects_empty_authenticated_dataset(
    tmp_path: Path,
) -> None:
    payload = b""

    with pytest.raises(ValueError, match="at least one record"):
        prepare_local_evaluation_schedule_bytes(
            _source(tmp_path / "records.jsonl", payload), payload
        )
