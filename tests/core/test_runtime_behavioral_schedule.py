from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from invarlock.core.runtime_provider.behavioral_schedule import (
    MAX_RUNTIME_BEHAVIORAL_RECORD_ID_CHARACTERS,
    MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES,
    MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS,
    MAX_RUNTIME_BEHAVIORAL_TEXT_CHARACTERS,
    RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT,
    build_runtime_behavioral_schedule,
    canonical_runtime_behavioral_schedule_json,
    load_runtime_behavioral_schedule,
    parse_runtime_behavioral_schedule_json,
)
from invarlock.evidence_pack_json import StrictJsonError


def _record(record_id: str, input_text: str, expected_output: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "input_text": input_text,
        "input_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
        "expected_output": expected_output,
    }


def _payload() -> dict[str, Any]:
    return {
        "format_version": RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT,
        "dataset_identity": {
            "provider": "hf_text",
            "dataset_name": "allenai/ai2_arc",
            "config_name": "ARC-Challenge",
            "revision": "a" * 40,
            "split": "validation",
        },
        "records": [
            _record("arc/1", "Question one?", "A"),
            _record("arc/2", "Question two?", "B"),
        ],
    }


def test_build_runtime_behavioral_schedule_recomputes_canonical_digest() -> None:
    payload = _payload()

    schedule = build_runtime_behavioral_schedule(payload)
    expected_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    assert schedule.to_payload() == payload
    assert canonical_runtime_behavioral_schedule_json(schedule) == expected_json
    assert schedule.schedule_sha256 == hashlib.sha256(expected_json).hexdigest()
    assert schedule.evaluation_batch().schedule_sha256 == schedule.schedule_sha256
    assert [record.record_id for record in schedule.records] == ["arc/1", "arc/2"]


def test_schedule_digest_binds_order_and_exact_record_material() -> None:
    original = _payload()
    reordered = copy.deepcopy(original)
    reordered["records"].reverse()
    mutated = copy.deepcopy(original)
    mutated_text = "Question one, revised?"
    mutated["records"][0]["input_text"] = mutated_text
    mutated["records"][0]["input_sha256"] = hashlib.sha256(
        mutated_text.encode("utf-8")
    ).hexdigest()

    original_schedule = build_runtime_behavioral_schedule(original)
    reordered_schedule = build_runtime_behavioral_schedule(reordered)
    mutated_schedule = build_runtime_behavioral_schedule(mutated)

    assert reordered_schedule.schedule_sha256 != original_schedule.schedule_sha256
    assert mutated_schedule.schedule_sha256 != original_schedule.schedule_sha256
    assert [record.record_id for record in reordered_schedule.records] == [
        "arc/2",
        "arc/1",
    ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("input_text", "tampered prompt"),
        ("input_sha256", "0" * 64),
    ],
)
def test_schedule_rejects_input_or_hash_tampering(field: str, replacement: str) -> None:
    payload = _payload()
    payload["records"][0][field] = replacement

    with pytest.raises(ValueError, match="does not match input_text"):
        build_runtime_behavioral_schedule(payload)


def test_schedule_rejects_duplicate_record_ids() -> None:
    payload = _payload()
    payload["records"][1]["record_id"] = "arc/1"

    with pytest.raises(ValueError, match="record IDs must be unique"):
        build_runtime_behavioral_schedule(payload)


@pytest.mark.parametrize("location", ["root", "dataset", "record"])
def test_schedule_rejects_unknown_fields(location: str) -> None:
    payload = _payload()
    target = {
        "root": payload,
        "dataset": payload["dataset_identity"],
        "record": payload["records"][0],
    }[location]
    target["private_path"] = "/Users/example/model"

    with pytest.raises(ValueError, match="unknown private_path"):
        build_runtime_behavioral_schedule(payload)


@pytest.mark.parametrize("field", ["input_text", "expected_output"])
@pytest.mark.parametrize("value", ["", " \n\t"])
def test_schedule_rejects_empty_or_whitespace_only_claim_material(
    field: str, value: str
) -> None:
    payload = _payload()
    payload["records"][0][field] = value
    if field == "input_text":
        payload["records"][0]["input_sha256"] = hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()

    with pytest.raises(ValueError, match="must be non-empty text"):
        build_runtime_behavioral_schedule(payload)


def test_schedule_enforces_record_and_text_bounds() -> None:
    too_many = _payload()
    too_many["records"] = [too_many["records"][0]] * (
        MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS + 1
    )
    with pytest.raises(ValueError, match="records must not exceed"):
        build_runtime_behavioral_schedule(too_many)

    long_text = "x" * (MAX_RUNTIME_BEHAVIORAL_TEXT_CHARACTERS + 1)
    oversized_text = _payload()
    oversized_text["records"][0] = _record("arc/1", long_text, "A")
    with pytest.raises(ValueError, match="input_text must not exceed"):
        build_runtime_behavioral_schedule(oversized_text)

    oversized_id = _payload()
    oversized_id["records"][0]["record_id"] = "r" * (
        MAX_RUNTIME_BEHAVIORAL_RECORD_ID_CHARACTERS + 1
    )
    with pytest.raises(ValueError, match="record_id must not exceed"):
        build_runtime_behavioral_schedule(oversized_id)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("dataset", "dataset_name", "/private/dataset"),
        ("dataset", "dataset_name", "../dataset"),
        ("dataset", "dataset_name", "C:\\private\\dataset"),
        ("dataset", "split", "validation/../private"),
        ("record", "record_id", "/tmp/record"),
        ("record", "record_id", "../record"),
        ("record", "record_id", "C:\\private\\record"),
    ],
)
def test_schedule_rejects_path_bearing_identity_fields(
    section: str, field: str, value: str
) -> None:
    payload = _payload()
    target = (
        payload["dataset_identity"] if section == "dataset" else payload["records"][0]
    )
    target[field] = value

    with pytest.raises(ValueError, match="absolute or traversal path"):
        build_runtime_behavioral_schedule(payload)


def test_schedule_preserves_arbitrary_nonempty_model_text() -> None:
    payload = _payload()
    prompt = "Explain why C:\\data\\example.txt is unsafe. 🧪"
    expected = "Do not publish /private/paths."
    payload["records"][0] = _record("arc/1", prompt, expected)

    schedule = build_runtime_behavioral_schedule(payload)

    assert schedule.records[0].input_text == prompt
    assert schedule.records[0].expected_output == expected


def test_hosted_dataset_identity_requires_immutable_complete_coordinates() -> None:
    mutable = _payload()
    mutable["dataset_identity"]["revision"] = "main"
    with pytest.raises(ValueError, match="immutable lowercase revision"):
        build_runtime_behavioral_schedule(mutable)

    incomplete = _payload()
    incomplete["dataset_identity"]["config_name"] = None
    with pytest.raises(ValueError, match="hosted dataset_identity requires"):
        build_runtime_behavioral_schedule(incomplete)


def test_strict_json_parser_rejects_duplicate_keys_and_nonfinite_values() -> None:
    valid = json.dumps(_payload(), separators=(",", ":"))
    duplicate = valid.replace(
        '"format_version":',
        '"format_version":"invarlock/runtime-behavioral-schedule-v1","format_version":',
        1,
    )
    with pytest.raises(StrictJsonError, match="duplicate key"):
        parse_runtime_behavioral_schedule_json(duplicate)

    nonfinite = valid.replace(
        '"format_version":"invarlock/runtime-behavioral-schedule-v1"',
        '"format_version":NaN',
        1,
    )
    with pytest.raises(StrictJsonError, match="non-standard constant"):
        parse_runtime_behavioral_schedule_json(nonfinite)


def test_file_loader_uses_bounded_regular_file_snapshot(tmp_path: Path) -> None:
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(json.dumps(_payload()), encoding="utf-8")

    loaded = load_runtime_behavioral_schedule(schedule_path)

    assert loaded.to_payload() == _payload()

    symlink = tmp_path / "schedule-link.json"
    symlink.symlink_to(schedule_path)
    with pytest.raises(StrictJsonError, match="must not be a symlink"):
        load_runtime_behavioral_schedule(symlink)

    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as handle:
        handle.seek(MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES)
        handle.write(b"x")
    with pytest.raises(StrictJsonError, match="size limit"):
        load_runtime_behavioral_schedule(oversized)
