from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from invarlock.core.runtime_provider import behavioral_schedule as schedule_module


def _record(
    record_id: object = "example-1",
    input_text: object = "Return the literal answer.",
    expected_output: object = "answer",
) -> dict[str, object]:
    digest = (
        hashlib.sha256(input_text.encode("utf-8")).hexdigest()
        if isinstance(input_text, str)
        else "a" * 64
    )
    return {
        "record_id": record_id,
        "input_text": input_text,
        "input_sha256": digest,
        "expected_output": expected_output,
    }


def _payload() -> dict[str, Any]:
    return {
        "format_version": schedule_module.RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT,
        "dataset_identity": {
            "provider": "local_manifest",
            "dataset_name": None,
            "config_name": None,
            "revision": None,
            "split": "validation",
        },
        "records": [_record()],
    }


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("root", "records"),
        ("dataset", "split"),
        ("record", "expected_output"),
    ],
)
def test_schedule_rejects_missing_public_fields(location: str, field: str) -> None:
    payload = _payload()
    target = {
        "root": payload,
        "dataset": payload["dataset_identity"],
        "record": payload["records"][0],
    }[location]
    del target[field]

    with pytest.raises(ValueError, match=rf"missing {field}"):
        schedule_module.build_runtime_behavioral_schedule(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provider", " Local_manifest", "non-empty trimmed"),
        ("provider", "local\x00manifest", "control characters"),
        ("provider", "UPPER", "canonical provider name"),
        ("split", ".", "absolute or traversal path"),
        ("split", "nested//split", "absolute or traversal path"),
        ("split", "z" * 129, "must not exceed"),
    ],
)
def test_dataset_identity_rejects_noncanonical_logical_values(
    field: str, value: str, message: str
) -> None:
    payload = _payload()
    payload["dataset_identity"][field] = value

    with pytest.raises(ValueError, match=message):
        schedule_module.build_runtime_behavioral_schedule(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_identity", []),
        ("records", ["not-an-object"]),
    ],
)
def test_schedule_rejects_nonobject_nested_material(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValueError, match="must be an object"):
        schedule_module.build_runtime_behavioral_schedule(payload)


def test_schedule_rejects_oversized_expected_output() -> None:
    payload = _payload()
    payload["records"][0]["expected_output"] = "x" * (
        schedule_module.MAX_RUNTIME_BEHAVIORAL_TEXT_CHARACTERS + 1
    )

    with pytest.raises(ValueError, match="expected_output must not exceed"):
        schedule_module.build_runtime_behavioral_schedule(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "schedule must be an object"),
        (
            {
                **_payload(),
                "format_version": "invarlock/runtime-behavioral-schedule-v0",
            },
            "format_version is unsupported",
        ),
        ({**_payload(), "records": []}, "records must be a non-empty array"),
        ({**_payload(), "records": "record"}, "records must be a non-empty array"),
    ],
)
def test_schedule_rejects_invalid_root_contract(payload: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        schedule_module.build_runtime_behavioral_schedule(payload)  # type: ignore[arg-type]


def test_schedule_rejects_canonical_payload_over_size_limit(monkeypatch) -> None:
    monkeypatch.setattr(schedule_module, "MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES", 1)

    with pytest.raises(ValueError, match="byte size limit"):
        schedule_module.build_runtime_behavioral_schedule(_payload())


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (["not-an-object"], r"records\[0\] must be an object"),
        (
            [
                {
                    "record_id": "example-1",
                    "input_text": 7,
                    "expected_output": "answer",
                }
            ],
            "input_text must be text",
        ),
    ],
)
def test_material_builder_rejects_unhashable_input_contract(
    records: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        schedule_module.build_runtime_behavioral_schedule_from_material(
            dataset_identity=_payload()["dataset_identity"],
            records=records,
        )


def test_text_parser_rejects_nontext_oversize_and_nonobject(monkeypatch) -> None:
    with pytest.raises(ValueError, match="JSON must be text"):
        schedule_module.parse_runtime_behavioral_schedule_json(7)  # type: ignore[arg-type]

    monkeypatch.setattr(schedule_module, "MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES", 1)
    with pytest.raises(ValueError, match="byte size limit"):
        schedule_module.parse_runtime_behavioral_schedule_json("{}")

    monkeypatch.setattr(
        schedule_module,
        "MAX_RUNTIME_BEHAVIORAL_SCHEDULE_BYTES",
        16 * 1024 * 1024,
    )
    with pytest.raises(ValueError, match="schedule must be an object"):
        schedule_module.parse_runtime_behavioral_schedule_json("[]")


def test_file_loader_rejects_nonobject_json(tmp_path: Path) -> None:
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="schedule must be an object"):
        schedule_module.load_runtime_behavioral_schedule(path)
