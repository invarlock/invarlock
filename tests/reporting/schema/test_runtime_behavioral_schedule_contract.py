from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from invarlock import public_contracts
from invarlock.core.runtime_provider.behavioral_schedule import (
    MAX_RUNTIME_BEHAVIORAL_RECORD_ID_CHARACTERS,
    MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS,
    MAX_RUNTIME_BEHAVIORAL_TEXT_CHARACTERS,
    RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT,
    build_runtime_behavioral_schedule,
)


def _record(
    record_id: str = "sample/1", input_text: str = "Question?"
) -> dict[str, str]:
    return {
        "record_id": record_id,
        "input_text": input_text,
        "input_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
        "expected_output": "A",
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
        "records": [_record()],
    }


def _validate(payload: dict[str, Any]) -> None:
    jsonschema.validate(
        instance=payload,
        schema=public_contracts.load_runtime_behavioral_schedule_schema(),
    )


def test_runtime_behavioral_schedule_schema_is_closed_and_canonical() -> None:
    schema = public_contracts.load_runtime_behavioral_schedule_schema()
    jsonschema.Draft202012Validator.check_schema(schema)

    _validate(_payload())

    for location in ("root", "dataset", "record"):
        malformed = copy.deepcopy(_payload())
        target = {
            "root": malformed,
            "dataset": malformed["dataset_identity"],
            "record": malformed["records"][0],
        }[location]
        target["unexpected"] = True
        with pytest.raises(jsonschema.ValidationError):
            _validate(malformed)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("dataset", "dataset_name", "/private/dataset"),
        ("dataset", "dataset_name", "../dataset"),
        ("dataset", "dataset_name", "C:\\private\\dataset"),
        ("dataset", "revision", "main"),
        ("record", "record_id", "/tmp/record"),
        ("record", "record_id", "../record"),
        ("record", "record_id", "C:\\private\\record"),
        ("record", "input_text", ""),
        ("record", "expected_output", " \n\t"),
    ],
)
def test_runtime_behavioral_schedule_schema_rejects_unsafe_or_trivial_material(
    section: str, field: str, value: str
) -> None:
    malformed = copy.deepcopy(_payload())
    target = (
        malformed["dataset_identity"]
        if section == "dataset"
        else malformed["records"][0]
    )
    target[field] = value

    with pytest.raises(jsonschema.ValidationError):
        _validate(malformed)


def test_runtime_behavioral_schedule_schema_enforces_public_bounds() -> None:
    too_many = _payload()
    too_many["records"] = [
        _record(record_id=f"sample/{index}")
        for index in range(MAX_RUNTIME_BEHAVIORAL_SCHEDULE_RECORDS + 1)
    ]
    with pytest.raises(jsonschema.ValidationError):
        _validate(too_many)

    long_text = _payload()
    long_text["records"][0] = _record(
        input_text="x" * (MAX_RUNTIME_BEHAVIORAL_TEXT_CHARACTERS + 1)
    )
    with pytest.raises(jsonschema.ValidationError):
        _validate(long_text)

    long_id = _payload()
    long_id["records"][0]["record_id"] = "r" * (
        MAX_RUNTIME_BEHAVIORAL_RECORD_ID_CHARACTERS + 1
    )
    with pytest.raises(jsonschema.ValidationError):
        _validate(long_id)


def test_schema_shape_is_not_substituted_for_semantic_hash_validation() -> None:
    malformed = _payload()
    malformed["records"][0]["input_sha256"] = "0" * 64

    _validate(malformed)
    with pytest.raises(ValueError, match="does not match input_text"):
        build_runtime_behavioral_schedule(malformed)


def test_hosted_schema_requires_complete_immutable_dataset_identity() -> None:
    for field in ("dataset_name", "config_name", "revision"):
        malformed = _payload()
        malformed["dataset_identity"][field] = None
        with pytest.raises(jsonschema.ValidationError):
            _validate(malformed)


def test_runtime_behavioral_schedule_contract_is_cataloged_and_packaged() -> None:
    catalog = public_contracts.contract_catalog()
    assert catalog["runtime_behavioral_schedule"]["path"] == (
        "contracts/runtime_behavioral_schedule.schema.json"
    )
    assert public_contracts.public_subcontract_catalog()[
        "runtime_behavioral_schedule"
    ] == {
        "version": RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT,
        "source": "contracts/runtime_behavioral_schedule.schema.json",
        "compatibility": "closed_versioned_schedule",
    }
    assert (
        public_contracts.RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT_VERSION
        == RUNTIME_BEHAVIORAL_SCHEDULE_FORMAT
    )

    repository = Path("contracts/runtime_behavioral_schedule.schema.json")
    packaged = public_contracts.PACKAGE_CONTRACTS_ROOT.joinpath(repository.name)
    assert packaged.is_file()
    assert packaged.read_bytes() == repository.read_bytes()
    assert isinstance(json.loads(packaged.read_text(encoding="utf-8")), dict)
