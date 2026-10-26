"""Typed behavioral schedule and pinned local JSONL contract differential."""

import copy
import hashlib
import json
from pathlib import Path

from invarlock.core.runtime_provider.behavioral_schedule import (
    build_runtime_behavioral_schedule,
    build_runtime_behavioral_schedule_from_material,
    canonical_runtime_behavioral_schedule_json,
    parse_runtime_behavioral_schedule_json,
)
from invarlock.core.schedule_preparation import (
    LocalDatasetRequest,
    local_dataset_preparation_payload,
    prepare_local_evaluation_schedule_bytes,
    validate_local_dataset_preparation,
)
from native.tests.test_contracts_differential import probe
from tests.runtime.test_runtime_import_authoring import _schedule


def reference(request):
    try:
        action = request.get("action", "build")
        task = request.get("task", "text_causal")
        if action == "prepare":
            source = LocalDatasetRequest(path=Path("unused"), **request["source"])
            result = prepare_local_evaluation_schedule_bytes(
                source, request["raw"].encode(), task=task
            )
        elif action == "material":
            result = build_runtime_behavioral_schedule_from_material(
                dataset_identity=request["identity"],
                records=request["records"],
                task=task,
            )
        elif action == "parse":
            result = parse_runtime_behavioral_schedule_json(request["raw"])
        else:
            result = build_runtime_behavioral_schedule(request["value"])
        if "descriptor" in request:
            validate_local_dataset_preparation(request["descriptor"], result)
        out = {
            "ok": True,
            "value": result.to_payload(),
            "canonical": canonical_runtime_behavioral_schedule_json(result).decode(),
            "digest": result.schedule_sha256,
        }
        if request.get("make_descriptor"):
            if action != "prepare":
                source = LocalDatasetRequest(path=Path("unused"), **request["source"])
            out["descriptor"] = local_dataset_preparation_payload(source, result)
        return out
    except (ValueError, TypeError) as error:
        return {"ok": False, "message": str(error)}


def compare(cases):
    expected = [reference(case) for case in cases]
    actual = probe([{"op": "schedule", **case} for case in cases])
    for index, (case, want, got) in enumerate(
        zip(cases, expected, actual, strict=True)
    ):
        got.pop("code", None)
        assert got == want, (index, case, want, got)


def test_typed_schedule_shapes_values_unicode_and_bounds():
    baseline = _schedule().to_payload()
    paths = [
        [],
        ["format_version"],
        ["task"],
        ["dataset_identity"],
        ["records"],
        ["dataset_identity", "provider"],
        ["dataset_identity", "dataset_name"],
        ["dataset_identity", "config_name"],
        ["dataset_identity", "revision"],
        ["dataset_identity", "split"],
        ["records", 0],
        ["records", 0, "record_id"],
        ["records", 0, "expected_output"],
        ["records", 0, "input_sha256"],
        ["records", 0, "input_parts"],
        ["records", 0, "input_parts", 0],
        ["records", 0, "input_parts", 0, "kind"],
        ["records", 0, "input_parts", 0, "role"],
        ["records", 0, "input_parts", 0, "text"],
        ["records", 0, "input_parts", 0, "sha256"],
    ]
    values = [
        None,
        False,
        True,
        0,
        1,
        1.0,
        {},
        [],
        "",
        " ",
        "\x1c",
        "../escape",
        "A" * 257,
        "é" * 256,
    ]
    cases = [{"value": baseline}]
    for path in paths:
        for replacement in values:
            value = copy.deepcopy(baseline)
            if not path:
                value = replacement
            else:
                target = value
                for part in path[:-1]:
                    target = target[part]
                target[path[-1]] = replacement
            cases.append({"value": value})
        if path:
            value = copy.deepcopy(baseline)
            target = value
            for part in path[:-1]:
                target = target[part]
            if isinstance(target, dict):
                del target[path[-1]]
                cases.append({"value": value})
    content = {
        "kind": "content",
        "role": "image",
        "content_id": "picture",
        "media_type": "image/png",
        "byte_length": 10**400,
        "sha256": "b" * 64,
    }
    material = [
        {
            "record_id": "vision/one",
            "expected_output": "answer",
            "input_parts": [content, baseline["records"][0]["input_parts"][0]],
        }
    ]
    cases.append(
        {
            "action": "material",
            "identity": baseline["dataset_identity"],
            "records": material,
            "task": "vision_text_generation",
        }
    )
    for key in content:
        for replacement in values:
            records = copy.deepcopy(material)
            records[0]["input_parts"][0][key] = replacement
            cases.append(
                {
                    "action": "material",
                    "identity": baseline["dataset_identity"],
                    "records": records,
                    "task": "vision_text_generation",
                }
            )
    duplicate = copy.deepcopy(baseline)
    duplicate["records"][0]["input_parts"] *= 2
    duplicate["records"][0]["input_sha256"] = hashlib.sha256(
        json.dumps(
            duplicate["records"][0]["input_parts"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    cases.append({"value": duplicate})
    for count in [0, 64, 65]:
        value = copy.deepcopy(baseline)
        value["records"][0]["input_parts"] *= count
        cases.append({"value": value})
    value = copy.deepcopy(baseline)
    value["records"] *= 5001
    cases.append({"value": value})
    value = copy.deepcopy(baseline)
    value["records"][0]["expected_output"] = "a" * 65537
    cases.append({"value": value})
    for raw in [
        json.dumps(baseline, indent=2),
        json.dumps(baseline) + "\n",
        '{"task":0,"task":1}',
        "[]",
        '{"format_version":NaN}',
    ]:
        cases.append({"action": "parse", "raw": raw})
    compare(cases)


def test_local_jsonl_preparation_and_descriptor():
    raw = '{"prompt":"héllo","answer":"A","id":"one"}\n{"prompt":"next","answer":"B","id":"two"}\n'

    def source(payload):
        return {
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "name": "local-fixture",
            "split": "validation",
            "input_field": "prompt",
            "expected_output_field": "answer",
        }

    cases = [
        {
            "action": "prepare",
            "source": source(raw),
            "raw": raw,
            "make_descriptor": True,
        }
    ]
    for payload in [
        "",
        "\n",
        " \t\n",
        "null\n",
        '{"prompt":"A","prompt":"B","answer":"A"}\n',
        '{"prompt":false,"answer":"A"}\n',
        raw.replace("\n", "\r\n"),
        raw.rstrip("\n"),
        raw + "malformed ignored after selection",
        raw + '{"prompt":NaN}\n',
    ]:
        cases.append({"action": "prepare", "source": source(payload), "raw": payload})
        limited = source(payload)
        limited["limit"] = 1
        cases.append({"action": "prepare", "source": limited, "raw": payload})
    for key in [
        "format",
        "sha256",
        "name",
        "split",
        "input_field",
        "expected_output_field",
        "id_field",
        "limit",
        "content_role",
        "content_id_field",
    ]:
        for replacement in [
            None,
            False,
            True,
            0,
            -1,
            1,
            1.0,
            1.5,
            10001,
            {},
            [],
            "",
            " ",
            "\x1c",
            "../x",
            "answer",
        ]:
            options = source(raw)
            options[key] = replacement
            cases.append({"action": "prepare", "source": options, "raw": raw})
    initial = reference(cases[0])
    descriptor = initial["descriptor"]
    schedule = initial["value"]
    for key in descriptor:
        for replacement in [None, False, True, 0, 1, 2, 2.0, {}, [], "", "x"]:
            value = copy.deepcopy(descriptor)
            value[key] = replacement
            cases.append({"value": schedule, "descriptor": value})
    vision = (
        json.dumps(
            {
                "prompt": "image question",
                "answer": "yes",
                "media_id": "picture",
                "media_sha": "c" * 64,
                "length": 10**400,
                "mime": "image/png",
            }
        )
        + "\n"
    )
    options = source(vision)
    options.update(
        content_role="image",
        content_id_field="media_id",
        content_sha256_field="media_sha",
        content_byte_length_field="length",
        content_media_type_field="mime",
    )
    cases.append(
        {
            "action": "prepare",
            "source": options,
            "raw": vision,
            "task": "vision_text_generation",
            "make_descriptor": True,
        }
    )
    compare(cases)


def test_nested_unused_material_and_raw_size_bounds():
    nested = "[" * 128 + "0" + "]" * 128
    raw = '{"prompt":"valid input","answer":"yes","unused":' + nested + "}\n"
    source = {
        "sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "name": "nested-fixture",
        "split": "validation",
        "input_field": "prompt",
        "expected_output_field": "answer",
    }
    compare([{"action": "prepare", "source": source, "raw": raw}])
    oversized = " " * (16 * 1024 * 1024 + 1)
    compare([{"action": "parse", "raw": oversized}])
    source["sha256"] = hashlib.sha256(oversized.encode()).hexdigest()
    compare([{"action": "prepare", "source": source, "raw": oversized}])
