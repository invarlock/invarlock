"""Differential checks against the production Python JSON and contract boundary.

Build contracts_probe first; set CARGO_TARGET_DIR or NATIVE_CONTRACTS_PROBE when
using a shared target directory. This suite never substitutes a Python result
for a native result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import struct
import subprocess
from pathlib import Path

import pytest

from invarlock.evaluation_record_contracts.contracts import validate
from invarlock.evidence_pack_json import parse_json_bytes
from tests.evaluation_records.test_contract_validation import (
    CLOSED_OBJECTS,
    at,
    documents,
)

ROOT = Path(__file__).resolve().parents[2]


def probe(requests):
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "native" / "target"))
    binary = Path(
        os.environ.get("NATIVE_CONTRACTS_PROBE", target / "debug" / "contracts_probe")
    )
    assert binary.is_file(), f"Build contracts_probe before running: {binary}"
    result = subprocess.run(
        [str(binary)],
        input="".join(json.dumps(item, ensure_ascii=True) + "\n" for item in requests),
        text=True,
        capture_output=True,
        check=True,
    )
    return [json.loads(line) for line in result.stdout.rstrip("\n").split("\n")]


def test_canonical_ieee_unicode_integer_and_decimal_bytes():
    rng = random.Random(772)
    values = [
        0.0,
        -0.0,
        1e-4,
        1e-5,
        1e15,
        1e16,
        1e23,
        5e-324,
        1.7976931348623157e308,
        10**100,
        {"é": "𐀀\u2028\n\x7f", "a": 1},
        -0,
    ]
    values.extend(
        v
        for _ in range(100_000)
        if math.isfinite(v := struct.unpack("d", rng.randbytes(8))[0])
    )
    unicode_points = [rng.randrange(0x110000) for _ in range(10_000)]
    values.append("".join(chr(n) for n in unicode_points if not 0xD800 <= n <= 0xDFFF))
    raw = [json.dumps(v, ensure_ascii=False) for v in values]
    raw += ["-0", "1.0000000000000000000000000000000000001", "1e-400", "-1e-400"]
    raw += [
        str(rng.randrange(10**39, 10**40)) + "e" + str(rng.randrange(-360, 260))
        for _ in range(1000)
    ]
    requests = [{"raw": text, "newline": i % 2 == 0} for i, text in enumerate(raw)]
    results = probe(requests)
    assert len(results) == len(requests)
    for request, result in zip(requests, results, strict=True):
        value = parse_json_bytes(request["raw"].encode(), label="probe")
        expected = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if request["newline"]:
            expected += "\n"
        assert result["ok"], (request, result)
        assert result["canonical"] == expected, (request, result, expected)
        assert (
            result["digest"]
            == "sha256:" + hashlib.sha256(expected.encode()).hexdigest()
        )


def test_strict_error_messages_and_explicit_surrogate_boundary():
    raw = [
        "NaN",
        "Infinity",
        "-Infinity",
        "1e400",
        "-1e400",
        "[1,]",
        "01",
        "",
        "\ufeff{}",
        '{"x":1,"x":2}',
        '{"x":1,"x":2,"y":NaN}',
        '{"x":{"a":1,"a":2}}',
        '{"\\u0061":1,"a":2}',
        '"\\x20"',
        "9" * 4301,
    ]
    for key in ["a'b", 'a"b', "\n", "\x00", "\u2028", "\u200b", "\U000e0001", "é"]:
        raw.append("{" + json.dumps(key) + ":1," + json.dumps(key) + ":2}")
    results = probe([{"raw": text} for text in raw])
    for text, result in zip(raw, results, strict=True):
        with pytest.raises(ValueError) as expected:
            parse_json_bytes(text.encode(), label="probe")
        assert result["ok"] is False
        assert result["message"] == str(expected.value)
    result = probe([{"raw": '"\\ud800"'}])[0]
    assert result["code"] == "unpaired_surrogate"
    assert parse_json_bytes(b'"\\ud800"', label="probe") == "\ud800"


def visit(value, path=()):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from visit(child, path + (key,))
    elif isinstance(value, list):
        for key, child in enumerate(value):
            yield from visit(child, path + (key,))


def test_closed_schemas_acceptance_and_exact_diagnostics():
    base = documents.__wrapped__()
    cases = [(name, value) for name, value in base.items()]
    for name, path in CLOSED_OBJECTS:
        value = copy.deepcopy(base[name])
        at(value, path)["unexpected"] = True
        cases.append((name, value))
        for key in at(base[name], path):
            value = copy.deepcopy(base[name])
            del at(value, path)[key]
            cases.append((name, value))
    for name, value in base.items():
        for path, _ in visit(value):
            for bad in [None, True, False, [], {}, "", -1, 1.5, "x\x7f", 10**100]:
                changed = copy.deepcopy(value)
                if path:
                    at(changed, path[:-1])[path[-1]] = bad
                else:
                    changed = bad
                cases.append((name, changed))
    rng = random.Random(91)
    for _ in range(2000):
        name = rng.choice(list(base))
        value = copy.deepcopy(base[name])
        paths = list(visit(value))
        for _ in range(rng.randrange(2, 5)):
            path, _ = rng.choice(paths)
            try:
                if path:
                    at(value, path[:-1])[path[-1]] = rng.choice(
                        [None, True, [], {}, "bad", -1]
                    )
            except (KeyError, TypeError, IndexError):
                pass
        cases.append((name, value))
    requests = [{"raw": json.dumps(value), "schema": name} for name, value in cases]
    results = probe(requests)
    assert len(results) == len(requests)
    for request, result in zip(requests, results, strict=True):
        value = parse_json_bytes(request["raw"].encode(), label="probe")
        try:
            validate(value, request["schema"])
        except ValueError as error:
            assert not result["ok"], (request, result)
            assert result["message"] == str(error), (request, result, str(error))
        else:
            assert result["ok"], (request, result)


def test_packaged_schema_bytes_and_identity_manifest():
    source = ROOT / "src" / "invarlock" / "_data" / "contracts"
    packaged = ROOT / "native" / "contracts" / "schemas"
    identities = json.loads((packaged / "source-digests.json").read_text())
    assert set(identities) == {p.name for p in source.glob("*.json")}
    for name, expected in identities.items():
        raw = (packaged / name).read_bytes()
        assert raw == (source / name).read_bytes()
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == expected


def test_schema_arbitrary_integers_and_exact_float_comparisons():
    from jsonschema import Draft202012Validator

    integers = [
        0,
        1,
        -1,
        2**53 - 1,
        2**53 + 1,
        2**63 + 1,
        2**64 + 1,
        10**100,
        -(10**100),
        10**400,
        -(10**400),
    ]
    floats = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        float(2**53),
        float(2**64),
        1e100,
        -1e100,
        1.7976931348623157e308,
    ]
    pairs = [
        (value, limit) for value in integers + floats for limit in integers + floats
    ]
    rng = random.Random(873)
    for _ in range(60):
        value = struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0]
        if math.isfinite(value):
            pairs.extend(
                [
                    (int(value) + 1, value),
                    (int(value) - 1, value),
                    (value, int(value) + 1),
                ]
            )
    cases = []
    for value, limit in pairs:
        for keyword in (
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "const",
            "enum",
        ):
            cases.append((value, {keyword: [limit] if keyword == "enum" else limit}))
    for value in integers + floats + [None, True, False, "1"]:
        cases.append((value, {"type": "integer"}))
        cases.append(([value, value], {"uniqueItems": True}))
        cases.append(([value, True], {"uniqueItems": True}))
    for value, limit in pairs[::7]:
        cases.append(([{"number": value}, {"number": limit}], {"uniqueItems": True}))
    requests = [
        {"raw": json.dumps(value), "schema_value": schema} for value, schema in cases
    ]
    actual = probe(requests)
    for (value, schema), result in zip(cases, actual, strict=True):
        error = next(Draft202012Validator(schema).iter_errors(value), None)
        assert result["ok"] == (error is None), (value, schema, result, str(error))
        if error is not None:
            assert result["message"] == error.message, (
                value,
                schema,
                result,
                error.message,
            )


@pytest.mark.parametrize(
    "shared,middle,bad_middle,bad_omega",
    [
        ({"type": "integer"}, {"type": "integer"}, "bad-middle", "bad-omega"),
        ({"minimum": 0}, {"minimum": 0}, -1, -2),
        ({"maximum": 0}, {"maximum": 0}, 1, 2),
        ({"exclusiveMinimum": 0}, {"exclusiveMinimum": 0}, -1, -2),
        ({"exclusiveMaximum": 0}, {"exclusiveMaximum": 0}, 1, 2),
        ({"const": 0}, {"const": 1}, "bad-middle", "bad-omega"),
        ({"enum": [0]}, {"enum": [1]}, "bad-middle", "bad-omega"),
        ({"uniqueItems": True}, {"uniqueItems": True}, [1, 1], [2, 2]),
        ({"pattern": "^valid$"}, {"pattern": "^valid$"}, "bad-middle", "bad-omega"),
    ],
    ids=[
        "type",
        "minimum",
        "maximum",
        "exclusive-minimum",
        "exclusive-maximum",
        "const",
        "enum",
        "unique-items",
        "pattern",
    ],
)
@pytest.mark.parametrize("reverse_instance", [False, True])
@pytest.mark.parametrize("container", ["object", "nested", "array"])
def test_shared_reference_diagnostics_follow_schema_use_site(
    shared, middle, bad_middle, bad_omega, reverse_instance, container
):
    """An absent earlier reference must not own a later use site's error order."""
    from jsonschema import Draft202012Validator

    node = {
        "type": "object",
        "properties": {
            "alpha": {"$ref": "#/$defs/shared~1value~0"},
            "middle": middle,
            "omega": {"$ref": "#/$defs/shared~1value~0"},
        },
    }
    entries = [("middle", bad_middle), ("omega", bad_omega)]
    if reverse_instance:
        entries.reverse()
    value = dict(entries)
    if container == "nested":
        node = {"type": "object", "properties": {"container": node}}
        value = {"container": value}
    elif container == "array":
        node = {"type": "array", "items": node}
        value = [value]
    schema = {"$defs": {"shared/value~": shared}, **node}
    expected = next(Draft202012Validator(schema).iter_errors(value))
    assert list(expected.path)[-1] == "middle"
    result = probe([{"raw": json.dumps(value), "schema_value": schema}])[0]
    assert result == {
        "ok": False,
        "code": "schema_validation",
        "message": expected.message,
    }
