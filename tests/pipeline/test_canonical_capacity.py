"""Bound canonical byte checks without allocating a complete second artifact."""

import hashlib
import sys

import pytest

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.pipeline import contracts


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        {"z": "é😀\n", "a": [-0.0, 1e-12, True, None]},
        {"nested": [{"a": "x" * 2000} for _ in range(10)]},
    ],
)
def test_incremental_canonical_digest_preserves_existing_wire_bytes(value):
    expected = "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    assert contracts.digest(value) == expected


def test_large_containers_only_use_bounded_composite_buffers(monkeypatch):
    value = {"records": [{"id": i, "text": "é\n🧪" * 30} for i in range(4000)]}
    expected = canonical_json_bytes(value)
    original = contracts.json.JSONEncoder.encode
    composite_sizes = []

    def measured(encoder, part):
        encoded = original(encoder, part)
        if type(part) in (dict, list):
            assert part is not value and part is not value["records"]
            size = len(encoded.encode("utf-8"))
            assert size <= contracts._CANONICAL_BUFFER_BYTES
            composite_sizes.append(size)
        return encoded

    monkeypatch.setattr(contracts.json.JSONEncoder, "encode", measured)
    assert b"".join(contracts._canonical_chunks(value)) == expected
    assert len(composite_sizes) > 1


@pytest.mark.parametrize(
    "value",
    [
        {"controls": '\x00\b\n\r\t\\"', "unicode": "é🧪 e\u0301"},
        [True, False, None, 0, -(10**1000), sys.float_info.max, 5e-324],
        ("tuple", {"nested": [1, 2]}),
        {False: "false", True: "true", 2: "two"},
        {None: "null"},
        {"large-leaf": "x" * 70000, "z": []},
        {"x" * 70000: [1, 2]},
        {"empty": {}, "list": []},
    ],
)
def test_subtree_encoding_preserves_canonical_edge_cases(value, monkeypatch):
    expected = canonical_json_bytes(value)
    # Exercise streaming even for small fixtures, without changing public limits.
    monkeypatch.setattr(contracts, "_CANONICAL_BUFFER_BYTES", 64)
    assert b"".join(contracts._canonical_chunks(value)) == expected


def test_shared_containers_are_not_mistaken_for_cycles():
    shared = {"a": [1, 2, 3]}
    value = [shared, shared]
    assert b"".join(contracts._canonical_chunks(value)) == canonical_json_bytes(value)


@pytest.mark.parametrize("shape", ["dict", "list", "tuple"])
def test_circular_values_are_rejected(shape):
    value = {} if shape == "dict" else []
    if shape == "dict":
        value["self"] = value
    elif shape == "tuple":
        value.append((value,))
    else:
        value.append(value)
    with pytest.raises(contracts.PipelineError, match="Circular reference"):
        contracts.digest(value)


@pytest.mark.parametrize(
    "value", [{1: "integer", "a": "string"}, "\ud800", {"a": float("nan")}]
)
def test_invalid_values_retain_standard_encoder_rejection(value):
    with pytest.raises((ValueError, TypeError)):
        canonical_json_bytes(value)
    with pytest.raises(contracts.PipelineError):
        contracts.digest(value)


def test_deep_streaming_does_not_add_recursive_python_traversal(monkeypatch):
    depth = sys.getrecursionlimit() + 10
    value = 0
    for _ in range(depth):
        value = {"a": value, "z": "x" * 32}
    expected = b'{"a":' * depth + b"0" + (b',"z":"' + b"x" * 32 + b'"}') * depth
    monkeypatch.setattr(contracts, "_CANONICAL_BUFFER_BYTES", 64)
    assert b"".join(contracts._canonical_chunks(value)) == expected + b"\n"


def test_size_limit_stops_before_encoding_later_values(monkeypatch):
    # A late unsupported value is reached only if the encoder keeps traversing
    # after the earlier field has already exceeded the input allowance.
    monkeypatch.setattr(contracts, "MAX_INPUT_BYTES", 16)
    with pytest.raises(contracts.PipelineError, match="exceeds the 16 byte limit"):
        contracts.validate({"a": "x" * 32, "z": object()}, "run")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"a": object()}])
def test_incremental_digest_rejects_non_json_values(value):
    with pytest.raises(ValueError):
        contracts.digest(value)
