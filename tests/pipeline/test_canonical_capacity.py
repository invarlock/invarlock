"""Bound canonical byte checks without allocating a complete second artifact."""

import hashlib

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
def test_incremental_canonical_digest_preserves_existing_wire_bytes(value, monkeypatch):
    expected = "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()

    def forbidden(*args, **kwargs):
        pytest.fail("pipeline digest must not allocate the complete canonical artifact")

    monkeypatch.setattr(contracts.json, "dumps", forbidden)
    assert contracts.digest(value) == expected


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
