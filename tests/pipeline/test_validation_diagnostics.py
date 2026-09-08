"""Reject excessive schedules before their contents become error diagnostics."""

from collections import deque

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from invarlock.pipeline import contracts
from invarlock.pipeline.validation_limits import (
    check_record_counts,
    format_validation_error,
)


@pytest.mark.parametrize(
    "name,value,path",
    [
        ("run", {"records": [{}, {}, {}]}, "records"),
        ("case_set", {"cases": [{}, {}, {}]}, "cases"),
        ("evidence", {"baseline": {"records": [{}, {}, {}]}}, "baseline.records"),
        ("evidence", {"candidate": {"records": [{}, {}, {}]}}, "candidate.records"),
    ],
)
def test_overcount_rejected_before_encoding_or_schema(monkeypatch, name, value, path):
    monkeypatch.setattr(contracts, "MAX_RECORDS", 2)

    def forbidden(*args, **kwargs):
        pytest.fail("overcount must reject before encoding or schema construction")

    monkeypatch.setattr(contracts, "_canonical_chunks", forbidden)
    monkeypatch.setattr(contracts, "_validator", forbidden)
    with pytest.raises(contracts.PipelineError, match=rf"{path}.*too long") as caught:
        contracts.validate(value, name)
    assert len(str(caught.value)) < 160


@pytest.mark.parametrize("limit", [0, 2, 7])
def test_count_preflight_uses_supplied_limit_without_visiting_rows(limit):
    class Uninspectable:
        def __repr__(self):
            pytest.fail("count rejection must not inspect or print record bodies")

    rows = [Uninspectable()] * limit
    check_record_counts({"records": rows}, "run", max_records=limit)
    rows.append(Uninspectable())
    with pytest.raises(ValueError, match=rf"records.*too long.*maximum {limit}"):
        check_record_counts({"records": rows}, "run", max_records=limit)


@pytest.mark.parametrize(
    "name,value",
    [
        ("run", None),
        ("run", {}),
        ("run", {"records": {"unexpected": "mapping"}}),
        ("run", {"records": ({}, {}, {})}),
        ("case_set", {"cases": "not a list"}),
        ("evidence", {"baseline": [], "candidate": "not a run"}),
        ("evidence", {"baseline": {"records": None}}),
        ("policy", {"records": [{}, {}, {}]}),
    ],
)
def test_count_preflight_does_not_replace_closed_schema(name, value):
    check_record_counts(value, name, max_records=2)


@pytest.mark.parametrize(
    "name,value",
    [
        ("run", {"format": "invarlock/pipeline-run-v1", "records": {}}),
        ("case_set", {"format": "invarlock/pipeline-case-set-v1", "cases": None}),
        (
            "evidence",
            {"format": "invarlock/pipeline-evidence-v1", "baseline": []},
        ),
    ],
)
def test_malformed_containers_still_reach_production_schema(monkeypatch, name, value):
    original = contracts._validator
    visited = []

    def observed(kind):
        visited.append(kind)
        return original(kind)

    monkeypatch.setattr(contracts, "_validator", observed)
    with pytest.raises(contracts.PipelineError) as caught:
        contracts.validate(value, name)
    assert visited == [name]
    assert "too long" not in str(caught.value)


def test_short_schema_diagnostic_is_unchanged():
    error = ValidationError("is not of type 'string'", path=deque(["records", 2, "id"]))
    assert format_validation_error(error, "run") == (
        "invalid run records.2.id: is not of type 'string'"
    )


def test_long_reason_retains_location_and_both_ends():
    error = ValidationError(
        "reason-start " + "🧪" * 12000 + " reason-end: is too long",
        path=deque(["records", 2, "output"]),
    )
    message = format_validation_error(error, "run")
    assert len(message) <= 2400
    assert message.startswith("invalid run records.2.output: reason-start")
    assert message.endswith("reason-end: is too long")
    assert "[truncated]" in message


def test_huge_property_location_retains_path_ends_and_reason():
    error = ValidationError(
        "reason-start " + "x" * 10000 + " reason-end",
        path=deque(["records", 1, "key-start" + "é" * 10000 + "key-end", "output"]),
    )
    message = format_validation_error(error, "run")
    assert len(message) <= 2400
    assert "records.1.key-start" in message
    assert "key-end.output" in message
    assert "reason-start" in message and message.endswith("reason-end")


def test_real_jsonschema_array_error_is_bounded():
    error = next(Draft202012Validator({"maxItems": 2}).iter_errors(["body" * 1000] * 3))
    assert len(error.message) > 10000
    message = format_validation_error(error, "policy")
    assert len(message) <= 2400
    assert message.startswith("invalid policy : [")
    assert message.endswith("is too long")


def test_production_schema_error_is_bounded(monkeypatch):
    value = {"format": "invarlock/pipeline-run-v1", "unexpected": "reason" * 5000}
    error = ValidationError(
        "reason-start " + repr(value) + " reason-end",
        path=deque(["records", 0, "output"]),
    )

    class Validator:
        def iter_errors(self, instance):
            assert instance is value
            yield error

    monkeypatch.setattr(contracts, "_validator", lambda name: Validator())
    with pytest.raises(contracts.PipelineError) as caught:
        contracts.validate(value, "run")
    message = str(caught.value)
    assert len(message) <= 2400
    assert message.startswith("invalid run records.0.output: reason-start")
    assert message.endswith("reason-end")
