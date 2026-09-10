"""Reject excessive schedules before their contents become error diagnostics."""

from collections import deque

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from invarlock.captured_contracts import CapturedContractError, load_payloads
from invarlock.evaluation_record_contracts import contracts
from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError
from invarlock.evaluation_record_contracts.validation_limits import (
    check_record_counts,
    format_validation_error,
)
from tests._evaluation_support import build_pack, example_project, rebind_pack


@pytest.mark.parametrize(
    "name,value,path",
    [
        ("run", {"records": [{}, {}, {}]}, "records"),
        ("case_set", {"cases": [{}, {}, {}]}, "cases"),
        ("pack_baseline", {"records": [{}, {}, {}]}, "records"),
        ("pack_subject", {"records": [{}, {}, {}]}, "records"),
    ],
)
def test_overcount_rejected_before_encoding_or_schema(monkeypatch, name, value, path):
    if name.startswith("pack_"):
        baseline, subject, policy = example_project("classification")
        baseline["records"] = baseline["records"][:2]
        subject["records"] = subject["records"][:2]
        key = Ed25519PrivateKey.generate()
        pack = rebind_pack(
            build_pack(baseline, subject, policy, key),
            key,
            **{name.removeprefix("pack_"): value},
        )
    monkeypatch.setattr(contracts, "MAX_RECORDS", 2)

    def forbidden(*args, **kwargs):
        pytest.fail("overcount must reject before encoding or schema construction")

    monkeypatch.setattr(contracts, "_canonical_chunks", forbidden)
    monkeypatch.setattr(contracts, "_validator", forbidden)
    error = (
        CapturedContractError if name.startswith("pack_") else EvaluationRecordsError
    )
    with pytest.raises(error, match=rf"{path}.*too long") as caught:
        if name.startswith("pack_"):
            load_payloads(pack)
        else:
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
        pytest.param("run", [], id="pack-baseline-not-a-run"),
        pytest.param("run", {"records": None}, id="pack-subject-null-records"),
        ("policy", {"records": [{}, {}, {}]}),
    ],
)
def test_count_preflight_does_not_replace_closed_schema(name, value):
    check_record_counts(value, name, max_records=2)


@pytest.mark.parametrize(
    "name,value",
    [
        ("run", {"format": "invarlock/evaluation-run-v1", "records": {}}),
        ("case_set", {"format": "invarlock/evaluation-case-set-v1", "cases": None}),
        (
            "pack_baseline",
            {"format": "invarlock/evaluation-run-v1", "records": {}},
        ),
    ],
)
def test_malformed_containers_still_reach_production_schema(monkeypatch, name, value):
    if name == "pack_baseline":
        key = Ed25519PrivateKey.generate()
        pack = rebind_pack(
            build_pack(*example_project("classification"), key), key, baseline=value
        )
    original = contracts._validator
    visited = []

    def observed(kind):
        visited.append(kind)
        return original(kind)

    monkeypatch.setattr(contracts, "_validator", observed)
    error = CapturedContractError if name == "pack_baseline" else EvaluationRecordsError
    with pytest.raises(error) as caught:
        if name == "pack_baseline":
            load_payloads(pack)
        else:
            contracts.validate(value, name)
    assert visited == (["policy", "run"] if name == "pack_baseline" else [name])
    assert "too long" not in str(caught.value)


def test_pack_nonobject_run_rejected_before_schema(monkeypatch):
    key = Ed25519PrivateKey.generate()
    pack = rebind_pack(
        build_pack(*example_project("classification"), key), key, baseline=[]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("nonobject pack payload must reject before schema construction")

    monkeypatch.setattr(contracts, "_validator", forbidden)
    with pytest.raises(CapturedContractError, match="baseline must be a JSON object"):
        load_payloads(pack)


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
    value = {"format": "invarlock/evaluation-run-v1", "unexpected": "reason" * 5000}
    error = ValidationError(
        "reason-start " + repr(value) + " reason-end",
        path=deque(["records", 0, "output"]),
    )

    class Validator:
        def iter_errors(self, instance):
            assert instance is value
            yield error

    monkeypatch.setattr(contracts, "_validator", lambda name: Validator())
    with pytest.raises(EvaluationRecordsError) as caught:
        contracts.validate(value, "run")
    message = str(caught.value)
    assert len(message) <= 2400
    assert message.startswith("invalid run records.0.output: reason-start")
    assert message.endswith("reason-end")
