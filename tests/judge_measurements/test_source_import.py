"""Retained-call imports validate plans, exact shard bytes, and every trial binding."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from invarlock.judge_measurements import source_import
from invarlock.judge_measurements.contracts import (
    JudgeMeasurementContractError,
    canonical_payload,
)
from tests.core.test_native_judge_transaction import _incomplete_measurements

FIXTURES = Path(__file__).parents[1] / "fixtures/judge_measurements"


@pytest.fixture
def material():
    return {
        name: json.loads((FIXTURES / f"{name}.json").read_bytes())
        for name in ("plan", "measurements", "baseline_run", "subject_run")
    }


def _import(material, sources=None):
    if sources is None:
        sources = {
            source["source_id"]: source["content"].encode()
            for source in material["measurements"]["sources"]
        }
    return source_import.import_judge_sources(
        sources,
        plan=material["plan"],
        baseline_run=material["baseline_run"],
        subject_run=material["subject_run"],
    )


def _source(trials):
    return canonical_payload(
        {"format": "invarlock/retained-judge-json-v1", "trials": trials}
    )


def test_retained_fixture_import_preserves_every_original_byte_and_fact(material):
    before = copy.deepcopy(material)
    result = _import(material)
    assert result == material["measurements"]
    assert material == before
    assert result["trials"] is not material["measurements"]["trials"]
    source = result["sources"][0]
    assert source["sha256"] == hashlib.sha256(source["content"].encode()).hexdigest()
    result["trials"][0]["parse"]["value"] = "0"
    assert material["measurements"]["trials"][0]["parse"]["value"] == "1"


@pytest.mark.parametrize("plan", [{}, None, {"schedule": {"expected_trials": 2}}, []])
def test_plan_is_validated_before_retained_bytes_are_parsed(
    material, monkeypatch, plan
):
    material["plan"] = plan
    monkeypatch.setattr(
        source_import,
        "parse_json_bytes",
        lambda *args, **kwargs: pytest.fail(
            "parsed a shard before validating the plan"
        ),
    )
    with pytest.raises(JudgeMeasurementContractError, match="plan"):
        _import(material)


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"\xff",
        b'{"trials":[],"trials":[]}',
        b'{"score":NaN}',
        b"[" * 1100 + b"]" * 1100,
    ],
)
def test_malformed_or_ambiguous_json_uses_public_contract_error(material, raw):
    with pytest.raises(JudgeMeasurementContractError):
        _import(material, {"source-1": raw})


@pytest.mark.parametrize(
    "sources", [None, [], {}, {str(i): b"{}" for i in range(1001)}]
)
def test_source_inventory_is_explicit_and_bounded(material, sources):
    with pytest.raises(JudgeMeasurementContractError, match="1..1000"):
        source_import.import_judge_sources(
            sources,
            plan=material["plan"],
            baseline_run=material["baseline_run"],
            subject_run=material["subject_run"],
        )


@pytest.mark.parametrize(
    "raw", ["{}", bytearray(b"{}"), None, b"", b"x" * (16 * 1024 * 1024 + 1)]
)
def test_source_bytes_are_immutable_nonempty_and_bounded(material, raw):
    with pytest.raises(JudgeMeasurementContractError, match="1 byte..16 MiB"):
        _import(material, {"source-1": raw})


def test_total_retained_bytes_are_checked_across_shards(material, monkeypatch):
    raw = _source([])
    monkeypatch.setattr(source_import, "MEASUREMENTS_MAX_BYTES", len(raw) * 2 - 1)
    with pytest.raises(JudgeMeasurementContractError, match="measurement byte limit"):
        _import(material, {"first": raw, "second": raw})


@pytest.mark.parametrize(
    "value",
    [
        [],
        1,
        None,
        {"accuracy": 1, "count": 20},
        {"format": "invarlock/retained-judge-json-v1", "trials": {}},
        {"format": "other", "trials": []},
        {"format": "invarlock/retained-judge-json-v1", "trials": [], "score": 1},
    ],
)
def test_summaries_and_unrecognized_source_shapes_cannot_form_trials(material, value):
    with pytest.raises(JudgeMeasurementContractError, match="not scores or summaries"):
        _import(material, {"source-1": canonical_payload(value)})


@pytest.mark.parametrize("trials", [[1], [None], [{}, "summary"], [{}] * 200001])
def test_trial_inventory_requires_bounded_objects(material, trials):
    with pytest.raises(JudgeMeasurementContractError, match="trial inventory"):
        _import(material, {"source-1": _source(trials)})


def test_trial_inventory_limit_applies_across_shards(material):
    with pytest.raises(JudgeMeasurementContractError, match="trial inventory"):
        _import(
            material,
            {"first": _source([{}] * 100000), "second": _source([{}] * 100001)},
        )


@pytest.mark.parametrize("source_id", ["", "bad\nname", "x" * 129, 1])
def test_source_identifiers_are_validated_without_relabeling(material, source_id):
    source = material["measurements"]["sources"][0]
    with pytest.raises(JudgeMeasurementContractError):
        _import(material, {source_id: source["content"].encode()})


def test_noncanonical_source_is_refused_without_rewriting_bytes(material):
    trials = material["measurements"]["trials"]
    raw = json.dumps(
        {"format": "invarlock/retained-judge-json-v1", "trials": trials}, indent=2
    ).encode()
    with pytest.raises(JudgeMeasurementContractError, match="canonical JSON"):
        _import(material, {"source-1": raw})


def test_shards_keep_declared_positions_and_exact_bytes(material):
    shards = {}
    for index, trial in enumerate(copy.deepcopy(material["measurements"]["trials"])):
        source_id = f"shard-{index}"
        for attempt in trial["attempts"]:
            attempt["source"].update(source_id=source_id, record_index=0)
        shards[source_id] = _source([trial])
    result = _import(material, shards)
    assert result["completeness"] == material["measurements"]["completeness"]
    assert {
        source["source_id"]: source["content"].encode() for source in result["sources"]
    } == shards
    assert len(result["trials"]) == 2


def test_explicit_incomplete_slots_remain_incomplete_without_fabrication(material):
    incomplete = _incomplete_measurements(plan=material["plan"])
    material["measurements"] = incomplete
    result = _import(material)
    assert result == incomplete
    assert result["completeness"]["completed_trials"] == 0
    assert all(trial["attempts"] == [] for trial in result["trials"])


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "unknown-source",
        "record-index",
        "attempt-index",
        "case-id",
        "trial-id",
        "plan-digest",
        "answer-digest",
        "request",
        "response",
        "rating",
        "model",
    ],
)
def test_trial_and_source_bindings_are_replayed_not_trusted(material, change):
    trials = copy.deepcopy(material["measurements"]["trials"])
    trial = trials[0]
    attempt = trial["attempts"][0]
    if change == "missing":
        trials.pop()
    elif change == "duplicate":
        trials.append(copy.deepcopy(trial))
    elif change == "unknown-source":
        attempt["source"]["source_id"] = "unknown"
    elif change == "record-index":
        attempt["source"]["record_index"] = 1
    elif change == "attempt-index":
        attempt["source"]["attempt_index"] = 1
    elif change == "case-id":
        trial["case_id"] = "unknown"
    elif change == "trial-id":
        trial["trial_id"] = "trial-" + "f" * 64
    elif change == "plan-digest":
        trial["plan_sha256"] = "f" * 64
    elif change == "answer-digest":
        trial["answer_sha256"] = "f" * 64
    elif change in ("request", "response"):
        attempt[change]["text"] = "changed"
    elif change == "rating":
        trial["parse"].update(rating="incorrect", value="0")
    else:
        attempt["resolved_model"] = "unapproved-model"
    with pytest.raises(JudgeMeasurementContractError):
        _import(material, {"source-1": _source(trials)})


@pytest.mark.parametrize("side", ["baseline_run", "subject_run"])
def test_import_authenticates_frozen_runs_before_accepting_calls(material, side):
    material[side]["records"][0]["output"] = "changed answer"
    with pytest.raises(JudgeMeasurementContractError):
        _import(material)
