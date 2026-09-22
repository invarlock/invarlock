"""Offline parity of per-case reference rendering and Inspect frozen rows."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from invarlock.judge_measurements import bind_requests, render_request
from invarlock.judge_measurements.collector import _render_request
from invarlock.judge_measurements.contracts import (
    canonical_payload,
    render_judge_request,
)
from invarlock.judge_measurements.runner import _frozen_rows

FIXTURES = Path(__file__).parent / "fixtures"


def _data():
    plan = json.loads((FIXTURES / "plan.json").read_text())
    frozen = json.loads((FIXTURES / "frozen.json").read_text())
    plan["prompt"]["reference_mode"] = "per_case"
    for row in frozen.values():
        row["expected"] = "gold reference"
    return bind_requests(plan, frozen), frozen


def test_core_and_addin_reference_request_bytes_are_identical_and_pinned():
    plan, frozen = _data()
    for binding in plan["answer_bindings"]:
        row = frozen[binding["case_id"]]
        for side in ("baseline", "subject"):
            request = render_request(
                plan,
                input_text=row["input"],
                answer=row[side],
                reference_text=row["expected"],
            )
            core = render_judge_request(
                plan,
                input_text=row["input"],
                answer_text=row[side],
                reference_text=row["expected"],
            )
            assert canonical_payload(request) == core
            assert hashlib.sha256(core).hexdigest() == binding[f"{side}_request_sha256"]
            user = json.loads(request["messages"][-1]["content"])
            assert user["reference"] == row["expected"]
            assert user["input"] == row["input"]
            altered = _render_request(
                plan,
                input_text=row["input"],
                answer=row[side],
                reference_text="replacement",
            )
            assert (
                hashlib.sha256(canonical_payload(altered)).hexdigest()
                != binding[f"{side}_request_sha256"]
            )


@pytest.mark.parametrize("expected", [None, {}, 1])
def test_addin_requires_per_case_text_reference(expected):
    plan, frozen = _data()
    row = next(iter(frozen.values()))
    with pytest.raises(ValueError, match="string reference"):
        render_request(
            plan,
            input_text=row["input"],
            answer=row["baseline"],
            reference_text=expected,
        )


def test_addin_frozen_rows_preserve_the_real_reference():
    baseline, subject = [
        json.loads((FIXTURES / f"{side}_run.json").read_text())
        for side in ("baseline", "subject")
    ]
    for run in (baseline, subject):
        for record in run["records"]:
            record["expected"] = "gold reference"
    rows = _frozen_rows(baseline, subject, per_case_reference=True)
    assert all(row["expected"] == "gold reference" for row in rows.values())
    assert all(
        "expected" not in row for row in _frozen_rows(baseline, subject).values()
    )
    subject["records"][0]["expected"] = "different gold"
    with pytest.raises(ValueError, match="paired references differ"):
        _frozen_rows(baseline, subject, per_case_reference=True)
    subject["records"][0]["expected"] = None
    with pytest.raises(ValueError, match="string reference"):
        _frozen_rows(baseline, subject, per_case_reference=True)


def test_opt_in_binding_rejects_a_dropped_reference():
    plan, frozen = _data()
    original = copy.deepcopy(frozen)
    for row in frozen.values():
        row.pop("expected")
    with pytest.raises(ValueError, match="frozen input"):
        bind_requests(plan, frozen)
    for row in original.values():
        row["expected"] = None
    with pytest.raises(ValueError, match="string reference"):
        bind_requests(plan, original)


def _reference_export():
    from invarlock.evaluation_records.cases import case_set_digest
    from invarlock.evaluation_records.io import run_digest
    from invarlock.judge_measurements import CollectionOptions
    from invarlock.judge_measurements.contracts import (
        expected_trial_id,
        measurement_plan_digest,
    )

    plan, frozen = _data()
    runs = {
        side: json.loads((FIXTURES / f"{side}_run.json").read_text())
        for side in ("baseline", "subject")
    }
    for side, run in runs.items():
        for record in run["records"]:
            record["expected"] = frozen[record["id"]]["expected"]
        plan[f"{side}_run_sha256"] = run_digest(run)
    plan["case_set_sha256"] = case_set_digest(
        {
            "format": "invarlock/evaluation-case-set-v1",
            "cases": [
                {key: row[key] for key in ("id", "input", "expected", "metadata")}
                for row in runs["baseline"]["records"]
            ],
        }
    )
    plan_digest = measurement_plan_digest(plan)
    exported = json.loads((FIXTURES / "export.json").read_text())
    for sample in exported["samples"]:
        metadata = sample["metadata"]
        row = frozen[metadata["case_id"]]
        request = render_request(
            plan,
            input_text=row["input"],
            answer=row[metadata["side"]],
            reference_text=row["expected"],
        )
        metadata["plan_sha256"] = plan_digest
        sample["id"] = expected_trial_id(
            plan_digest, metadata["case_id"], metadata["side"], metadata["repetition"]
        )
        for event in sample["events"]:
            event["input"] = request["messages"]
            event["call"]["request"] = request
    return (
        plan,
        frozen,
        runs,
        exported,
        CollectionOptions.from_mapping(exported["collection"]),
    )


def test_addin_import_and_live_checkpoint_replay_per_case_references():
    from invarlock.judge_measurements import import_export
    from invarlock.judge_measurements.collector import _LiveCheckpoint
    from invarlock.judge_measurements.contracts import validate_measurements

    plan, frozen, runs, exported, options = _reference_export()
    measurements = import_export(
        canonical_payload(exported),
        plan=plan,
        options=options,
        baseline_run=runs["baseline"],
        subject_run=runs["subject"],
    )
    validate_measurements(
        measurements, plan, baseline_run=runs["baseline"], subject_run=runs["subject"]
    )
    state = _LiveCheckpoint(
        plan=plan,
        options=options,
        exported=exported,
        checkpoint=measurements,
        frozen_inputs=frozen,
    )
    sample = exported["samples"][0]
    state.replace_event(sample["id"], copy.deepcopy(sample["events"][0]))
    event = copy.deepcopy(sample["events"][0])
    user = json.loads(event["input"][-1]["content"])
    user.pop("reference")
    event["input"][-1]["content"] = canonical_payload(user).decode()
    event["call"]["request"]["messages"] = event["input"]
    with pytest.raises(ValueError):
        state.replace_event(sample["id"], event)


@pytest.mark.parametrize("expected", [None, "substituted gold"])
def test_addin_import_rejects_missing_or_changed_per_case_reference(expected):
    from invarlock.judge_measurements import import_export

    plan, _, runs, exported, options = _reference_export()
    runs["subject"]["records"][0]["expected"] = expected
    with pytest.raises(ValueError, match="reference"):
        import_export(
            canonical_payload(exported),
            plan=plan,
            options=options,
            baseline_run=runs["baseline"],
            subject_run=runs["subject"],
        )
