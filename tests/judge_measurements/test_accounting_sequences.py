from __future__ import annotations

import copy
import itertools
from pathlib import Path

import pytest

from invarlock.judge_measurements import RunnerOptions, render_request
from invarlock.judge_measurements.collector import _LiveCheckpoint
from invarlock.judge_measurements.contracts import canonical_payload
from invarlock.judge_measurements.runner import (
    _admission_event,
    _empty_export,
    _frozen_rows,
)
from tests.judge_measurements.test_inspect_judge import (
    _failure,
    frozen_runs,
    ingest,
)
from tests.judge_measurements.test_inspect_judge import data as data

ORDERS = [
    order
    for order in itertools.permutations(("a0", "c0", "a1", "c1"))
    if order.index("a0") < order.index("c0") and order.index("a1") < order.index("c1")
]


def _state_contents(state):
    return copy.deepcopy(
        (
            state.samples,
            state.trials,
            state.event_ids,
            state.response_owners,
            state.spent_calls,
            state.record_bytes,
            state.retained_bytes,
            state.sizes,
            state.capacity_details(),
        )
    )


def _semantic_trial(trial):
    result = copy.deepcopy(trial)
    for attempt in result["attempts"]:
        # Full import reassigns retained-source positions; it independently
        # validates these references before the semantic values are compared.
        attempt.pop("source", None)
    return result


@pytest.mark.parametrize("outcome", ["complete", "transport_error", "invalid_parse"])
@pytest.mark.parametrize("order", ORDERS)
def test_incremental_accounting_matches_full_reconstruction_after_each_transition(
    data, outcome, order
):
    plan, original, _, options = data
    frozen = _frozen_rows(**frozen_runs(data))
    exported = _empty_export(
        plan, options, RunnerOptions(Path("unused"), "correctness", 10)
    )

    def reconstruct(value):
        return _LiveCheckpoint(
            plan=plan,
            options=options,
            exported=copy.deepcopy(value),
            checkpoint=ingest(data, value),
            frozen_inputs=frozen,
        )

    state = reconstruct(exported)
    for operation in order:
        sample = copy.deepcopy(original["samples"][int(operation[-1])])
        case, side = sample["metadata"]["case_id"], sample["metadata"]["side"]
        identifier = sample["id"]
        if operation.startswith("a"):
            event = _admission_event(
                trial_id=identifier,
                attempt=1,
                request=render_request(
                    plan, input_text=frozen[case]["input"], answer=frozen[case][side]
                ),
                options=options,
            )
        else:
            event = sample["events"][0]
            if outcome == "transport_error":
                _failure(event, "transport_error")
            elif outcome == "invalid_parse":
                event["output"]["completion"] = "not a rating"
                event["call"]["response"] = {
                    "choices": [
                        {
                            "message": {"content": "not a rating"},
                            "finish_reason": "stop",
                        }
                    ]
                }
        state.replace_event(identifier, event)
        retained = copy.deepcopy(exported)
        for slot in retained["samples"]:
            slot["events"] = copy.deepcopy(state.samples[slot["id"]]["events"])
        fresh = reconstruct(retained)
        assert state.spent_calls == sum(
            len(slot["events"]) for slot in retained["samples"]
        )
        assert state.spent_calls == fresh.spent_calls
        assert state.event_ids == fresh.event_ids
        assert state.response_owners == fresh.response_owners
        assert state.record_bytes == fresh.record_bytes
        assert state.retained_bytes == fresh.retained_bytes
        assert state._storage_bounds() == fresh._storage_bounds()
        assert state.capacity_details() == fresh.capacity_details()
        assert {key: _semantic_trial(value) for key, value in state.trials.items()} == {
            key: _semantic_trial(value) for key, value in fresh.trials.items()
        }
        assert state._storage_bounds()[0] >= len(
            canonical_payload(ingest(data, retained))
        )
        for fault in ("extra-field", "credential"):
            invalid = copy.deepcopy(event)
            if fault == "extra-field":
                invalid["unexpected"] = True
            else:
                invalid["call"]["request"]["api_key"] = "test-only-placeholder"
            before = _state_contents(state)
            with pytest.raises(ValueError):
                state.replace_event(identifier, invalid)
            assert _state_contents(state) == before
        state = fresh
