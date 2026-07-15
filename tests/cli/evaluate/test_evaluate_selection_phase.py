from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import typer

from invarlock.cli.evaluate_selection_phase import (
    EvaluateSelectionRequest,
    EvaluateSelectionRuntime,
    SelectionArtifactInputs,
    load_evaluate_selection_contexts,
)
from invarlock.core.evaluate_plan import EvaluateExecutionPolicy
from invarlock.runtime_security import RuntimeSecurityPolicy

EMPTY_INPUTS = SelectionArtifactInputs(None, None, None, None, None)
COMPLETE_INPUTS = SelectionArtifactInputs(
    "selection.json",
    "execution.json",
    "replay.json",
    "runtime.json",
    2,
)


def _request(**changes: object) -> EvaluateSelectionRequest:
    request = EvaluateSelectionRequest(
        baseline="baseline",
        subject="subject",
        assurance="strict",
        allow_network=False,
        allow_remote_code=False,
        allow_third_party_plugins=False,
        clean_selection=EMPTY_INPUTS,
        clean_pruning_selection=EMPTY_INPUTS,
    )
    return replace(request, **changes)


def _runtime(**changes: object) -> EvaluateSelectionRuntime:
    runtime = EvaluateSelectionRuntime(
        execution_policy=EvaluateExecutionPolicy(
            execution_mode="container",
            allow_host_execution=False,
            prefer_local_files_only=False,
            allow_unverified_provenance=False,
        ),
        current_security_policy_fn=RuntimeSecurityPolicy,
        delegate_model_command_fn=lambda: None,
        load_clean_selection_fn=lambda **kwargs: kwargs,
        load_clean_pruning_selection_fn=lambda **kwargs: kwargs,
    )
    return replace(runtime, **changes)


def test_selection_artifact_inputs_preserve_cli_field_order() -> None:
    assert COMPLETE_INPUTS.values() == (
        "selection.json",
        "execution.json",
        "replay.json",
        "runtime.json",
        2,
    )


def test_no_selection_delegates_model_command_without_loading_artifacts() -> None:
    events: list[str] = []
    runtime = _runtime(
        delegate_model_command_fn=lambda: events.append("delegated"),
        load_clean_selection_fn=lambda **_kwargs: pytest.fail("unexpected load"),
        load_clean_pruning_selection_fn=lambda **_kwargs: pytest.fail(
            "unexpected load"
        ),
    )
    assert load_evaluate_selection_contexts(_request(), runtime) == (None, None)
    assert events == ["delegated"]


def test_complete_clean_selection_loads_all_bound_paths_after_delegation() -> None:
    events: list[str] = []

    def load(**kwargs: object) -> dict[str, object]:
        events.append("loaded")
        return kwargs

    clean, pruning = load_evaluate_selection_contexts(
        _request(clean_selection=COMPLETE_INPUTS),
        _runtime(
            delegate_model_command_fn=lambda: events.append("delegated"),
            load_clean_selection_fn=load,
        ),
    )
    assert pruning is None
    assert events == ["delegated", "loaded"]
    assert clean == {
        "selection_config_path": Path("selection.json"),
        "execution_receipt_path": Path("execution.json"),
        "replay_path": Path("replay.json"),
        "runtime_proof_path": Path("runtime.json"),
        "repeat_index": 2,
        "baseline_path": Path("baseline"),
        "subject_path": Path("subject"),
    }


def test_complete_pruning_selection_uses_pruning_loader() -> None:
    expected = object()
    clean, pruning = load_evaluate_selection_contexts(
        _request(clean_pruning_selection=COMPLETE_INPUTS),
        _runtime(load_clean_pruning_selection_fn=lambda **_kwargs: expected),
    )
    assert clean is None
    assert pruning is expected


def test_generic_and_pruning_selection_cannot_be_combined() -> None:
    with pytest.raises(typer.BadParameter, match="mutually exclusive"):
        load_evaluate_selection_contexts(
            _request(
                clean_selection=COMPLETE_INPUTS,
                clean_pruning_selection=COMPLETE_INPUTS,
            ),
            _runtime(),
        )


def test_partial_selection_is_rejected_before_delegation() -> None:
    partial = replace(COMPLETE_INPUTS, runtime_proof=None)
    with pytest.raises(typer.BadParameter, match="requires .* together"):
        load_evaluate_selection_contexts(
            _request(clean_selection=partial),
            _runtime(
                delegate_model_command_fn=lambda: pytest.fail(
                    "invalid request must not delegate"
                )
            ),
        )


@pytest.mark.parametrize(
    ("request_changes", "runtime_changes", "expected"),
    [
        (
            {},
            {
                "execution_policy": replace(
                    _runtime().execution_policy, execution_mode="host"
                )
            },
            "execution-mode container",
        ),
        ({"assurance": "audit"}, {}, "assurance strict"),
        ({"allow_network": True}, {}, "fail-closed container runtime"),
        ({"allow_remote_code": True}, {}, "fail-closed container runtime"),
        ({"allow_third_party_plugins": True}, {}, "fail-closed container runtime"),
        (
            {},
            {"current_security_policy_fn": lambda: None},
            "fail-closed container runtime",
        ),
        (
            {},
            {
                "current_security_policy_fn": lambda: RuntimeSecurityPolicy(
                    allow_network=True
                )
            },
            "fail-closed container runtime",
        ),
        (
            {},
            {
                "current_security_policy_fn": lambda: RuntimeSecurityPolicy(
                    allow_remote_code=True
                )
            },
            "fail-closed container runtime",
        ),
        (
            {},
            {
                "current_security_policy_fn": lambda: RuntimeSecurityPolicy(
                    allow_third_party_plugins=True
                )
            },
            "fail-closed container runtime",
        ),
        (
            {},
            {
                "current_security_policy_fn": lambda: RuntimeSecurityPolicy(
                    allow_unverified_provenance=True
                )
            },
            "fail-closed container runtime",
        ),
        (
            {},
            {
                "current_security_policy_fn": lambda: RuntimeSecurityPolicy(
                    allow_host_execution=True
                )
            },
            "fail-closed container runtime",
        ),
    ],
)
def test_receipt_bound_selection_rejects_each_unsafe_runtime_allowance(
    request_changes: dict[str, object],
    runtime_changes: dict[str, object],
    expected: str,
) -> None:
    with pytest.raises(typer.BadParameter, match=expected):
        load_evaluate_selection_contexts(
            _request(clean_selection=COMPLETE_INPUTS, **request_changes),
            _runtime(**runtime_changes),
        )


@pytest.mark.parametrize("selection", ["clean", "pruning"])
def test_invalid_selection_context_is_reported_as_cli_parameter_error(
    selection: str,
) -> None:
    def invalid(**_kwargs: object) -> None:
        raise ValueError("receipt digest mismatch")

    request_changes = {
        "clean_selection"
        if selection == "clean"
        else "clean_pruning_selection": COMPLETE_INPUTS
    }
    runtime_changes = {
        "load_clean_selection_fn"
        if selection == "clean"
        else "load_clean_pruning_selection_fn": invalid
    }
    with pytest.raises(
        typer.BadParameter,
        match=f"Invalid clean{'-pruning' if selection == 'pruning' else ''}-selection evaluator context: receipt digest mismatch",
    ):
        load_evaluate_selection_contexts(
            _request(**request_changes),
            _runtime(**runtime_changes),
        )
