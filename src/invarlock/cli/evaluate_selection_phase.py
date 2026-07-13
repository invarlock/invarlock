"""Receipt-bound clean-selection preflight for the evaluate command."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from invarlock.core.evaluate_plan import EvaluateExecutionPolicy
from invarlock.runtime_security import RuntimeSecurityPolicy


@dataclass(frozen=True)
class SelectionArtifactInputs:
    config: str | None
    execution_receipt: str | None
    replay: str | None
    runtime_proof: str | None
    repeat_index: int | None

    def values(self) -> tuple[str | int | None, ...]:
        return (
            self.config,
            self.execution_receipt,
            self.replay,
            self.runtime_proof,
            self.repeat_index,
        )


@dataclass(frozen=True)
class EvaluateSelectionRequest:
    baseline: str
    subject: str
    assurance: str
    allow_network: bool
    allow_remote_code: bool
    allow_third_party_plugins: bool
    clean_selection: SelectionArtifactInputs
    clean_pruning_selection: SelectionArtifactInputs


@dataclass(frozen=True)
class EvaluateSelectionRuntime:
    execution_policy: EvaluateExecutionPolicy
    current_security_policy_fn: Callable[[], RuntimeSecurityPolicy | None]
    delegate_model_command_fn: Callable[[], Any]
    load_clean_selection_fn: Callable[..., Any]
    load_clean_pruning_selection_fn: Callable[..., Any]


def _validate_selection_inputs(
    *,
    selection_name: str,
    inputs: SelectionArtifactInputs,
    flags: str,
    request: EvaluateSelectionRequest,
    runtime: EvaluateSelectionRuntime,
) -> bool:
    values = inputs.values()
    present = any(value is not None for value in values)
    if not present:
        return False
    if any(value is None for value in values):
        raise typer.BadParameter(f"{selection_name} requires {flags} together.")
    runtime_policy = runtime.current_security_policy_fn()
    if runtime.execution_policy.execution_mode != "container":
        raise typer.BadParameter(
            f"Receipt-bound {selection_name.lower()} requires --execution-mode container."
        )
    if str(request.assurance).strip().lower() != "strict":
        raise typer.BadParameter(
            f"Receipt-bound {selection_name.lower()} requires --assurance strict."
        )
    if (
        request.allow_network
        or request.allow_remote_code
        or request.allow_third_party_plugins
        or runtime_policy is None
        or runtime_policy.allow_network
        or runtime_policy.allow_remote_code
        or runtime_policy.allow_third_party_plugins
        or runtime_policy.allow_unverified_provenance
        or runtime_policy.allow_host_execution
    ):
        raise typer.BadParameter(
            f"Receipt-bound {selection_name.lower()} requires a fail-closed container "
            "runtime with network, remote code, plugins, host bypass, and "
            "unverified provenance disabled."
        )
    return True


def _load_selection_context(
    *,
    inputs: SelectionArtifactInputs,
    request: EvaluateSelectionRequest,
    loader: Callable[..., Any],
    error_label: str,
) -> Any:
    assert inputs.config is not None
    assert inputs.execution_receipt is not None
    assert inputs.replay is not None
    assert inputs.runtime_proof is not None
    assert inputs.repeat_index is not None
    try:
        return loader(
            selection_config_path=Path(inputs.config),
            execution_receipt_path=Path(inputs.execution_receipt),
            replay_path=Path(inputs.replay),
            runtime_proof_path=Path(inputs.runtime_proof),
            repeat_index=inputs.repeat_index,
            baseline_path=Path(request.baseline),
            subject_path=Path(request.subject),
        )
    except ValueError as exc:
        raise typer.BadParameter(
            f"Invalid {error_label} evaluator context: {exc}"
        ) from exc


def load_evaluate_selection_contexts(
    request: EvaluateSelectionRequest,
    runtime: EvaluateSelectionRuntime,
) -> tuple[Any | None, Any | None]:
    """Validate and pin the optional selection artifacts before evaluation."""

    clean_values = request.clean_selection.values()
    pruning_values = request.clean_pruning_selection.values()
    if any(value is not None for value in clean_values) and any(
        value is not None for value in pruning_values
    ):
        raise typer.BadParameter(
            "Generic clean selection and clean pruning selection are mutually exclusive."
        )
    has_clean_selection = _validate_selection_inputs(
        selection_name="Clean selection",
        inputs=request.clean_selection,
        flags="--clean-selection-config, --clean-selection-execution-receipt, "
        "--clean-selection-replay, --clean-selection-runtime-proof, and "
        "--clean-selection-repeat-index",
        request=request,
        runtime=runtime,
    )
    has_clean_pruning_selection = _validate_selection_inputs(
        selection_name="Clean pruning selection",
        inputs=request.clean_pruning_selection,
        flags="--clean-pruning-selection-config, "
        "--clean-pruning-selection-execution-receipt, "
        "--clean-pruning-selection-replay, "
        "--clean-pruning-selection-runtime-proof, and "
        "--clean-pruning-selection-repeat-index",
        request=request,
        runtime=runtime,
    )
    runtime.delegate_model_command_fn()

    clean_context = None
    if has_clean_selection:
        clean_context = _load_selection_context(
            inputs=request.clean_selection,
            request=request,
            loader=runtime.load_clean_selection_fn,
            error_label="clean-selection",
        )
    pruning_context = None
    if has_clean_pruning_selection:
        pruning_context = _load_selection_context(
            inputs=request.clean_pruning_selection,
            request=request,
            loader=runtime.load_clean_pruning_selection_fn,
            error_label="clean-pruning-selection",
        )
    return clean_context, pruning_context


__all__ = [
    "EvaluateSelectionRequest",
    "EvaluateSelectionRuntime",
    "SelectionArtifactInputs",
    "load_evaluate_selection_contexts",
]
