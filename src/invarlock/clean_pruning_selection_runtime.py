"""Evaluator-side finalization for receipt-bound clean-pruning campaigns.

The evaluator receives immutable candidate inputs before it starts.  This
module snapshots and authenticates them, then writes only report-native
provenance and a derived report binding after the strict report exists.  It is
kept separate from generic transformation selection because magnitude pruning
has a different replay, scope, and topology contract.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .clean_pruning_selection_artifacts import (
    build_clean_pruning_candidate_report_binding,
    build_clean_pruning_evaluator_execution_provenance,
    validate_clean_pruning_candidate_replay_runtime,
)
from .clean_pruning_selection_common import (
    CleanPruningSelectionEvidenceError,
    canonical_json_sha256,
    strict_json_object_snapshot,
)
from .clean_pruning_selection_contract import (
    validate_clean_pruning_execution_receipt,
)
from .core.checkpoint_identity import checkpoint_tree_sha256
from .evidence_pack_json import sha256_prefixed


class CleanPruningSelectionRuntimeError(CleanPruningSelectionEvidenceError):
    """Raised when a candidate evaluator cannot retain a trusted binding."""


@dataclass(frozen=True)
class CleanPruningSelectionEvaluationContext:
    """Exact candidate inputs pinned before one evaluator repeat starts."""

    selection_config_path: Path
    selection_config_bytes: bytes
    selection_config: dict[str, object]
    execution_receipt_path: Path
    execution_receipt_bytes: bytes
    execution_receipt: dict[str, object]
    execution_receipt_sha256: str
    replay_path: Path
    replay_bytes: bytes
    replay: dict[str, object]
    runtime_proof_path: Path
    runtime_proof_bytes: bytes
    runtime_proof: dict[str, object]
    original_model_key: str
    candidate_id: str
    pruning: dict[str, object]
    baseline_identity: dict[str, str]
    repeat_index: int


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CleanPruningSelectionRuntimeError(f"{label} must be an object")
    return value


def _snapshot_unchanged(
    path: Path, *, expected: bytes, label: str
) -> dict[str, object]:
    raw, payload = strict_json_object_snapshot(path, label=label)
    if raw != expected:
        raise CleanPruningSelectionRuntimeError(
            f"{label} changed after evaluator startup"
        )
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise CleanPruningSelectionRuntimeError(
            f"could not atomically write pruning candidate evaluation report: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _assert_checkpoint_identity(
    path: Path, *, expected: Mapping[str, str], label: str
) -> None:
    if path.is_symlink() or not path.is_dir():
        raise CleanPruningSelectionRuntimeError(
            f"{label} must be a regular checkpoint directory"
        )
    try:
        observed = checkpoint_tree_sha256(path)
    except (OSError, ValueError) as exc:
        raise CleanPruningSelectionRuntimeError(
            f"{label} identity is unavailable"
        ) from exc
    if observed != expected.get("sha256"):
        raise CleanPruningSelectionRuntimeError(
            f"{label} does not match the pre-evaluation selection identity"
        )


def load_clean_pruning_selection_evaluation_context(
    *,
    selection_config_path: Path,
    execution_receipt_path: Path,
    replay_path: Path,
    runtime_proof_path: Path,
    repeat_index: int,
    baseline_path: Path | None = None,
    subject_path: Path | None = None,
) -> CleanPruningSelectionEvaluationContext:
    """Authenticate and pin one pruning candidate before evaluation starts."""

    config_bytes, selection_config = strict_json_object_snapshot(
        selection_config_path, label="pruning candidate selection config"
    )
    receipt_bytes, raw_receipt = strict_json_object_snapshot(
        execution_receipt_path, label="pruning candidate execution receipt"
    )
    receipt = validate_clean_pruning_execution_receipt(
        raw_receipt, expected_selection_config=selection_config
    )
    schedule = _mapping(
        selection_config.get("schedule"), label="pruning selection config.schedule"
    )
    declared_repeats = schedule.get("evaluation_repeats")
    if (
        isinstance(repeat_index, bool)
        or not isinstance(repeat_index, int)
        or isinstance(declared_repeats, bool)
        or not isinstance(declared_repeats, int)
        or repeat_index < 0
        or repeat_index >= declared_repeats
    ):
        raise CleanPruningSelectionRuntimeError(
            "candidate repeat index is outside the frozen pruning selection schedule"
        )
    replay_bytes, replay = strict_json_object_snapshot(
        replay_path, label="candidate pruning replay"
    )
    runtime_bytes, runtime = strict_json_object_snapshot(
        runtime_proof_path, label="candidate runtime reload proof"
    )
    baseline = _mapping(
        receipt.get("baseline_identity"), label="candidate receipt baseline identity"
    )
    pruning = _mapping(receipt.get("pruning"), label="candidate receipt pruning")
    original_model_key = receipt.get("original_model_key")
    candidate_id = receipt.get("candidate_id")
    if not isinstance(original_model_key, str) or not isinstance(candidate_id, str):
        raise CleanPruningSelectionRuntimeError("candidate receipt identity is invalid")
    baseline_identity = {str(key): str(value) for key, value in baseline.items()}
    artifact_identity = validate_clean_pruning_candidate_replay_runtime(
        replay=replay,
        runtime=runtime,
        pruning=pruning,
        baseline_identity=baseline_identity,
    )
    if baseline_path is not None:
        _assert_checkpoint_identity(
            baseline_path,
            expected=baseline_identity,
            label="candidate baseline checkpoint",
        )
    if subject_path is not None:
        _assert_checkpoint_identity(
            subject_path,
            expected=artifact_identity,
            label="candidate subject checkpoint",
        )
    return CleanPruningSelectionEvaluationContext(
        selection_config_path=selection_config_path,
        selection_config_bytes=config_bytes,
        selection_config=selection_config,
        execution_receipt_path=execution_receipt_path,
        execution_receipt_bytes=receipt_bytes,
        execution_receipt=receipt,
        execution_receipt_sha256=sha256_prefixed(receipt_bytes),
        replay_path=replay_path,
        replay_bytes=replay_bytes,
        replay=replay,
        runtime_proof_path=runtime_proof_path,
        runtime_proof_bytes=runtime_bytes,
        runtime_proof=runtime,
        original_model_key=original_model_key,
        candidate_id=candidate_id,
        pruning=dict(pruning),
        baseline_identity=baseline_identity,
        repeat_index=repeat_index,
    )


def finalize_clean_pruning_selection_evaluation_report(
    report_path: Path,
    *,
    context: CleanPruningSelectionEvaluationContext,
) -> dict[str, object]:
    """Attach evaluator-native pruning provenance to an already strict report."""

    _snapshot_unchanged(
        context.selection_config_path,
        expected=context.selection_config_bytes,
        label="pruning candidate selection config",
    )
    _snapshot_unchanged(
        context.execution_receipt_path,
        expected=context.execution_receipt_bytes,
        label="pruning candidate execution receipt",
    )
    _snapshot_unchanged(
        context.replay_path,
        expected=context.replay_bytes,
        label="candidate pruning replay",
    )
    _snapshot_unchanged(
        context.runtime_proof_path,
        expected=context.runtime_proof_bytes,
        label="candidate runtime reload proof",
    )
    _, original = strict_json_object_snapshot(
        report_path, label="pruning candidate evaluation report"
    )
    report: dict[str, object] = dict(original)
    meta = dict(_mapping(report.get("meta"), label="pruning candidate report.meta"))
    # The candidate receipt exists before evaluation, so this logical key is
    # evaluator-authored context rather than a later pack metadata rewrite.
    meta["model_id"] = context.original_model_key
    report["meta"] = meta
    raw_provenance = report.get("provenance")
    provenance = (
        {}
        if raw_provenance is None
        else dict(_mapping(raw_provenance, label="report.provenance"))
    )
    native = build_clean_pruning_evaluator_execution_provenance(
        report=report,
        execution_receipt=context.execution_receipt,
        execution_receipt_sha256=context.execution_receipt_sha256,
        repeat_index=context.repeat_index,
    )
    existing_native = provenance.get("clean_pruning_selection_execution")
    if existing_native is not None and existing_native != native:
        raise CleanPruningSelectionRuntimeError(
            "candidate report already has incompatible pruning evaluator provenance"
        )
    provenance["clean_pruning_selection_execution"] = native
    report["provenance"] = provenance
    binding = build_clean_pruning_candidate_report_binding(
        report=report,
        replay=context.replay,
        runtime=context.runtime_proof,
        original_model_key=context.original_model_key,
        candidate_id=context.candidate_id,
        pruning=context.pruning,
        selection_config=context.selection_config,
        execution_receipt=context.execution_receipt,
        execution_receipt_sha256=context.execution_receipt_sha256,
        repeat_index=context.repeat_index,
    )
    existing_binding = report.get("clean_pruning_selection")
    if existing_binding is not None and existing_binding != binding:
        raise CleanPruningSelectionRuntimeError(
            "candidate report already has incompatible pruning selection binding"
        )
    report["clean_pruning_selection"] = binding
    _atomic_write_json(report_path, report)
    run_id = report.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CleanPruningSelectionRuntimeError("candidate report run_id is invalid")
    return {
        "execution_receipt_sha256": context.execution_receipt_sha256,
        "selection_config_sha256": canonical_json_sha256(context.selection_config),
        "original_model_key": context.original_model_key,
        "candidate_id": context.candidate_id,
        "repeat_index": context.repeat_index,
        "report_run_id": run_id,
        "pruning": context.pruning,
        "baseline_identity": context.baseline_identity,
    }


__all__ = [
    "CleanPruningSelectionEvaluationContext",
    "CleanPruningSelectionRuntimeError",
    "finalize_clean_pruning_selection_evaluation_report",
    "load_clean_pruning_selection_evaluation_context",
]
