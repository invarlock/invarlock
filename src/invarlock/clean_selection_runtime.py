# ruff: noqa: UP045  # Public evidence tooling is still parsed on Python 3.9.
"""Evaluator-side completion of receipt-bound clean-selection reports.

This module deliberately lives with the shipped verifier instead of the shell
helpers.  A candidate campaign supplies the immutable context *before* an
evaluation starts.  The evaluator then derives report-native provenance and
the report-local selection binding from its actual output, before the normal
runtime manifest is emitted.  It is not a general-purpose report stamper.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

from .clean_selection.artifacts import (
    build_evaluator_execution_provenance,
    validate_candidate_replay_runtime,
    validate_selection_execution_receipt,
)
from .clean_selection.binding import build_candidate_report_binding
from .clean_selection.common import (
    CleanSelectionEvidenceError,
    canonical_json_sha256,
    strict_json_object_snapshot,
)
from .core.checkpoint_identity import checkpoint_tree_sha256
from .evidence_pack_json import sha256_prefixed


class CleanSelectionRuntimeError(CleanSelectionEvidenceError):
    """Raised when a candidate evaluator cannot complete a bound report."""


@dataclass(frozen=True)
class CleanSelectionEvaluationContext:
    """Pinned inputs supplied to the evaluator before candidate evaluation."""

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
    transformation: dict[str, object]
    baseline_identity: dict[str, str]
    repeat_index: int


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CleanSelectionRuntimeError(f"{label} must be an object")
    return value


def _snapshot_unchanged(
    path: Path, *, expected: bytes, label: str
) -> dict[str, object]:
    raw, payload = strict_json_object_snapshot(path, label=label)
    if raw != expected:
        raise CleanSelectionRuntimeError(f"{label} changed after evaluator startup")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary: Optional[Path] = None
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
        raise CleanSelectionRuntimeError(
            f"could not atomically write candidate evaluation report: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _assert_checkpoint_identity(
    path: Path, *, expected: Mapping[str, str], label: str
) -> None:
    if path.is_symlink() or not path.is_dir():
        raise CleanSelectionRuntimeError(
            f"{label} must be a regular checkpoint directory"
        )
    try:
        observed = checkpoint_tree_sha256(path)
    except (OSError, ValueError) as exc:
        raise CleanSelectionRuntimeError(f"{label} identity is unavailable") from exc
    if observed != expected.get("sha256"):
        raise CleanSelectionRuntimeError(
            f"{label} does not match the pre-evaluation selection identity"
        )


def load_clean_selection_evaluation_context(
    *,
    selection_config_path: Path,
    execution_receipt_path: Path,
    replay_path: Path,
    runtime_proof_path: Path,
    repeat_index: int,
    baseline_path: Optional[Path] = None,
    subject_path: Optional[Path] = None,
) -> CleanSelectionEvaluationContext:
    """Load and pin all declared candidate inputs before evaluation begins."""

    config_bytes, selection_config = strict_json_object_snapshot(
        selection_config_path, label="candidate selection config"
    )
    receipt_bytes, raw_receipt = strict_json_object_snapshot(
        execution_receipt_path, label="candidate selection execution receipt"
    )
    receipt = validate_selection_execution_receipt(
        raw_receipt,
        expected_selection_config=selection_config,
    )
    schedule = _mapping(
        selection_config.get("schedule"), label="candidate selection config.schedule"
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
        raise CleanSelectionRuntimeError(
            "candidate repeat index is outside the frozen selection schedule"
        )
    replay_bytes, replay = strict_json_object_snapshot(
        replay_path, label="candidate transformation replay"
    )
    runtime_bytes, runtime = strict_json_object_snapshot(
        runtime_proof_path, label="candidate runtime reload proof"
    )
    baseline = receipt.get("baseline_identity")
    transformation = receipt.get("transformation")
    if not isinstance(baseline, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in baseline.items()
    ):
        raise CleanSelectionRuntimeError(
            "candidate receipt baseline identity is invalid"
        )
    if not isinstance(transformation, Mapping) or not all(
        isinstance(key, str) for key in transformation
    ):
        raise CleanSelectionRuntimeError("candidate receipt transformation is invalid")
    original_model_key = receipt.get("original_model_key")
    candidate_id = receipt.get("candidate_id")
    if not isinstance(original_model_key, str) or not isinstance(candidate_id, str):
        raise CleanSelectionRuntimeError("candidate receipt identity is invalid")
    artifact_identity = validate_candidate_replay_runtime(
        replay=replay,
        runtime=runtime,
        transformation=cast(Mapping[str, object], transformation),
        baseline_identity=cast(Mapping[str, str], baseline),
    )
    if baseline_path is not None:
        _assert_checkpoint_identity(
            baseline_path,
            expected=cast(Mapping[str, str], baseline),
            label="candidate baseline checkpoint",
        )
    if subject_path is not None:
        _assert_checkpoint_identity(
            subject_path,
            expected=artifact_identity,
            label="candidate subject checkpoint",
        )
    return CleanSelectionEvaluationContext(
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
        transformation=dict(transformation),
        baseline_identity=dict(cast(Mapping[str, str], baseline)),
        repeat_index=repeat_index,
    )


def finalize_clean_selection_evaluation_report(
    report_path: Path,
    *,
    context: CleanSelectionEvaluationContext,
) -> dict[str, object]:
    """Bind a just-produced evaluator report before its runtime manifest exists.

    The report's `meta.model_id` becomes the stable original model key.  The
    concrete subject is still authenticated separately through
    `meta.model_identity`, which the binding verifies against the replayed
    artifact identity.  This prevents a filesystem path from becoming a
    public model identity while retaining the exact physical checkpoint proof.
    """

    _snapshot_unchanged(
        context.selection_config_path,
        expected=context.selection_config_bytes,
        label="candidate selection config",
    )
    _snapshot_unchanged(
        context.execution_receipt_path,
        expected=context.execution_receipt_bytes,
        label="candidate selection execution receipt",
    )
    _snapshot_unchanged(
        context.replay_path,
        expected=context.replay_bytes,
        label="candidate transformation replay",
    )
    _snapshot_unchanged(
        context.runtime_proof_path,
        expected=context.runtime_proof_bytes,
        label="candidate runtime reload proof",
    )
    _, original = strict_json_object_snapshot(
        report_path, label="candidate evaluation report"
    )
    report: dict[str, object] = dict(original)
    meta = dict(_mapping(report.get("meta"), label="candidate evaluation report.meta"))
    # The receipt is supplied before evaluation, so this logical model label is
    # evaluator-authored rather than a post-publication metadata rewrite.
    meta["model_id"] = context.original_model_key
    report["meta"] = meta
    raw_provenance = report.get("provenance")
    if raw_provenance is None:
        provenance: dict[str, object] = {}
    else:
        provenance = dict(
            _mapping(raw_provenance, label="candidate evaluation report.provenance")
        )
    native = build_evaluator_execution_provenance(
        report=report,
        execution_receipt=context.execution_receipt,
        execution_receipt_sha256=context.execution_receipt_sha256,
        repeat_index=context.repeat_index,
    )
    existing_native = provenance.get("clean_selection_execution")
    if existing_native is not None and existing_native != native:
        raise CleanSelectionRuntimeError(
            "candidate report already has incompatible evaluator selection provenance"
        )
    provenance["clean_selection_execution"] = native
    report["provenance"] = provenance
    binding = build_candidate_report_binding(
        report=report,
        replay=context.replay,
        runtime=context.runtime_proof,
        original_model_key=context.original_model_key,
        candidate_id=context.candidate_id,
        transformation=context.transformation,
        selection_config=context.selection_config,
        execution_receipt=context.execution_receipt,
        execution_receipt_sha256=context.execution_receipt_sha256,
        repeat_index=context.repeat_index,
    )
    existing_binding = report.get("clean_selection")
    if existing_binding is not None and existing_binding != binding:
        raise CleanSelectionRuntimeError(
            "candidate report already has incompatible selection binding"
        )
    report["clean_selection"] = binding
    _atomic_write_json(report_path, report)
    run_id = report.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CleanSelectionRuntimeError("candidate report run_id is invalid")
    return {
        "execution_receipt_sha256": context.execution_receipt_sha256,
        "selection_config_sha256": canonical_json_sha256(context.selection_config),
        "original_model_key": context.original_model_key,
        "candidate_id": context.candidate_id,
        "repeat_index": context.repeat_index,
        "report_run_id": run_id,
        "transformation": context.transformation,
        "baseline_identity": context.baseline_identity,
    }


__all__ = [
    "CleanSelectionEvaluationContext",
    "CleanSelectionRuntimeError",
    "finalize_clean_selection_evaluation_report",
    "load_clean_selection_evaluation_context",
]
