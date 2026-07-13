"""Report-local binding for authenticated clean-selection evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from invarlock.clean_selection.artifacts import (
    _assert_candidate_replay_runtime,
    _assert_report_native_execution_provenance,
    _eligible_report_quality_loss,
    _execution_receipt,
)
from invarlock.clean_selection.common import (
    _CANDIDATE_ID_RE,
    REPORT_SELECTION_BINDING_SCHEMA,
    CleanSelectionEvidenceError,
    _identity,
    _positive_int,
    _selection_config,
    _sha256,
    _text,
    _transform,
    canonical_json_sha256,
)


def build_candidate_report_binding(
    *,
    report: Mapping[str, object],
    replay: Mapping[str, object],
    runtime: Mapping[str, object],
    original_model_key: str,
    candidate_id: str,
    transformation: Mapping[str, object],
    selection_config: Mapping[str, object],
    execution_receipt: Mapping[str, object],
    execution_receipt_sha256: str,
    repeat_index: int,
) -> dict[str, object]:
    """Build a binding only after report and sidecar authentication."""

    model_key = _text(original_model_key, label="original_model_key")
    if _CANDIDATE_ID_RE.fullmatch(_text(candidate_id, label="candidate_id")) is None:
        raise CleanSelectionEvidenceError("candidate_id is invalid")
    domain_transform = _transform(transformation, label="candidate transformation")
    config = _selection_config(selection_config)
    execution_digest = _sha256(
        execution_receipt_sha256, label="execution_receipt_sha256"
    )
    if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
        raise CleanSelectionEvidenceError("repeat_index must be an integer")
    expected_repeats = _positive_int(
        cast(Mapping[str, object], config["schedule"])["evaluation_repeats"],
        label="selection_config.schedule.evaluation_repeats",
    )
    if repeat_index < 0 or repeat_index >= expected_repeats:
        raise CleanSelectionEvidenceError(
            "repeat_index is outside the selection schedule"
        )
    baseline = _identity(
        replay.get("baseline_identity"), label="replay.baseline_identity"
    )
    artifact = _identity(
        replay.get("artifact_identity"), label="replay.artifact_identity"
    )
    _execution_receipt(
        execution_receipt,
        expected_model_key=model_key,
        expected_candidate_id=candidate_id,
        expected_transformation=domain_transform,
        expected_baseline_identity=baseline,
        expected_selection_config=config,
    )
    _assert_candidate_replay_runtime(
        replay,
        runtime,
        transformation=domain_transform,
        baseline_identity=baseline,
        artifact_identity=artifact,
    )
    _assert_report_native_execution_provenance(
        report,
        execution_receipt_sha256=execution_digest,
        selection_config=config,
        original_model_key=model_key,
        candidate_id=candidate_id,
        transformation=domain_transform,
        baseline_identity=baseline,
        repeat_index=repeat_index,
    )
    quality_loss = _eligible_report_quality_loss(
        report,
        model_key=model_key,
        baseline_identity=baseline,
        artifact_identity=artifact,
    )
    return {
        "schema": REPORT_SELECTION_BINDING_SCHEMA,
        "selection_config_sha256": canonical_json_sha256(config),
        "execution_receipt_sha256": execution_digest,
        "candidate_id": candidate_id,
        "original_model_key": model_key,
        "repeat_index": repeat_index,
        "transformation": domain_transform,
        "baseline_identity": baseline,
        "artifact_identity": artifact,
        "quality_loss": quality_loss,
    }
