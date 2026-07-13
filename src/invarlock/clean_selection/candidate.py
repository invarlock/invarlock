"""Candidate and candidate-set validation for clean selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from invarlock.clean_selection.common import (
    _CANDIDATE_ID_RE,
    _SUPPORTED_PARAMETERS,
    CANDIDATE_EVALUATION_SCHEMA,
    CANDIDATE_RECORD_SCHEMA,
    CLEAN_SELECTION_CONTRACT_VERSION,
    MINIMUM_CLEAN_SELECTION_CANDIDATES,
    TRANSFORMATION_SCOPE_POLICY,
    CleanSelectionEvidenceError,
    _candidate_report_reference,
    _decision_rule,
    _exact_mapping,
    _finite,
    _identity,
    _mapping,
    _no_bare_selected_by,
    _positive_int,
    _reference,
    _selection_config,
    _sha256,
    _sidecar_reference,
    _text,
    _transform,
    canonical_json_sha256,
)


def _candidate(
    value: object,
    *,
    edit_type: str,
    baseline_identity: Mapping[str, str],
    selection_config: Mapping[str, object],
) -> dict[str, object]:
    payload = _exact_mapping(
        value,
        label="candidate",
        fields=frozenset({"candidate_id", "transformation", "evaluation"}),
    )
    candidate_id = _text(payload["candidate_id"], label="candidate.candidate_id")
    if _CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        raise CleanSelectionEvidenceError("candidate.candidate_id is invalid")
    transform = _transform(
        payload["transformation"],
        label="candidate.transformation",
        expected_edit_type=edit_type,
    )
    evaluation = _exact_mapping(
        payload["evaluation"],
        label="candidate.evaluation",
        fields=frozenset(
            {
                "schema",
                "selection_config_sha256",
                "execution",
                "reports",
                "replay",
                "runtime",
                "metrics",
            }
        ),
    )
    if evaluation["schema"] != CANDIDATE_EVALUATION_SCHEMA:
        raise CleanSelectionEvidenceError(
            "candidate.evaluation has an unrecognized schema"
        )
    selection_config_sha256 = canonical_json_sha256(selection_config)
    if (
        _sha256(
            evaluation["selection_config_sha256"],
            label="candidate.evaluation.selection_config_sha256",
        )
        != selection_config_sha256
    ):
        raise CleanSelectionEvidenceError(
            "candidate.evaluation.selection_config_sha256 mismatch"
        )
    execution = _sidecar_reference(
        evaluation["execution"], label="candidate.evaluation.execution"
    )
    raw_reports = evaluation["reports"]
    if not isinstance(raw_reports, list):
        raise CleanSelectionEvidenceError("candidate.evaluation.reports must be a list")
    schedule = _mapping(selection_config["schedule"], label="selection_config.schedule")
    expected_repeats = _positive_int(
        schedule["evaluation_repeats"],
        label="selection_config.schedule.evaluation_repeats",
    )
    if len(raw_reports) != expected_repeats:
        raise CleanSelectionEvidenceError(
            "candidate.evaluation.reports must retain exactly evaluation_repeats reports"
        )
    if not raw_reports:
        raise CleanSelectionEvidenceError(
            "candidate.evaluation.reports must be non-empty"
        )
    reports = [
        _candidate_report_reference(
            report,
            label=f"candidate.evaluation.reports[{index}]",
            baseline_identity=baseline_identity,
        )
        for index, report in enumerate(raw_reports)
    ]
    report_paths = [
        cast(str, cast(Mapping[str, object], report["report"])["path"])
        for report in reports
    ]
    manifest_paths = [
        cast(str, cast(Mapping[str, object], report["runtime_manifest"])["path"])
        for report in reports
    ]
    if len(report_paths) != len(set(report_paths)) or len(manifest_paths) != len(
        set(manifest_paths)
    ):
        raise CleanSelectionEvidenceError(
            "candidate.evaluation.reports must retain distinct report and runtime-manifest paths"
        )
    artifact_identity = cast(
        Mapping[str, str],
        cast(Mapping[str, object], reports[0]["report"])["artifact_identity"],
    )
    for report in reports[1:]:
        observed = cast(Mapping[str, object], report["report"])
        if observed["artifact_identity"] != dict(artifact_identity):
            raise CleanSelectionEvidenceError(
                "candidate.evaluation.reports artifact identities must match"
            )
    replay = _reference(
        evaluation["replay"],
        label="candidate.evaluation.replay",
        baseline_identity=baseline_identity,
        artifact_identity=artifact_identity,
    )
    runtime = _reference(
        evaluation["runtime"],
        label="candidate.evaluation.runtime",
        baseline_identity=baseline_identity,
        artifact_identity=artifact_identity,
        replay_identity=artifact_identity,
    )
    metrics = _exact_mapping(
        evaluation["metrics"],
        label="candidate.evaluation.metrics",
        fields=frozenset({"quality_loss"}),
    )
    return {
        "candidate_id": candidate_id,
        "transformation": transform,
        "evaluation": {
            "schema": CANDIDATE_EVALUATION_SCHEMA,
            "selection_config_sha256": selection_config_sha256,
            "execution": execution,
            "reports": reports,
            "replay": replay,
            "runtime": runtime,
            "metrics": {
                "quality_loss": _finite(
                    metrics["quality_loss"],
                    label="candidate.evaluation.metrics.quality_loss",
                )
            },
        },
    }


def _candidate_record(
    value: object, *, require_candidate_set_digest: bool
) -> dict[str, object]:
    """Canonicalize a pre-selection candidate record from the public schema."""

    _no_bare_selected_by(value)
    fields = {
        "schema",
        "contract_version",
        "original_model_key",
        "baseline_identity",
        "selection_domain",
        "selection_config",
        "decision_rule",
        "candidates",
    }
    if require_candidate_set_digest:
        fields.add("candidate_set_sha256")
    payload = _exact_mapping(value, label="candidate record", fields=frozenset(fields))
    if payload["schema"] != CANDIDATE_RECORD_SCHEMA:
        raise CleanSelectionEvidenceError("candidate record has an unrecognized schema")
    if payload["contract_version"] != CLEAN_SELECTION_CONTRACT_VERSION:
        raise CleanSelectionEvidenceError(
            "candidate record has an unrecognized contract version"
        )
    model_key = _text(payload["original_model_key"], label="original_model_key")
    baseline = _identity(payload["baseline_identity"], label="baseline_identity")
    domain = _exact_mapping(
        payload["selection_domain"],
        label="selection_domain",
        fields=frozenset({"edit_type", "scope_policy"}),
    )
    edit_type = _text(domain["edit_type"], label="selection_domain.edit_type")
    if edit_type not in _SUPPORTED_PARAMETERS:
        raise CleanSelectionEvidenceError("selection_domain.edit_type is unsupported")
    if domain["scope_policy"] != TRANSFORMATION_SCOPE_POLICY:
        raise CleanSelectionEvidenceError(
            "selection_domain.scope_policy is unsupported"
        )
    normalized_domain = {
        "edit_type": edit_type,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
    }
    config = _selection_config(payload["selection_config"])
    rule = _decision_rule(payload["decision_rule"])
    raw_candidates = payload["candidates"]
    if (
        not isinstance(raw_candidates, list)
        or len(raw_candidates) < MINIMUM_CLEAN_SELECTION_CANDIDATES
    ):
        raise CleanSelectionEvidenceError(
            "candidate record.candidates must contain at least two candidates"
        )
    candidates = [
        _candidate(
            candidate,
            edit_type=edit_type,
            baseline_identity=baseline,
            selection_config=config,
        )
        for candidate in raw_candidates
    ]
    candidate_ids = [cast(str, candidate["candidate_id"]) for candidate in candidates]
    if candidate_ids != sorted(candidate_ids) or len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise CleanSelectionEvidenceError(
            "candidate record.candidates must be sorted and unique"
        )
    transformations = [
        canonical_json_sha256(candidate["transformation"]) for candidate in candidates
    ]
    if len(transformations) != len(set(transformations)):
        raise CleanSelectionEvidenceError(
            "candidate record contains duplicate canonical transformations"
        )
    candidate_set = {
        "schema": CANDIDATE_RECORD_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "selection_domain": normalized_domain,
        "candidates": candidates,
    }
    digest = canonical_json_sha256(candidate_set)
    if (
        require_candidate_set_digest
        and _sha256(payload["candidate_set_sha256"], label="candidate_set_sha256")
        != digest
    ):
        raise CleanSelectionEvidenceError(
            "candidate_set_sha256 does not match the canonical candidate set"
        )
    return {
        "original_model_key": model_key,
        "baseline_identity": baseline,
        "selection_domain": normalized_domain,
        "selection_config": config,
        "decision_rule": rule,
        "candidate_set_sha256": digest,
        "candidates": candidates,
    }


def canonical_candidate_set_sha256(candidate_record: Mapping[str, object]) -> str:
    """Compute the v1 candidate digest before a producer writes it."""

    return cast(
        str,
        _candidate_record(candidate_record, require_candidate_set_digest=False)[
            "candidate_set_sha256"
        ],
    )
