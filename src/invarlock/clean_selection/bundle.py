"""Canonical selected-entry and selection-bundle validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from invarlock.clean_selection.candidate import (
    _candidate,
    _candidate_record,
)
from invarlock.clean_selection.common import (
    _SUPPORTED_PARAMETERS,
    CANDIDATE_RECORD_SCHEMA,
    CLEAN_SELECTION_BUNDLE_SCHEMA,
    CLEAN_SELECTION_CONTRACT_VERSION,
    MINIMUM_CLEAN_SELECTION_CANDIDATES,
    SELECTED_ENTRY_SCHEMA,
    SELECTION_RECEIPT_SCHEMA,
    TRANSFORMATION_SCOPE_POLICY,
    CleanSelectionEvidenceError,
    _decision_rule,
    _exact_mapping,
    _identity,
    _mapping,
    _no_bare_selected_by,
    _selection_config,
    _sha256,
    _text,
    canonical_json_sha256,
)


def verify_selected_entry(value: object) -> dict[str, object]:
    """Verify one selected-entry wrapper and recompute its winner."""

    return _entry(value)


def select_clean_transformation(
    candidate_record: Mapping[str, object],
) -> dict[str, object]:
    """Select the eligible winner structurally; sidecars are verified at staging."""

    record = _candidate_record(candidate_record, require_candidate_set_digest=True)
    candidates = cast(Sequence[Mapping[str, object]], record["candidates"])
    winner = min(
        candidates,
        key=lambda candidate: (
            cast(Mapping[str, Any], candidate["evaluation"])["metrics"]["quality_loss"],
            cast(str, candidate["candidate_id"]),
        ),
    )
    receipt = {
        "schema": SELECTION_RECEIPT_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "original_model_key": record["original_model_key"],
        "baseline_identity": record["baseline_identity"],
        "selection_domain": record["selection_domain"],
        "selection_config": record["selection_config"],
        "selection_config_sha256": canonical_json_sha256(record["selection_config"]),
        "decision_rule": record["decision_rule"],
        "decision_rule_sha256": canonical_json_sha256(record["decision_rule"]),
        "candidate_set_sha256": record["candidate_set_sha256"],
        "candidates": list(candidates),
        "selected_candidate_id": winner["candidate_id"],
        "selected_transformation": winner["transformation"],
        "selected_evaluation": winner["evaluation"],
    }
    transformation = cast(Mapping[str, object], winner["transformation"])
    selected_entry = {
        "status": "selected",
        "edit_type": transformation["edit_type"],
        "parameters": transformation["parameters"],
        "scope": transformation["scope"],
        "selection_receipt": receipt,
        "selection_receipt_sha256": canonical_json_sha256(receipt),
    }
    return {
        "schema": SELECTED_ENTRY_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "original_model_key": record["original_model_key"],
        "selected_entry": selected_entry,
    }


def _record_from_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    expected = frozenset(
        {
            "schema",
            "contract_version",
            "original_model_key",
            "baseline_identity",
            "selection_domain",
            "selection_config",
            "selection_config_sha256",
            "decision_rule",
            "decision_rule_sha256",
            "candidate_set_sha256",
            "candidates",
            "selected_candidate_id",
            "selected_transformation",
            "selected_evaluation",
        }
    )
    payload = _exact_mapping(receipt, label="selection receipt", fields=expected)
    if payload["schema"] != SELECTION_RECEIPT_SCHEMA:
        raise CleanSelectionEvidenceError(
            "selection receipt has an unrecognized schema"
        )
    if payload["contract_version"] != CLEAN_SELECTION_CONTRACT_VERSION:
        raise CleanSelectionEvidenceError(
            "selection receipt has an unrecognized contract version"
        )
    model_key = _text(
        payload["original_model_key"], label="selection receipt.original_model_key"
    )
    baseline = _identity(
        payload["baseline_identity"], label="selection receipt.baseline_identity"
    )
    domain = _exact_mapping(
        payload["selection_domain"],
        label="selection receipt.selection_domain",
        fields=frozenset({"edit_type", "scope_policy"}),
    )
    edit_type = _text(
        domain["edit_type"], label="selection receipt.selection_domain.edit_type"
    )
    if edit_type not in _SUPPORTED_PARAMETERS:
        raise CleanSelectionEvidenceError(
            "selection receipt selection domain is unsupported"
        )
    if domain["scope_policy"] != TRANSFORMATION_SCOPE_POLICY:
        raise CleanSelectionEvidenceError(
            "selection receipt scope policy is unsupported"
        )
    normalized_domain = {
        "edit_type": edit_type,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
    }
    config = _selection_config(payload["selection_config"])
    config_digest = canonical_json_sha256(config)
    if (
        _sha256(payload["selection_config_sha256"], label="selection_config_sha256")
        != config_digest
    ):
        raise CleanSelectionEvidenceError(
            "selection receipt selection_config_sha256 mismatch"
        )
    rule = _decision_rule(payload["decision_rule"])
    if _sha256(
        payload["decision_rule_sha256"], label="decision_rule_sha256"
    ) != canonical_json_sha256(rule):
        raise CleanSelectionEvidenceError(
            "selection receipt decision_rule_sha256 mismatch"
        )
    raw_candidates = payload["candidates"]
    if (
        not isinstance(raw_candidates, list)
        or len(raw_candidates) < MINIMUM_CLEAN_SELECTION_CANDIDATES
    ):
        raise CleanSelectionEvidenceError(
            "selection receipt candidates must contain at least two candidates"
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
            "selection receipt candidates must be sorted and unique"
        )
    transformations = [
        canonical_json_sha256(candidate["transformation"]) for candidate in candidates
    ]
    if len(transformations) != len(set(transformations)):
        raise CleanSelectionEvidenceError(
            "selection receipt has duplicate transformations"
        )
    candidate_set = {
        "schema": CANDIDATE_RECORD_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "selection_domain": normalized_domain,
        "candidates": candidates,
    }
    candidate_set_digest = canonical_json_sha256(candidate_set)
    if (
        _sha256(payload["candidate_set_sha256"], label="candidate_set_sha256")
        != candidate_set_digest
    ):
        raise CleanSelectionEvidenceError(
            "selection receipt candidate_set_sha256 mismatch"
        )
    winner = min(
        candidates,
        key=lambda candidate: (
            cast(Mapping[str, Any], candidate["evaluation"])["metrics"]["quality_loss"],
            cast(str, candidate["candidate_id"]),
        ),
    )
    if payload["selected_candidate_id"] != winner["candidate_id"]:
        raise CleanSelectionEvidenceError(
            "selection receipt winner does not match the deterministic rule"
        )
    if payload["selected_transformation"] != winner["transformation"]:
        raise CleanSelectionEvidenceError(
            "selection receipt selected_transformation mismatch"
        )
    if payload["selected_evaluation"] != winner["evaluation"]:
        raise CleanSelectionEvidenceError(
            "selection receipt selected_evaluation mismatch"
        )
    return {
        "original_model_key": model_key,
        "baseline_identity": baseline,
        "selection_domain": normalized_domain,
        "selection_config": config,
        "decision_rule": rule,
        "candidate_set_sha256": candidate_set_digest,
        "candidates": candidates,
        "selected_candidate_id": winner["candidate_id"],
        "selected_transformation": winner["transformation"],
        "selected_evaluation": winner["evaluation"],
    }


def _entry(value: object) -> dict[str, object]:
    wrapper = _exact_mapping(
        value,
        label="selected entry",
        fields=frozenset(
            {"schema", "contract_version", "original_model_key", "selected_entry"}
        ),
    )
    if wrapper["schema"] != SELECTED_ENTRY_SCHEMA:
        raise CleanSelectionEvidenceError("selected entry has an unrecognized schema")
    if wrapper["contract_version"] != CLEAN_SELECTION_CONTRACT_VERSION:
        raise CleanSelectionEvidenceError(
            "selected entry has an unrecognized contract version"
        )
    model_key = _text(
        wrapper["original_model_key"], label="selected entry.original_model_key"
    )
    selected = _exact_mapping(
        wrapper["selected_entry"],
        label="selected entry.selected_entry",
        fields=frozenset(
            {
                "status",
                "edit_type",
                "parameters",
                "scope",
                "selection_receipt",
                "selection_receipt_sha256",
            }
        ),
    )
    if selected["status"] != "selected":
        raise CleanSelectionEvidenceError("selected entry status must be selected")
    receipt = _record_from_receipt(
        _mapping(selected["selection_receipt"], label="selection receipt")
    )
    # Reconstruct the contract's public receipt form exactly before checking the
    # outer selected fields; no preselected parameter can escape this recompute.
    reconstructed_receipt = {
        "schema": SELECTION_RECEIPT_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "original_model_key": receipt["original_model_key"],
        "baseline_identity": receipt["baseline_identity"],
        "selection_domain": receipt["selection_domain"],
        "selection_config": receipt["selection_config"],
        "selection_config_sha256": canonical_json_sha256(receipt["selection_config"]),
        "decision_rule": receipt["decision_rule"],
        "decision_rule_sha256": canonical_json_sha256(receipt["decision_rule"]),
        "candidate_set_sha256": receipt["candidate_set_sha256"],
        "candidates": receipt["candidates"],
        "selected_candidate_id": receipt["selected_candidate_id"],
        "selected_transformation": receipt["selected_transformation"],
        "selected_evaluation": receipt["selected_evaluation"],
    }
    if selected["selection_receipt"] != reconstructed_receipt:
        raise CleanSelectionEvidenceError("selected entry receipt is not canonical")
    if _sha256(
        selected["selection_receipt_sha256"], label="selection_receipt_sha256"
    ) != canonical_json_sha256(reconstructed_receipt):
        raise CleanSelectionEvidenceError("selected entry receipt digest mismatch")
    transformation = cast(Mapping[str, object], receipt["selected_transformation"])
    if (
        selected["edit_type"] != transformation["edit_type"]
        or selected["parameters"] != transformation["parameters"]
        or selected["scope"] != transformation["scope"]
    ):
        raise CleanSelectionEvidenceError(
            "selected entry does not match the computed winner"
        )
    if model_key != receipt["original_model_key"]:
        raise CleanSelectionEvidenceError("selected entry original_model_key mismatch")
    return {
        "schema": SELECTED_ENTRY_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "original_model_key": model_key,
        "selected_entry": {
            "status": "selected",
            "edit_type": transformation["edit_type"],
            "parameters": transformation["parameters"],
            "scope": transformation["scope"],
            "selection_receipt": reconstructed_receipt,
            "selection_receipt_sha256": canonical_json_sha256(reconstructed_receipt),
        },
    }


def verify_selection_bundle(value: object) -> dict[str, object]:
    """Verify a v1 bundle structurally and recompute each candidate winner."""

    _no_bare_selected_by(value)
    bundle = _exact_mapping(
        value,
        label="selection bundle",
        fields=frozenset({"schema", "contract_version", "entries"}),
    )
    if bundle["schema"] != CLEAN_SELECTION_BUNDLE_SCHEMA:
        raise CleanSelectionEvidenceError("selection bundle has an unrecognized schema")
    if bundle["contract_version"] != CLEAN_SELECTION_CONTRACT_VERSION:
        raise CleanSelectionEvidenceError(
            "selection bundle has an unrecognized contract version"
        )
    raw_entries = bundle["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise CleanSelectionEvidenceError("selection bundle.entries must be non-empty")
    entries = [_entry(entry) for entry in raw_entries]
    keys: list[tuple[str, str]] = []
    for entry in entries:
        selected = cast(Mapping[str, object], entry["selected_entry"])
        keys.append(
            (cast(str, entry["original_model_key"]), cast(str, selected["edit_type"]))
        )
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise CleanSelectionEvidenceError(
            "selection bundle entries must be unique and sorted by model and edit type"
        )
    return {
        "schema": CLEAN_SELECTION_BUNDLE_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "entries": entries,
    }
