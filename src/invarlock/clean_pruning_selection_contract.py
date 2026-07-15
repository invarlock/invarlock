"""Structural clean-pruning candidate/receipt selection contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from .clean_pruning_selection_common import (
    _CANDIDATE_ID_RE,
    CLEAN_PRUNING_CANDIDATE_EVALUATION_SCHEMA,
    CLEAN_PRUNING_CANDIDATE_RECORD_SCHEMA,
    CLEAN_PRUNING_EXECUTION_RECEIPT_SCHEMA,
    CLEAN_PRUNING_SELECTED_ENTRY_SCHEMA,
    CLEAN_PRUNING_SELECTION_BUNDLE_SCHEMA,
    CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
    CLEAN_PRUNING_SELECTION_RECEIPT_SCHEMA,
    MINIMUM_CLEAN_PRUNING_SELECTION_CANDIDATES,
    CleanPruningSelectionEvidenceError,
    _bound_reference,
    _decision_rule,
    _exact_mapping,
    _finite,
    _identity,
    _mapping,
    _positive_int,
    _pruning_spec,
    _report_reference,
    _selection_config,
    _selection_domain,
    _sha256,
    _sidecar_reference,
    _text,
    canonical_json_sha256,
)


def _candidate(
    value: object,
    *,
    baseline_identity: Mapping[str, str],
    selection_config: Mapping[str, object],
) -> dict[str, object]:
    payload = _exact_mapping(
        value,
        label="candidate",
        fields=frozenset({"candidate_id", "pruning", "evaluation"}),
    )
    candidate_id = _text(payload["candidate_id"], label="candidate.candidate_id")
    if _CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        raise CleanPruningSelectionEvidenceError("candidate.candidate_id is invalid")
    pruning = _pruning_spec(payload["pruning"], label="candidate.pruning")
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
    if evaluation["schema"] != CLEAN_PRUNING_CANDIDATE_EVALUATION_SCHEMA:
        raise CleanPruningSelectionEvidenceError(
            "candidate.evaluation has an unrecognized schema"
        )
    config_digest = canonical_json_sha256(selection_config)
    if (
        _sha256(
            evaluation["selection_config_sha256"],
            label="candidate.evaluation.selection_config_sha256",
        )
        != config_digest
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate.evaluation.selection_config_sha256 mismatch"
        )
    execution = _sidecar_reference(
        evaluation["execution"], label="candidate.evaluation.execution"
    )
    raw_reports = evaluation["reports"]
    if not isinstance(raw_reports, list):
        raise CleanPruningSelectionEvidenceError(
            "candidate.evaluation.reports must be a list"
        )
    schedule = _mapping(selection_config["schedule"], label="selection_config.schedule")
    expected_repeats = _positive_int(
        schedule["evaluation_repeats"],
        label="selection_config.schedule.evaluation_repeats",
    )
    if len(raw_reports) != expected_repeats or not raw_reports:
        raise CleanPruningSelectionEvidenceError(
            "candidate.evaluation.reports must retain exactly evaluation_repeats reports"
        )
    reports = [
        _report_reference(
            item,
            label=f"candidate.evaluation.reports[{index}]",
            baseline_identity=baseline_identity,
        )
        for index, item in enumerate(raw_reports)
    ]
    report_paths = [
        cast(str, cast(Mapping[str, object], item["report"])["path"])
        for item in reports
    ]
    manifest_paths = [
        cast(str, cast(Mapping[str, object], item["runtime_manifest"])["path"])
        for item in reports
    ]
    if len(report_paths) != len(set(report_paths)) or len(manifest_paths) != len(
        set(manifest_paths)
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate.evaluation.reports must retain distinct report and runtime-manifest paths"
        )
    artifact_identity = cast(
        Mapping[str, str],
        cast(Mapping[str, object], reports[0]["report"])["artifact_identity"],
    )
    for report in reports[1:]:
        observed = cast(Mapping[str, object], report["report"])
        if observed["artifact_identity"] != dict(artifact_identity):
            raise CleanPruningSelectionEvidenceError(
                "candidate.evaluation.reports artifact identities must match"
            )
    replay = _bound_reference(
        evaluation["replay"],
        label="candidate.evaluation.replay",
        baseline_identity=baseline_identity,
        artifact_identity=artifact_identity,
    )
    runtime = _bound_reference(
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
        "pruning": pruning,
        "evaluation": {
            "schema": CLEAN_PRUNING_CANDIDATE_EVALUATION_SCHEMA,
            "selection_config_sha256": config_digest,
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


def _candidate_set_payload(
    *,
    original_model_key: str,
    baseline_identity: Mapping[str, str],
    selection_domain: Mapping[str, str],
    selection_config: Mapping[str, object],
    decision_rule: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return every immutable input that makes a candidate list meaningful."""

    return {
        "schema": CLEAN_PRUNING_CANDIDATE_RECORD_SCHEMA,
        "contract_version": CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
        "original_model_key": original_model_key,
        "baseline_identity": dict(baseline_identity),
        "selection_domain": dict(selection_domain),
        "selection_config_sha256": canonical_json_sha256(selection_config),
        "decision_rule_sha256": canonical_json_sha256(decision_rule),
        "candidates": list(candidates),
    }


def _candidate_record(
    value: object, *, require_candidate_set_digest: bool
) -> dict[str, object]:
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
    if payload["schema"] != CLEAN_PRUNING_CANDIDATE_RECORD_SCHEMA:
        raise CleanPruningSelectionEvidenceError(
            "candidate record has an unrecognized schema"
        )
    if payload["contract_version"] != CLEAN_PRUNING_SELECTION_CONTRACT_VERSION:
        raise CleanPruningSelectionEvidenceError(
            "candidate record has an unrecognized contract version"
        )
    original_model_key = _text(
        payload["original_model_key"], label="original_model_key"
    )
    baseline_identity = _identity(
        payload["baseline_identity"], label="baseline_identity"
    )
    selection_domain = _selection_domain(payload["selection_domain"])
    selection_config = _selection_config(payload["selection_config"])
    decision_rule = _decision_rule(payload["decision_rule"])
    raw_candidates = payload["candidates"]
    if (
        not isinstance(raw_candidates, list)
        or len(raw_candidates) < MINIMUM_CLEAN_PRUNING_SELECTION_CANDIDATES
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate record.candidates must contain at least two candidates"
        )
    candidates = [
        _candidate(
            candidate,
            baseline_identity=baseline_identity,
            selection_config=selection_config,
        )
        for candidate in raw_candidates
    ]
    candidate_ids = [cast(str, candidate["candidate_id"]) for candidate in candidates]
    if candidate_ids != sorted(candidate_ids) or len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate record.candidates must be sorted and unique"
        )
    candidate_specs = [
        canonical_json_sha256(candidate["pruning"]) for candidate in candidates
    ]
    if len(candidate_specs) != len(set(candidate_specs)):
        raise CleanPruningSelectionEvidenceError(
            "candidate record contains duplicate canonical pruning specifications"
        )
    candidate_set = _candidate_set_payload(
        original_model_key=original_model_key,
        baseline_identity=baseline_identity,
        selection_domain=selection_domain,
        selection_config=selection_config,
        decision_rule=decision_rule,
        candidates=candidates,
    )
    candidate_set_digest = canonical_json_sha256(candidate_set)
    if require_candidate_set_digest and (
        _sha256(payload["candidate_set_sha256"], label="candidate_set_sha256")
        != candidate_set_digest
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate_set_sha256 does not match all retained candidates and inputs"
        )
    return {
        "original_model_key": original_model_key,
        "baseline_identity": baseline_identity,
        "selection_domain": selection_domain,
        "selection_config": selection_config,
        "decision_rule": decision_rule,
        "candidate_set_sha256": candidate_set_digest,
        "candidates": candidates,
    }


def canonical_clean_pruning_candidate_set_sha256(
    candidate_record: Mapping[str, object],
) -> str:
    """Compute the digest a real candidate producer must publish before select."""

    return cast(
        str,
        _candidate_record(candidate_record, require_candidate_set_digest=False)[
            "candidate_set_sha256"
        ],
    )


def _winner(candidates: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return min(
        candidates,
        key=lambda candidate: (
            cast(Mapping[str, Any], candidate["evaluation"])["metrics"]["quality_loss"],
            cast(str, candidate["candidate_id"]),
        ),
    )


def select_clean_pruning(
    candidate_record: Mapping[str, object],
) -> dict[str, object]:
    """Build a selected-entry wrapper only from a fully retained candidate record.

    This is deterministic bookkeeping, not a pruning producer.  Final use must
    call :func:`verify_clean_pruning_selection_bundle_file`, which authenticates
    the retained execution, replay, report, manifest, and runtime sidecars.
    """

    record = _candidate_record(candidate_record, require_candidate_set_digest=True)
    candidates = cast(Sequence[Mapping[str, object]], record["candidates"])
    winner = _winner(candidates)
    receipt = {
        "schema": CLEAN_PRUNING_SELECTION_RECEIPT_SCHEMA,
        "contract_version": CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
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
        "selected_pruning": winner["pruning"],
        "selected_evaluation": winner["evaluation"],
    }
    pruning = cast(Mapping[str, object], winner["pruning"])
    selected_entry = {
        "status": "selected",
        "edit_type": "magnitude_prune",
        "scope": pruning["scope"],
        "target_sparsity": pruning["target_sparsity"],
        "selection_receipt": receipt,
        "selection_receipt_sha256": canonical_json_sha256(receipt),
    }
    return {
        "schema": CLEAN_PRUNING_SELECTED_ENTRY_SCHEMA,
        "contract_version": CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
        "original_model_key": record["original_model_key"],
        "selected_entry": selected_entry,
    }


def _record_from_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    fields = frozenset(
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
            "selected_pruning",
            "selected_evaluation",
        }
    )
    payload = _exact_mapping(receipt, label="selection receipt", fields=fields)
    if payload["schema"] != CLEAN_PRUNING_SELECTION_RECEIPT_SCHEMA:
        raise CleanPruningSelectionEvidenceError(
            "selection receipt has an unrecognized schema"
        )
    if payload["contract_version"] != CLEAN_PRUNING_SELECTION_CONTRACT_VERSION:
        raise CleanPruningSelectionEvidenceError(
            "selection receipt has an unrecognized contract version"
        )
    record = _candidate_record(
        {
            "schema": CLEAN_PRUNING_CANDIDATE_RECORD_SCHEMA,
            "contract_version": CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
            "original_model_key": payload["original_model_key"],
            "baseline_identity": payload["baseline_identity"],
            "selection_domain": payload["selection_domain"],
            "selection_config": payload["selection_config"],
            "decision_rule": payload["decision_rule"],
            "candidates": payload["candidates"],
            "candidate_set_sha256": payload["candidate_set_sha256"],
        },
        require_candidate_set_digest=True,
    )
    if _sha256(
        payload["selection_config_sha256"],
        label="selection receipt.selection_config_sha256",
    ) != canonical_json_sha256(record["selection_config"]):
        raise CleanPruningSelectionEvidenceError(
            "selection receipt selection_config_sha256 mismatch"
        )
    if _sha256(
        payload["decision_rule_sha256"], label="selection receipt.decision_rule_sha256"
    ) != canonical_json_sha256(record["decision_rule"]):
        raise CleanPruningSelectionEvidenceError(
            "selection receipt decision_rule_sha256 mismatch"
        )
    candidates = cast(Sequence[Mapping[str, object]], record["candidates"])
    winner = _winner(candidates)
    if payload["selected_candidate_id"] != winner["candidate_id"]:
        raise CleanPruningSelectionEvidenceError(
            "selection receipt winner does not match deterministic mean quality loss"
        )
    if payload["selected_pruning"] != winner["pruning"]:
        raise CleanPruningSelectionEvidenceError(
            "selection receipt selected_pruning does not match winner"
        )
    if payload["selected_evaluation"] != winner["evaluation"]:
        raise CleanPruningSelectionEvidenceError(
            "selection receipt selected_evaluation does not match winner"
        )
    return {
        **record,
        "selected_candidate_id": winner["candidate_id"],
        "selected_pruning": winner["pruning"],
        "selected_evaluation": winner["evaluation"],
    }


def _selected_entry(value: object) -> dict[str, object]:
    wrapper = _exact_mapping(
        value,
        label="selected pruning entry",
        fields=frozenset(
            {"schema", "contract_version", "original_model_key", "selected_entry"}
        ),
    )
    if wrapper["schema"] != CLEAN_PRUNING_SELECTED_ENTRY_SCHEMA:
        raise CleanPruningSelectionEvidenceError(
            "selected pruning entry has an unrecognized schema"
        )
    if wrapper["contract_version"] != CLEAN_PRUNING_SELECTION_CONTRACT_VERSION:
        raise CleanPruningSelectionEvidenceError(
            "selected pruning entry has an unrecognized contract version"
        )
    original_model_key = _text(
        wrapper["original_model_key"], label="selected pruning entry.original_model_key"
    )
    selected = _exact_mapping(
        wrapper["selected_entry"],
        label="selected pruning entry.selected_entry",
        fields=frozenset(
            {
                "status",
                "edit_type",
                "scope",
                "target_sparsity",
                "selection_receipt",
                "selection_receipt_sha256",
            }
        ),
    )
    if selected["status"] != "selected" or selected["edit_type"] != "magnitude_prune":
        raise CleanPruningSelectionEvidenceError(
            "selected pruning entry must select magnitude_prune"
        )
    receipt = _record_from_receipt(
        _mapping(selected["selection_receipt"], label="selection receipt")
    )
    canonical_receipt = {
        "schema": CLEAN_PRUNING_SELECTION_RECEIPT_SCHEMA,
        "contract_version": CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
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
        "selected_pruning": receipt["selected_pruning"],
        "selected_evaluation": receipt["selected_evaluation"],
    }
    if selected["selection_receipt"] != canonical_receipt:
        raise CleanPruningSelectionEvidenceError(
            "selected pruning entry receipt is not canonical"
        )
    if _sha256(
        selected["selection_receipt_sha256"],
        label="selection_receipt_sha256",
    ) != canonical_json_sha256(canonical_receipt):
        raise CleanPruningSelectionEvidenceError(
            "selected pruning entry receipt digest mismatch"
        )
    pruning = cast(Mapping[str, object], receipt["selected_pruning"])
    if (
        selected["scope"] != pruning["scope"]
        or selected["target_sparsity"] != pruning["target_sparsity"]
        or original_model_key != receipt["original_model_key"]
    ):
        raise CleanPruningSelectionEvidenceError(
            "selected pruning entry does not match the computed winner"
        )
    return {
        "schema": CLEAN_PRUNING_SELECTED_ENTRY_SCHEMA,
        "contract_version": CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
        "original_model_key": original_model_key,
        "selected_entry": {
            "status": "selected",
            "edit_type": "magnitude_prune",
            "scope": pruning["scope"],
            "target_sparsity": pruning["target_sparsity"],
            "selection_receipt": canonical_receipt,
            "selection_receipt_sha256": canonical_json_sha256(canonical_receipt),
        },
    }


def _no_bare_selected_by(value: object, *, location: str = "$") -> None:
    """Reject the stale preset vocabulary before it can masquerade as proof."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower().startswith("selected_by_"):
                raise CleanPruningSelectionEvidenceError(
                    f"bare selected_by claim is not evidence at {location}.{key}"
                )
            _no_bare_selected_by(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _no_bare_selected_by(child, location=f"{location}[{index}]")
    elif isinstance(value, str) and value.lower().startswith("selected_by_"):
        raise CleanPruningSelectionEvidenceError(
            f"bare selected_by claim is not evidence at {location}"
        )


def _bundle_payload(entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "schema": CLEAN_PRUNING_SELECTION_BUNDLE_SCHEMA,
        "contract_version": CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
        "entries": list(entries),
    }


def canonical_clean_pruning_bundle_sha256(
    entries: Sequence[Mapping[str, object]],
) -> str:
    """Return the canonical bundle digest over every selected receipt."""

    return canonical_json_sha256(_bundle_payload(entries))


def verify_clean_pruning_selection_bundle(value: object) -> dict[str, object]:
    """Verify a pruning-only selection bundle and recompute every winner."""

    _no_bare_selected_by(value)
    payload = _exact_mapping(
        value,
        label="clean pruning selection bundle",
        fields=frozenset({"schema", "contract_version", "entries", "bundle_sha256"}),
    )
    if payload["schema"] != CLEAN_PRUNING_SELECTION_BUNDLE_SCHEMA:
        raise CleanPruningSelectionEvidenceError(
            "clean pruning selection bundle has an unrecognized schema"
        )
    if payload["contract_version"] != CLEAN_PRUNING_SELECTION_CONTRACT_VERSION:
        raise CleanPruningSelectionEvidenceError(
            "clean pruning selection bundle has an unrecognized contract version"
        )
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise CleanPruningSelectionEvidenceError(
            "clean pruning selection bundle.entries must be a non-empty list"
        )
    entries = [_selected_entry(entry) for entry in raw_entries]
    keys = [cast(str, entry["original_model_key"]) for entry in entries]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise CleanPruningSelectionEvidenceError(
            "clean pruning selection bundle entries must be sorted and unique by model"
        )
    expected_digest = canonical_clean_pruning_bundle_sha256(entries)
    if _sha256(payload["bundle_sha256"], label="bundle_sha256") != expected_digest:
        raise CleanPruningSelectionEvidenceError(
            "bundle_sha256 does not match every selected pruning receipt"
        )
    return {
        **_bundle_payload(entries),
        "bundle_sha256": expected_digest,
    }


def build_clean_pruning_execution_receipt(
    *,
    original_model_key: str,
    candidate_id: str,
    pruning: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    selection_config: Mapping[str, object],
) -> dict[str, object]:
    """Build the immutable receipt a real runner writes before evaluation.

    This is intentionally not an execution engine.  A real campaign must write
    this receipt before materialization/evaluation, pass its raw digest into the
    evaluator, and retain that evaluator-native binding in each report and
    runtime manifest.  Writing this object after evaluation alone is not
    sufficient evidence because the verifier requires those independent links.
    """

    normalized_model_key = _text(original_model_key, label="original_model_key")
    normalized_candidate_id = _text(candidate_id, label="candidate_id")
    if _CANDIDATE_ID_RE.fullmatch(normalized_candidate_id) is None:
        raise CleanPruningSelectionEvidenceError("candidate_id is invalid")
    normalized_pruning = _pruning_spec(pruning, label="candidate.pruning")
    normalized_baseline = _identity(baseline_identity, label="baseline_identity")
    normalized_config = _selection_config(selection_config)
    return {
        "schema": CLEAN_PRUNING_EXECUTION_RECEIPT_SCHEMA,
        "contract_version": CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
        "phase": "prepared_before_evaluation",
        "original_model_key": normalized_model_key,
        "candidate_id": normalized_candidate_id,
        "pruning": normalized_pruning,
        "baseline_identity": normalized_baseline,
        "selection_config": normalized_config,
        "selection_config_sha256": canonical_json_sha256(normalized_config),
    }


def validate_clean_pruning_execution_receipt(
    value: object,
    *,
    expected_model_key: str | None = None,
    expected_candidate_id: str | None = None,
    expected_pruning: Mapping[str, object] | None = None,
    expected_baseline_identity: Mapping[str, str] | None = None,
    expected_selection_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate a pre-evaluation pruning receipt at a public package boundary."""

    payload = _exact_mapping(
        value,
        label="clean pruning execution receipt",
        fields=frozenset(
            {
                "schema",
                "contract_version",
                "phase",
                "original_model_key",
                "candidate_id",
                "pruning",
                "baseline_identity",
                "selection_config",
                "selection_config_sha256",
            }
        ),
    )
    if payload["schema"] != CLEAN_PRUNING_EXECUTION_RECEIPT_SCHEMA:
        raise CleanPruningSelectionEvidenceError(
            "clean pruning execution receipt has an unrecognized schema"
        )
    if payload["contract_version"] != CLEAN_PRUNING_SELECTION_CONTRACT_VERSION:
        raise CleanPruningSelectionEvidenceError(
            "clean pruning execution receipt has an unrecognized contract version"
        )
    if payload["phase"] != "prepared_before_evaluation":
        raise CleanPruningSelectionEvidenceError(
            "clean pruning execution receipt must be prepared_before_evaluation"
        )
    original_model_key = _text(
        payload["original_model_key"],
        label="clean pruning execution receipt.original_model_key",
    )
    candidate_id = _text(
        payload["candidate_id"],
        label="clean pruning execution receipt.candidate_id",
    )
    if _CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        raise CleanPruningSelectionEvidenceError(
            "clean pruning execution receipt candidate_id is invalid"
        )
    pruning = _pruning_spec(
        payload["pruning"], label="clean pruning execution receipt.pruning"
    )
    baseline_identity = _identity(
        payload["baseline_identity"],
        label="clean pruning execution receipt.baseline_identity",
    )
    selection_config = _selection_config(payload["selection_config"])
    selection_config_sha256 = canonical_json_sha256(selection_config)
    if (
        _sha256(
            payload["selection_config_sha256"],
            label="clean pruning execution receipt.selection_config_sha256",
        )
        != selection_config_sha256
    ):
        raise CleanPruningSelectionEvidenceError(
            "clean pruning execution receipt selection_config_sha256 mismatch"
        )
    if expected_model_key is not None and original_model_key != expected_model_key:
        raise CleanPruningSelectionEvidenceError(
            "clean pruning execution receipt model key mismatch"
        )
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise CleanPruningSelectionEvidenceError(
            "clean pruning execution receipt candidate id mismatch"
        )
    if expected_pruning is not None and pruning != dict(expected_pruning):
        raise CleanPruningSelectionEvidenceError(
            "clean pruning execution receipt pruning specification mismatch"
        )
    if expected_baseline_identity is not None and baseline_identity != dict(
        expected_baseline_identity
    ):
        raise CleanPruningSelectionEvidenceError(
            "clean pruning execution receipt baseline identity mismatch"
        )
    if expected_selection_config is not None and selection_config != dict(
        expected_selection_config
    ):
        raise CleanPruningSelectionEvidenceError(
            "clean pruning execution receipt selection config mismatch"
        )
    return {
        "schema": CLEAN_PRUNING_EXECUTION_RECEIPT_SCHEMA,
        "contract_version": CLEAN_PRUNING_SELECTION_CONTRACT_VERSION,
        "phase": "prepared_before_evaluation",
        "original_model_key": original_model_key,
        "candidate_id": candidate_id,
        "pruning": pruning,
        "baseline_identity": baseline_identity,
        "selection_config": selection_config,
        "selection_config_sha256": selection_config_sha256,
    }
