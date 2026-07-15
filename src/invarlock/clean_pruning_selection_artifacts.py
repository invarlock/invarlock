"""Artifact-side verification and staging bridge for clean pruning selection."""

from __future__ import annotations

import math
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from .clean_pruning_selection_common import (
    _CANDIDATE_ID_RE,
    _ELIGIBLE_VALIDATION_FIELDS,
    _PRUNING_REPLAY_FIELDS,
    _RUNTIME_DEVICE_RE,
    _RUNTIME_LOAD_DIAGNOSTIC_FIELDS,
    _RUNTIME_LOAD_DIAGNOSTICS_FIELDS,
    _RUNTIME_RELOAD_PROMPT_SHA256,
    _RUNTIME_RELOAD_PROOF_FIELDS,
    _RUNTIME_STORAGE_KEY_AUDIT_ENVELOPE_FIELDS,
    _RUNTIME_STORAGE_KEY_AUDIT_FIELDS,
    CLEAN_PRUNING_EVALUATOR_PROVENANCE_SCHEMA,
    CLEAN_PRUNING_REPORT_BINDING_SCHEMA,
    PRUNING_ALGORITHM,
    PRUNING_REPLAY_SCHEMA,
    PRUNING_SCOPE_POLICY,
    PRUNING_STORAGE_POLICY,
    RUNTIME_LOAD_DIAGNOSTICS_SCHEMA,
    RUNTIME_RELOAD_PROOF_SCHEMA,
    RUNTIME_STORAGE_KEY_AUDIT_SCHEMA,
    CleanPruningSelectionEvidenceError,
    _exact_mapping,
    _finite,
    _identity,
    _mapping,
    _nonnegative_int,
    _positive_int,
    _pruning_spec,
    _selection_config,
    _sha256,
    _text,
    canonical_json_sha256,
    strict_json_object_snapshot,
)
from .clean_pruning_selection_contract import (
    validate_clean_pruning_execution_receipt,
)
from .evidence_pack_json import sha256_prefixed
from .pruning_contract import (
    PruningContractError,
    validate_pruning_target_manifest,
)


def _path_below(root: Path, relative: str, *, label: str) -> Path:
    """Resolve one reference without following symlinks at any path segment."""

    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise CleanPruningSelectionEvidenceError(
            "clean pruning selection evidence root is missing"
        ) from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise CleanPruningSelectionEvidenceError(
            "clean pruning selection evidence root must be a regular directory"
        )
    current = root
    for index, component in enumerate(relative.split("/")):
        current = current / component
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise CleanPruningSelectionEvidenceError(f"{label} is missing") from exc
        if stat.S_ISLNK(mode):
            raise CleanPruningSelectionEvidenceError(
                f"{label} must not traverse a symlink"
            )
        if index < len(relative.split("/")) - 1:
            if not stat.S_ISDIR(mode):
                raise CleanPruningSelectionEvidenceError(
                    f"{label} has a non-directory parent"
                )
        elif not stat.S_ISREG(mode):
            raise CleanPruningSelectionEvidenceError(f"{label} must be a regular file")
    return current


def _read_referenced_json_snapshot(
    reference: Mapping[str, object],
    *,
    evidence_root: Path,
    label: str,
) -> tuple[bytes, dict[str, object]]:
    path = _path_below(
        evidence_root,
        cast(str, reference["path"]),
        label=label,
    )
    raw, payload = strict_json_object_snapshot(path, label=label)
    if sha256_prefixed(raw) != reference["sha256"]:
        raise CleanPruningSelectionEvidenceError(f"{label} digest mismatch")
    return raw, payload


def _ordered_two_arm_schedule(report: Mapping[str, object]) -> dict[str, object]:
    windows = _mapping(
        report.get("evaluation_windows"), label="candidate report.evaluation_windows"
    )
    ordered: dict[str, object] = {}
    for arm in ("preview", "final"):
        arm_payload = _mapping(
            windows.get(arm), label=f"candidate report.evaluation_windows.{arm}"
        )
        ids = arm_payload.get("window_ids")
        if not isinstance(ids, list) or not ids:
            raise CleanPruningSelectionEvidenceError(
                f"candidate report.evaluation_windows.{arm}.window_ids must be non-empty"
            )
        if any(
            isinstance(item, bool) or not isinstance(item, (int, str)) for item in ids
        ):
            raise CleanPruningSelectionEvidenceError(
                f"candidate report.evaluation_windows.{arm}.window_ids are invalid"
            )
        ordered[arm] = list(ids)
    return ordered


def _assert_report_native_execution_provenance(
    report: Mapping[str, object],
    *,
    execution_receipt_sha256: str,
    selection_config: Mapping[str, object],
    original_model_key: str,
    candidate_id: str,
    pruning: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    repeat_index: int,
) -> None:
    """Require evaluator-created identity/schedule provenance, not a static claim."""

    config = _selection_config(selection_config)
    schedule = cast(Mapping[str, object], config["schedule"])
    dataset = cast(Mapping[str, object], config["dataset"])
    provenance = _mapping(report.get("provenance"), label="candidate report.provenance")
    if not isinstance(provenance.get("clean_pruning_selection_execution"), Mapping):
        raise CleanPruningSelectionEvidenceError(
            "candidate report evaluator provenance is missing or not an object"
        )
    native = _exact_mapping(
        provenance.get("clean_pruning_selection_execution"),
        label="candidate report.provenance.clean_pruning_selection_execution",
        fields=frozenset(
            {
                "schema",
                "execution_receipt_sha256",
                "selection_config_sha256",
                "original_model_key",
                "candidate_id",
                "repeat_index",
                "report_run_id",
                "pruning",
                "baseline_identity",
                "dataset",
                "seed",
                "effective_schedule",
                "ordered_two_arm_schedule_sha256",
            }
        ),
    )
    if native["schema"] != CLEAN_PRUNING_EVALUATOR_PROVENANCE_SCHEMA:
        raise CleanPruningSelectionEvidenceError(
            "candidate report evaluator provenance schema is invalid"
        )
    if (
        native["execution_receipt_sha256"] != execution_receipt_sha256
        or native["selection_config_sha256"] != canonical_json_sha256(config)
        or native["original_model_key"] != original_model_key
        or native["candidate_id"] != candidate_id
        or native["pruning"] != dict(pruning)
        or native["baseline_identity"] != dict(baseline_identity)
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate report evaluator provenance identity mismatch"
        )
    if (
        isinstance(native["repeat_index"], bool)
        or not isinstance(native["repeat_index"], int)
        or native["repeat_index"] != repeat_index
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate report evaluator provenance repeat index mismatch"
        )
    if native["report_run_id"] != report.get("run_id"):
        raise CleanPruningSelectionEvidenceError(
            "candidate report evaluator provenance run id mismatch"
        )
    if native["dataset"] != dict(dataset) or native["seed"] != config["seed"]:
        raise CleanPruningSelectionEvidenceError(
            "candidate report evaluator provenance dataset or seed mismatch"
        )
    if native["effective_schedule"] != dict(schedule):
        raise CleanPruningSelectionEvidenceError(
            "candidate report evaluator provenance schedule mismatch"
        )
    meta = _mapping(report.get("meta"), label="candidate report.meta")
    if meta.get("seed") != config["seed"]:
        raise CleanPruningSelectionEvidenceError(
            "candidate report seed does not match selection config"
        )
    report_dataset = _mapping(report.get("dataset"), label="candidate report.dataset")
    if (
        report_dataset.get("dataset_name") != dataset["name"]
        or report_dataset.get("revision") != dataset["revision"]
        or report_dataset.get("split") != dataset["split"]
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate report dataset fields do not match immutable selection dataset"
        )
    dataset_hash = _mapping(
        report_dataset.get("hash"), label="candidate report.dataset.hash"
    )
    if dataset_hash.get("source") == "config_fallback":
        raise CleanPruningSelectionEvidenceError(
            "candidate report dataset hash must not use config_fallback"
        )
    dataset_windows = _mapping(
        report_dataset.get("windows"), label="candidate report.dataset.windows"
    )
    if dataset_windows.get("seed") != config["seed"]:
        raise CleanPruningSelectionEvidenceError(
            "candidate report dataset window seed does not match selection config"
        )
    ordered = _ordered_two_arm_schedule(report)
    max_examples = _positive_int(
        schedule["max_examples"], label="selection_config.schedule.max_examples"
    )
    if any(
        len(cast(list[object], ordered[arm])) != max_examples
        for arm in ("preview", "final")
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate report ordered schedule does not retain max_examples per arm"
        )
    if native["ordered_two_arm_schedule_sha256"] != canonical_json_sha256(ordered):
        raise CleanPruningSelectionEvidenceError(
            "candidate report evaluator provenance ordered schedule digest mismatch"
        )


def _eligible_report_quality_loss(
    report: Mapping[str, object],
    *,
    original_model_key: str,
    baseline_identity: Mapping[str, str],
    artifact_identity: Mapping[str, str],
) -> float:
    from .reporting.report_schema import validate_report

    if not validate_report(dict(report)):
        raise CleanPruningSelectionEvidenceError(
            "candidate report is not a schema-valid InvarLock evaluation report"
        )
    meta = _mapping(report.get("meta"), label="candidate report.meta")
    if meta.get("model_id") != original_model_key or meta.get("model_identity") != dict(
        artifact_identity
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate report model or artifact identity mismatch"
        )
    baseline_ref = _mapping(
        report.get("baseline_ref"), label="candidate report.baseline_ref"
    )
    if baseline_ref.get("model_identity") != dict(baseline_identity):
        raise CleanPruningSelectionEvidenceError(
            "candidate report baseline identity mismatch"
        )
    assurance = _mapping(report.get("assurance"), label="candidate report.assurance")
    if (
        assurance.get("mode") != "strict"
        or assurance.get("report_local_verdict") != "pass"
        or assurance.get("canonical_guard_chain_enforced") is not True
        or assurance.get("fallback_fields_used") is not False
        or assurance.get("blocking_reasons") != []
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate report is not an eligible strict assurance pass"
        )
    validation = _mapping(report.get("validation"), label="candidate report.validation")
    if any(validation.get(field) is not True for field in _ELIGIBLE_VALIDATION_FIELDS):
        raise CleanPruningSelectionEvidenceError(
            "candidate report has an ineligible guard result"
        )
    invariants = _mapping(report.get("invariants"), label="candidate report.invariants")
    if invariants.get("passed") is not True or invariants.get("supported") is not True:
        raise CleanPruningSelectionEvidenceError(
            "candidate report invariants are not eligible"
        )
    primary_metric = _mapping(
        report.get("primary_metric"), label="candidate report.primary_metric"
    )
    ratio = _finite(
        primary_metric.get("ratio_vs_baseline"),
        label="candidate report.primary_metric.ratio_vs_baseline",
    )
    if ratio <= 0.0:
        raise CleanPruningSelectionEvidenceError(
            "candidate report quality ratio must be positive"
        )
    return ratio - 1.0


def _assert_report_binding(
    report: Mapping[str, object],
    *,
    selection_config_sha256: str,
    execution_receipt_sha256: str,
    original_model_key: str,
    candidate_id: str,
    pruning: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    artifact_identity: Mapping[str, str],
    repeat_index: int,
) -> float:
    quality_loss = _eligible_report_quality_loss(
        report,
        original_model_key=original_model_key,
        baseline_identity=baseline_identity,
        artifact_identity=artifact_identity,
    )
    binding = _exact_mapping(
        report.get("clean_pruning_selection"),
        label="candidate report.clean_pruning_selection",
        fields=frozenset(
            {
                "schema",
                "selection_config_sha256",
                "execution_receipt_sha256",
                "candidate_id",
                "original_model_key",
                "repeat_index",
                "pruning",
                "baseline_identity",
                "artifact_identity",
                "quality_loss",
            }
        ),
    )
    if binding["schema"] != CLEAN_PRUNING_REPORT_BINDING_SCHEMA:
        raise CleanPruningSelectionEvidenceError(
            "candidate report pruning selection binding schema is invalid"
        )
    if (
        binding["selection_config_sha256"] != selection_config_sha256
        or binding["execution_receipt_sha256"] != execution_receipt_sha256
        or binding["candidate_id"] != candidate_id
        or binding["original_model_key"] != original_model_key
        or binding["pruning"] != dict(pruning)
        or binding["baseline_identity"] != dict(baseline_identity)
        or binding["artifact_identity"] != dict(artifact_identity)
        or binding["repeat_index"] != repeat_index
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate report pruning selection binding mismatch"
        )
    if not math.isclose(
        _finite(
            binding["quality_loss"],
            label="candidate report clean_pruning_selection.quality_loss",
        ),
        quality_loss,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate report quality loss is not derived from primary metric"
        )
    return quality_loss


def _assert_report_runtime_manifest(
    *,
    report_bytes: bytes,
    report: Mapping[str, object],
    manifest: Mapping[str, object],
    report_reference: Mapping[str, object],
    manifest_reference: Mapping[str, object],
    execution_receipt_sha256: str,
    selection_config_sha256: str,
    original_model_key: str,
    candidate_id: str,
    pruning: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    repeat_index: int,
) -> None:
    """Verify strict runtime binding from the already-read report/manifest bytes."""

    from .runtime_verify import verify_runtime_manifest_snapshot

    result = verify_runtime_manifest_snapshot(
        report_bytes,
        dict(manifest),
        report=Path(cast(str, report_reference["path"])),
        manifest=Path(cast(str, manifest_reference["path"])),
        require_strict_runtime=True,
    )
    if not result.ok:
        detail = "; ".join(result.errors) or "unknown runtime-manifest failure"
        raise CleanPruningSelectionEvidenceError(
            "candidate report runtime manifest is not an eligible strict binding: "
            f"{detail}"
        )
    runtime = _mapping(
        manifest.get("runtime"), label="candidate runtime manifest.runtime"
    )
    if runtime.get("allow_network") is not False:
        raise CleanPruningSelectionEvidenceError(
            "candidate report runtime manifest must record allow_network=false"
        )
    context = _mapping(
        manifest.get("context"), label="candidate runtime manifest.context"
    )
    link = _exact_mapping(
        context.get("clean_pruning_selection_execution"),
        label="candidate runtime manifest.context.clean_pruning_selection_execution",
        fields=frozenset(
            {
                "execution_receipt_sha256",
                "selection_config_sha256",
                "original_model_key",
                "candidate_id",
                "repeat_index",
                "report_run_id",
                "pruning",
                "baseline_identity",
            }
        ),
    )
    if (
        link["execution_receipt_sha256"] != execution_receipt_sha256
        or link["selection_config_sha256"] != selection_config_sha256
        or link["original_model_key"] != original_model_key
        or link["candidate_id"] != candidate_id
        or link["repeat_index"] != repeat_index
        or link["report_run_id"] != report.get("run_id")
        or link["pruning"] != dict(pruning)
        or link["baseline_identity"] != dict(baseline_identity)
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate runtime manifest clean-pruning execution linkage mismatch"
        )


def _target_manifest(
    value: object,
    *,
    expected_scope: str,
) -> dict[str, object]:
    """Validate replay topology with the package-owned pruning policy."""

    try:
        return validate_pruning_target_manifest(
            value,
            expected_scope=expected_scope,
        )
    except PruningContractError as exc:
        raise CleanPruningSelectionEvidenceError(str(exc)) from exc


def _assert_pruning_replay(
    replay: Mapping[str, object],
    *,
    pruning: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    artifact_identity: Mapping[str, str],
) -> None:
    """Verify the exact artifact replay emitted by magnitude-prune validation.

    The algorithm identifier is the tie-breaker commitment: it names the exact
    flattened-magnitude routine that resolves equal magnitudes by local flat
    index.  The target manifest is checked in full rather than trusted as a
    digest-only topology assertion.
    """

    if frozenset(replay) != _PRUNING_REPLAY_FIELDS:
        raise CleanPruningSelectionEvidenceError(
            "pruning replay sidecar has unbound or missing fields"
        )
    if (
        replay["schema"] != PRUNING_REPLAY_SCHEMA
        or replay["ok"] is not True
        or replay["issues"] != []
        or replay["edit_type"] != "magnitude_prune"
        or replay["scope"] != pruning["scope"]
        or replay["scope_policy"] != PRUNING_SCOPE_POLICY
        or replay["pruning_algorithm"] != PRUNING_ALGORITHM
        or replay["storage_policy"] != PRUNING_STORAGE_POLICY
        or replay["baseline_identity"] != dict(baseline_identity)
        or replay["artifact_identity"] != dict(artifact_identity)
    ):
        raise CleanPruningSelectionEvidenceError(
            "pruning replay sidecar does not bind the exact candidate or identities"
        )
    target_sparsity = _finite(
        replay["target_sparsity"], label="pruning replay.target_sparsity"
    )
    if not math.isclose(
        target_sparsity,
        cast(float, pruning["target_sparsity"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise CleanPruningSelectionEvidenceError(
            "pruning replay target_sparsity does not match candidate"
        )
    manifest = _target_manifest(
        replay["target_manifest"], expected_scope=cast(str, pruning["scope"])
    )
    if (
        replay["model_type"] != manifest["model_type"]
        or replay["architecture"] != manifest["architecture"]
        or replay["config_sha256"] != manifest["config_sha256"]
    ):
        raise CleanPruningSelectionEvidenceError(
            "pruning replay topology does not match its target manifest"
        )
    if _sha256(
        replay["target_manifest_sha256"],
        label="pruning replay.target_manifest_sha256",
    ) != canonical_json_sha256(manifest):
        raise CleanPruningSelectionEvidenceError(
            "pruning replay target manifest digest mismatch"
        )
    targets = cast(list[Mapping[str, object]], manifest["targets"])
    selected_tensors = _positive_int(
        replay["selected_tensors"], label="pruning replay.selected_tensors"
    )
    selected_params = _positive_int(
        replay["selected_params"], label="pruning replay.selected_params"
    )
    total_params = _positive_int(
        replay["total_params"], label="pruning replay.total_params"
    )
    if selected_params > total_params:
        raise CleanPruningSelectionEvidenceError(
            "pruning replay selected parameters exceed total parameters"
        )
    target_params = sum(cast(int, target["numel"]) for target in targets)
    if selected_tensors != len(targets) or selected_params != target_params:
        raise CleanPruningSelectionEvidenceError(
            "pruning replay selected topology counters do not match target manifest"
        )
    expected_pruned_params = _positive_int(
        replay["expected_pruned_params"],
        label="pruning replay.expected_pruned_params",
    )
    expected_from_targets = sum(
        int(cast(int, target["numel"]) * target_sparsity) for target in targets
    )
    if expected_pruned_params != expected_from_targets:
        raise CleanPruningSelectionEvidenceError(
            "pruning replay expected_pruned_params does not match exact per-tensor rule"
        )
    expected_changed = _positive_int(
        replay["expected_changed_params"],
        label="pruning replay.expected_changed_params",
    )
    observed_changed = _positive_int(
        replay["observed_changed_params"],
        label="pruning replay.observed_changed_params",
    )
    if (
        expected_changed > expected_pruned_params
        or observed_changed != expected_changed
    ):
        raise CleanPruningSelectionEvidenceError(
            "pruning replay observed changes do not match exact replay"
        )
    original_zero = _nonnegative_int(
        replay["original_zero_params"], label="pruning replay.original_zero_params"
    )
    observed_zero = _nonnegative_int(
        replay["observed_zero_params"], label="pruning replay.observed_zero_params"
    )
    if observed_zero < original_zero or observed_zero < expected_pruned_params:
        raise CleanPruningSelectionEvidenceError(
            "pruning replay zero counts are incompatible with magnitude pruning"
        )
    checked_tensors = _positive_int(
        replay["checked_tensors"], label="pruning replay.checked_tensors"
    )
    out_of_scope_tensors = _nonnegative_int(
        replay["out_of_scope_tensors_checked"],
        label="pruning replay.out_of_scope_tensors_checked",
    )
    if checked_tensors != selected_tensors + out_of_scope_tensors:
        raise CleanPruningSelectionEvidenceError(
            "pruning replay checked tensor count does not cover exact topology"
        )
    _nonnegative_int(
        replay["out_of_scope_bytes_checked"],
        label="pruning replay.out_of_scope_bytes_checked",
    )
    if out_of_scope_tensors and replay["out_of_scope_bytes_checked"] == 0:
        raise CleanPruningSelectionEvidenceError(
            "pruning replay did not bind byte coverage for out-of-scope tensors"
        )
    _positive_int(
        replay["support_files_checked"], label="pruning replay.support_files_checked"
    )


def _runtime_shape(value: object, *, label: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        raise CleanPruningSelectionEvidenceError(f"{label} must be a positive shape")


def _assert_runtime_reload_proof(
    runtime: Mapping[str, object],
    *,
    artifact_identity: Mapping[str, str],
) -> None:
    if frozenset(runtime) != _RUNTIME_RELOAD_PROOF_FIELDS:
        raise CleanPruningSelectionEvidenceError(
            "pruning runtime reload proof has unbound or missing fields"
        )
    for field in ("prompt_sha256", "token_ids_sha256", "logits_sha256"):
        _sha256(runtime[field], label=f"pruning runtime reload proof.{field}")
    if (
        runtime["schema"] != RUNTIME_RELOAD_PROOF_SCHEMA
        or runtime["ok"] is not True
        or runtime["replay_schema"] != PRUNING_REPLAY_SCHEMA
        or runtime["edit_type"] != "magnitude_prune"
        or runtime["artifact_identity"] != dict(artifact_identity)
        or runtime["replay_artifact_identity"] != dict(artifact_identity)
        or runtime["prompt_sha256"] != _RUNTIME_RELOAD_PROMPT_SHA256
        or runtime["reload_runs"] != 2
        or runtime["all_logits_finite"] is not True
        or runtime["repeat_deterministic"] is not True
        or not isinstance(runtime["device"], str)
        or _RUNTIME_DEVICE_RE.fullmatch(runtime["device"]) is None
        or not isinstance(runtime["input_device"], str)
        or _RUNTIME_DEVICE_RE.fullmatch(runtime["input_device"]) is None
    ):
        raise CleanPruningSelectionEvidenceError(
            "pruning runtime reload proof is not an eligible two-reload proof"
        )
    _runtime_shape(runtime["token_ids_shape"], label="pruning runtime token_ids_shape")
    _runtime_shape(runtime["logits_shape"], label="pruning runtime logits_shape")
    load_diagnostics = runtime.get("load_diagnostics")
    if (
        not isinstance(load_diagnostics, Mapping)
        or frozenset(load_diagnostics) != _RUNTIME_LOAD_DIAGNOSTICS_FIELDS
        or load_diagnostics.get("schema") != RUNTIME_LOAD_DIAGNOSTICS_SCHEMA
    ):
        raise CleanPruningSelectionEvidenceError(
            "pruning runtime reload proof load diagnostics are invalid"
        )
    reloads = load_diagnostics.get("reloads")
    if not isinstance(reloads, list) or len(reloads) != 2:
        raise CleanPruningSelectionEvidenceError(
            "pruning runtime reload proof load diagnostics must bind exactly two reloads"
        )
    for index, diagnostic in enumerate(reloads):
        if (
            not isinstance(diagnostic, Mapping)
            or frozenset(diagnostic) != _RUNTIME_LOAD_DIAGNOSTIC_FIELDS
        ):
            raise CleanPruningSelectionEvidenceError(
                f"pruning runtime reload proof load diagnostics reload {index} is invalid"
            )
        for field in _RUNTIME_LOAD_DIAGNOSTIC_FIELDS:
            entries = diagnostic.get(field)
            if not isinstance(entries, list) or entries:
                raise CleanPruningSelectionEvidenceError(
                    "pruning runtime reload proof load diagnostics "
                    f"reload {index} reports {field}"
                )
    storage_key_audit = runtime.get("storage_key_audit")
    if (
        not isinstance(storage_key_audit, Mapping)
        or frozenset(storage_key_audit) != _RUNTIME_STORAGE_KEY_AUDIT_ENVELOPE_FIELDS
        or storage_key_audit.get("schema") != RUNTIME_STORAGE_KEY_AUDIT_SCHEMA
    ):
        raise CleanPruningSelectionEvidenceError(
            "pruning runtime reload proof storage-key audit is invalid"
        )
    audits = storage_key_audit.get("reloads")
    if not isinstance(audits, list) or len(audits) != 2:
        raise CleanPruningSelectionEvidenceError(
            "pruning runtime reload proof storage-key audit must bind exactly two reloads"
        )
    expected_audit: dict[str, object] | None = None
    for index, audit in enumerate(audits):
        if (
            not isinstance(audit, Mapping)
            or frozenset(audit) != _RUNTIME_STORAGE_KEY_AUDIT_FIELDS
        ):
            raise CleanPruningSelectionEvidenceError(
                f"pruning runtime reload proof storage-key audit reload {index} is invalid"
            )
        artifact_storage_key_count = _positive_int(
            audit.get("artifact_storage_key_count"),
            label=(
                "pruning runtime reload proof storage-key audit "
                f"reload {index} artifact_storage_key_count"
            ),
        )
        model_state_key_count = _positive_int(
            audit.get("model_state_key_count"),
            label=(
                "pruning runtime reload proof storage-key audit "
                f"reload {index} model_state_key_count"
            ),
        )
        if artifact_storage_key_count > model_state_key_count:
            raise CleanPruningSelectionEvidenceError(
                "pruning runtime reload proof storage-key audit "
                f"reload {index} has more artifact storage keys than model state keys"
            )
        for field in (
            "artifact_storage_keys_sha256",
            "model_state_keys_sha256",
        ):
            _sha256(
                audit.get(field),
                label=(
                    "pruning runtime reload proof storage-key audit "
                    f"reload {index} {field}"
                ),
            )
        if audit.get("unexpected_storage_keys") != []:
            raise CleanPruningSelectionEvidenceError(
                f"pruning runtime reload proof storage-key audit reload {index} has unexpected storage keys"
            )
        normalized = dict(audit)
        if expected_audit is None:
            expected_audit = normalized
        elif normalized != expected_audit:
            raise CleanPruningSelectionEvidenceError(
                "pruning runtime reload proof storage-key audits disagree across reloads"
            )


def validate_clean_pruning_candidate_replay_runtime(
    *,
    replay: Mapping[str, object],
    runtime: Mapping[str, object],
    pruning: Mapping[str, object],
    baseline_identity: Mapping[str, str],
) -> dict[str, str]:
    """Validate a candidate's independent replay and two-reload proof.

    Evaluator startup needs to reject a swapped subject before it performs a
    costly strict evaluation.  This public narrow validator deliberately
    authenticates only the replay/runtime pair; report eligibility remains the
    responsibility of :func:`build_clean_pruning_candidate_report_binding`.
    """

    normalized_pruning = _pruning_spec(pruning, label="candidate.pruning")
    normalized_baseline = _identity(
        baseline_identity, label="candidate.baseline_identity"
    )
    artifact_identity = _identity(
        replay.get("artifact_identity"), label="pruning replay.artifact_identity"
    )
    _assert_pruning_replay(
        replay,
        pruning=normalized_pruning,
        baseline_identity=normalized_baseline,
        artifact_identity=artifact_identity,
    )
    _assert_runtime_reload_proof(runtime, artifact_identity=artifact_identity)
    return artifact_identity


def build_clean_pruning_evaluator_execution_provenance(
    *,
    report: Mapping[str, object],
    execution_receipt: Mapping[str, object],
    execution_receipt_sha256: str,
    repeat_index: int,
) -> dict[str, object]:
    """Derive report-native pre-evaluation provenance from real evaluator output.

    The caller cannot provide a run ID, window order, model identity, or
    schedule as independent arguments.  Those values are derived from the
    pinned pre-evaluation receipt and the evaluator's already-produced report.
    A later binding verifier checks this record against the report and the
    sibling runtime manifest.
    """

    receipt = validate_clean_pruning_execution_receipt(execution_receipt)
    config = cast(Mapping[str, object], receipt["selection_config"])
    schedule = cast(Mapping[str, object], config["schedule"])
    expected_repeats = _positive_int(
        schedule["evaluation_repeats"],
        label="selection_config.schedule.evaluation_repeats",
    )
    if (
        isinstance(repeat_index, bool)
        or not isinstance(repeat_index, int)
        or repeat_index < 0
        or repeat_index >= expected_repeats
    ):
        raise CleanPruningSelectionEvidenceError(
            "repeat_index is outside the pruning selection schedule"
        )
    execution_digest = _sha256(
        execution_receipt_sha256, label="execution_receipt_sha256"
    )
    run_id = _text(report.get("run_id"), label="candidate report.run_id")
    ordered = _ordered_two_arm_schedule(report)
    max_examples = _positive_int(
        schedule["max_examples"], label="selection_config.schedule.max_examples"
    )
    if any(
        len(cast(list[object], ordered[arm])) != max_examples
        for arm in ("preview", "final")
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate report ordered schedule does not retain max_examples per arm"
        )
    dataset = cast(Mapping[str, object], config["dataset"])
    meta = _mapping(report.get("meta"), label="candidate report.meta")
    report_dataset = _mapping(report.get("dataset"), label="candidate report.dataset")
    if (
        meta.get("seed") != config["seed"]
        or report_dataset.get("dataset_name") != dataset["name"]
        or report_dataset.get("revision") != dataset["revision"]
        or report_dataset.get("split") != dataset["split"]
    ):
        raise CleanPruningSelectionEvidenceError(
            "candidate report does not match the immutable pruning selection dataset"
        )
    dataset_hash = _mapping(
        report_dataset.get("hash"), label="candidate report.dataset.hash"
    )
    if dataset_hash.get("source") == "config_fallback":
        raise CleanPruningSelectionEvidenceError(
            "candidate report dataset hash must not use config_fallback"
        )
    dataset_windows = _mapping(
        report_dataset.get("windows"), label="candidate report.dataset.windows"
    )
    if dataset_windows.get("seed") != config["seed"]:
        raise CleanPruningSelectionEvidenceError(
            "candidate report dataset window seed does not match selection config"
        )
    return {
        "schema": CLEAN_PRUNING_EVALUATOR_PROVENANCE_SCHEMA,
        "execution_receipt_sha256": execution_digest,
        "selection_config_sha256": canonical_json_sha256(config),
        "original_model_key": receipt["original_model_key"],
        "candidate_id": receipt["candidate_id"],
        "repeat_index": repeat_index,
        "report_run_id": run_id,
        "pruning": receipt["pruning"],
        "baseline_identity": receipt["baseline_identity"],
        "dataset": config["dataset"],
        "seed": config["seed"],
        "effective_schedule": schedule,
        "ordered_two_arm_schedule_sha256": canonical_json_sha256(ordered),
    }


def build_clean_pruning_candidate_report_binding(
    *,
    report: Mapping[str, object],
    replay: Mapping[str, object],
    runtime: Mapping[str, object],
    original_model_key: str,
    candidate_id: str,
    pruning: Mapping[str, object],
    selection_config: Mapping[str, object],
    execution_receipt: Mapping[str, object],
    execution_receipt_sha256: str,
    repeat_index: int,
) -> dict[str, object]:
    """Derive one report-local binding from pinned real pruning sidecars.

    This is not a general report stamper.  It rejects an unbound replay,
    copied runtime proof, post-hoc execution receipt, non-strict report, or
    caller-supplied quality ranking.  The returned quality loss is derived
    solely from the report's primary metric.
    """

    model_key = _text(original_model_key, label="original_model_key")
    normalized_candidate_id = _text(candidate_id, label="candidate_id")
    if _CANDIDATE_ID_RE.fullmatch(normalized_candidate_id) is None:
        raise CleanPruningSelectionEvidenceError("candidate_id is invalid")
    normalized_pruning = _pruning_spec(pruning, label="candidate.pruning")
    normalized_config = _selection_config(selection_config)
    execution_digest = _sha256(
        execution_receipt_sha256, label="execution_receipt_sha256"
    )
    receipt = validate_clean_pruning_execution_receipt(
        execution_receipt,
        expected_model_key=model_key,
        expected_candidate_id=normalized_candidate_id,
        expected_pruning=normalized_pruning,
        expected_selection_config=normalized_config,
    )
    baseline_identity = _identity(
        replay.get("baseline_identity"), label="pruning replay.baseline_identity"
    )
    artifact_identity = _identity(
        replay.get("artifact_identity"), label="pruning replay.artifact_identity"
    )
    if baseline_identity != receipt["baseline_identity"]:
        raise CleanPruningSelectionEvidenceError(
            "pruning replay baseline identity does not match execution receipt"
        )
    _assert_pruning_replay(
        replay,
        pruning=normalized_pruning,
        baseline_identity=baseline_identity,
        artifact_identity=artifact_identity,
    )
    _assert_runtime_reload_proof(runtime, artifact_identity=artifact_identity)
    _assert_report_native_execution_provenance(
        report,
        execution_receipt_sha256=execution_digest,
        selection_config=normalized_config,
        original_model_key=model_key,
        candidate_id=normalized_candidate_id,
        pruning=normalized_pruning,
        baseline_identity=baseline_identity,
        repeat_index=repeat_index,
    )
    quality_loss = _eligible_report_quality_loss(
        report,
        original_model_key=model_key,
        baseline_identity=baseline_identity,
        artifact_identity=artifact_identity,
    )
    return {
        "schema": CLEAN_PRUNING_REPORT_BINDING_SCHEMA,
        "selection_config_sha256": canonical_json_sha256(normalized_config),
        "execution_receipt_sha256": execution_digest,
        "candidate_id": normalized_candidate_id,
        "original_model_key": model_key,
        "repeat_index": repeat_index,
        "pruning": normalized_pruning,
        "baseline_identity": baseline_identity,
        "artifact_identity": artifact_identity,
        "quality_loss": quality_loss,
    }
