"""Authenticated candidate report, replay, and runtime evidence validation."""

from __future__ import annotations

import math
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from invarlock.clean_selection.common import (
    _ALGORITHMS,
    _CANDIDATE_ID_RE,
    _ELIGIBLE_VALIDATION_FIELDS,
    _RUNTIME_DEVICE_RE,
    _RUNTIME_LOAD_DIAGNOSTIC_FIELDS,
    _RUNTIME_LOAD_DIAGNOSTICS_FIELDS,
    _RUNTIME_RELOAD_PROMPT_SHA256,
    _RUNTIME_RELOAD_PROOF_FIELDS,
    _RUNTIME_STORAGE_KEY_AUDIT_ENVELOPE_FIELDS,
    _RUNTIME_STORAGE_KEY_AUDIT_FIELDS,
    CLEAN_SELECTION_CONTRACT_VERSION,
    EVALUATOR_PROVENANCE_SCHEMA,
    REPORT_SELECTION_BINDING_SCHEMA,
    RUNTIME_LOAD_DIAGNOSTICS_SCHEMA,
    RUNTIME_RELOAD_PROOF_SCHEMA,
    RUNTIME_STORAGE_KEY_AUDIT_SCHEMA,
    SELECTION_EXECUTION_RECEIPT_SCHEMA,
    TRANSFORMATION_CONTRACT_VERSION,
    TRANSFORMATION_PARAMETERS_SCHEMA,
    TRANSFORMATION_REPLAY_SCHEMA,
    TRANSFORMATION_SCOPE_POLICY,
    CleanSelectionEvidenceError,
    _exact_mapping,
    _finite,
    _identity,
    _mapping,
    _positive_int,
    _selection_config,
    _sha256,
    _text,
    _transform,
    canonical_json_sha256,
)
from invarlock.transformation_target_manifest import (
    TransformationTargetManifestError,
    transformation_target_manifest_sha256,
    validate_transformation_target_manifest,
)


def _path_below(root: Path, relative: str, *, label: str) -> Path:
    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise CleanSelectionEvidenceError(
            "selection evidence root is unavailable"
        ) from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise CleanSelectionEvidenceError(
            "selection evidence root must be a regular directory"
        )
    current = root
    parts = relative.split("/")
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise CleanSelectionEvidenceError(f"{label} is missing") from exc
        if stat.S_ISLNK(mode):
            raise CleanSelectionEvidenceError(f"{label} must not traverse a symlink")
        if index < len(parts) - 1:
            if not stat.S_ISDIR(mode):
                raise CleanSelectionEvidenceError(f"{label} has a non-directory parent")
        elif not stat.S_ISREG(mode):
            raise CleanSelectionEvidenceError(f"{label} must be a regular file")
    return current


def _eligible_report_quality_loss(
    report: Mapping[str, object],
    *,
    model_key: str,
    baseline_identity: Mapping[str, str],
    artifact_identity: Mapping[str, str],
) -> float:
    # A few hand-written fields that resemble a strict verdict are not a normal
    # InvarLock evaluation report.  The caller separately verifies the sibling
    # runtime manifest; this schema boundary prevents metadata-only fixtures
    # from becoming an authoritative candidate report.
    from invarlock.reporting.report_schema import validate_report

    if not validate_report(dict(report)):
        raise CleanSelectionEvidenceError(
            "candidate report is not a schema-valid InvarLock evaluation report"
        )
    meta = _mapping(report.get("meta"), label="candidate report.meta")
    if meta.get("model_identity") != dict(artifact_identity):
        raise CleanSelectionEvidenceError("candidate report artifact identity mismatch")
    if meta.get("model_id") != model_key:
        raise CleanSelectionEvidenceError("candidate report model id mismatch")
    baseline_ref = _mapping(
        report.get("baseline_ref"), label="candidate report.baseline_ref"
    )
    if baseline_ref.get("model_identity") != dict(baseline_identity):
        raise CleanSelectionEvidenceError("candidate report baseline identity mismatch")
    assurance = _mapping(report.get("assurance"), label="candidate report.assurance")
    if (
        assurance.get("mode") != "strict"
        or assurance.get("report_local_verdict") != "pass"
        or assurance.get("canonical_guard_chain_enforced") is not True
        or assurance.get("fallback_fields_used") is not False
        or assurance.get("blocking_reasons") != []
    ):
        raise CleanSelectionEvidenceError(
            "candidate report is not an eligible strict assurance pass"
        )
    validation = _mapping(report.get("validation"), label="candidate report.validation")
    if any(validation.get(key) is not True for key in _ELIGIBLE_VALIDATION_FIELDS):
        raise CleanSelectionEvidenceError(
            "candidate report has an ineligible guard result"
        )
    invariants = _mapping(report.get("invariants"), label="candidate report.invariants")
    if invariants.get("passed") is not True or invariants.get("supported") is not True:
        raise CleanSelectionEvidenceError(
            "candidate report invariants are not eligible"
        )
    primary_metric = _mapping(
        report.get("primary_metric"), label="candidate report.primary_metric"
    )
    ratio = _finite(
        primary_metric.get("ratio_vs_baseline"),
        label="candidate report.primary_metric.ratio_vs_baseline",
    )
    if ratio <= 0:
        raise CleanSelectionEvidenceError(
            "candidate report quality ratio must be positive"
        )
    return ratio - 1.0


def _execution_receipt(
    value: object,
    *,
    expected_model_key: str | None = None,
    expected_candidate_id: str | None = None,
    expected_transformation: Mapping[str, object] | None = None,
    expected_baseline_identity: Mapping[str, str] | None = None,
    expected_selection_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate the immutable pre-evaluation candidate execution receipt."""

    payload = _exact_mapping(
        value,
        label="selection execution receipt",
        fields=frozenset(
            {
                "schema",
                "contract_version",
                "original_model_key",
                "candidate_id",
                "transformation",
                "baseline_identity",
                "selection_config",
                "selection_config_sha256",
            }
        ),
    )
    if payload["schema"] != SELECTION_EXECUTION_RECEIPT_SCHEMA:
        raise CleanSelectionEvidenceError(
            "selection execution receipt has an unrecognized schema"
        )
    if payload["contract_version"] != CLEAN_SELECTION_CONTRACT_VERSION:
        raise CleanSelectionEvidenceError(
            "selection execution receipt has an unrecognized contract version"
        )
    model_key = _text(
        payload["original_model_key"],
        label="selection execution receipt.original_model_key",
    )
    candidate_id = _text(
        payload["candidate_id"], label="selection execution receipt.candidate_id"
    )
    if _CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        raise CleanSelectionEvidenceError(
            "selection execution receipt candidate id is invalid"
        )
    transformation = _transform(
        payload["transformation"], label="selection execution receipt.transformation"
    )
    baseline = _identity(
        payload["baseline_identity"],
        label="selection execution receipt.baseline_identity",
    )
    config = _selection_config(payload["selection_config"])
    config_digest = canonical_json_sha256(config)
    if (
        _sha256(
            payload["selection_config_sha256"],
            label="selection execution receipt.selection_config_sha256",
        )
        != config_digest
    ):
        raise CleanSelectionEvidenceError(
            "selection execution receipt selection_config_sha256 mismatch"
        )
    if expected_model_key is not None and model_key != expected_model_key:
        raise CleanSelectionEvidenceError(
            "selection execution receipt model key mismatch"
        )
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise CleanSelectionEvidenceError(
            "selection execution receipt candidate id mismatch"
        )
    if expected_transformation is not None and transformation != dict(
        expected_transformation
    ):
        raise CleanSelectionEvidenceError(
            "selection execution receipt transformation mismatch"
        )
    if expected_baseline_identity is not None and baseline != dict(
        expected_baseline_identity
    ):
        raise CleanSelectionEvidenceError(
            "selection execution receipt baseline identity mismatch"
        )
    if expected_selection_config is not None and config != dict(
        expected_selection_config
    ):
        raise CleanSelectionEvidenceError(
            "selection execution receipt selection config mismatch"
        )
    return {
        "schema": SELECTION_EXECUTION_RECEIPT_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "original_model_key": model_key,
        "candidate_id": candidate_id,
        "transformation": transformation,
        "baseline_identity": baseline,
        "selection_config": config,
        "selection_config_sha256": config_digest,
    }


def build_selection_execution_receipt(
    *,
    original_model_key: str,
    candidate_id: str,
    transformation: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    selection_config: Mapping[str, object],
) -> dict[str, object]:
    """Build the immutable receipt that a candidate runner writes before eval.

    This is deliberately separate from the post-evaluation report binder.  A
    runner must pass this exact receipt into the evaluator, which emits the
    report-native provenance checked below; the binder cannot manufacture it.
    """

    model_key = _text(original_model_key, label="original_model_key")
    normalized_candidate_id = _text(candidate_id, label="candidate_id")
    if _CANDIDATE_ID_RE.fullmatch(normalized_candidate_id) is None:
        raise CleanSelectionEvidenceError("candidate_id is invalid")
    normalized_transform = _transform(transformation, label="candidate transformation")
    normalized_baseline = _identity(baseline_identity, label="baseline_identity")
    normalized_config = _selection_config(selection_config)
    return {
        "schema": SELECTION_EXECUTION_RECEIPT_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "original_model_key": model_key,
        "candidate_id": normalized_candidate_id,
        "transformation": normalized_transform,
        "baseline_identity": normalized_baseline,
        "selection_config": normalized_config,
        "selection_config_sha256": canonical_json_sha256(normalized_config),
    }


def validate_selection_execution_receipt(
    value: object,
    *,
    expected_model_key: str | None = None,
    expected_candidate_id: str | None = None,
    expected_transformation: Mapping[str, object] | None = None,
    expected_baseline_identity: Mapping[str, str] | None = None,
    expected_selection_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate and canonicalize one immutable pre-evaluation receipt.

    Candidate campaign producers and the evaluator use this public boundary
    rather than reaching into the verifier's private parsing helpers.  It does
    not create a receipt and therefore cannot turn an after-the-fact report
    into pre-evaluation evidence.
    """

    return _execution_receipt(
        value,
        expected_model_key=expected_model_key,
        expected_candidate_id=expected_candidate_id,
        expected_transformation=expected_transformation,
        expected_baseline_identity=expected_baseline_identity,
        expected_selection_config=expected_selection_config,
    )


def _ordered_two_arm_schedule(report: Mapping[str, object]) -> dict[str, object]:
    """Extract evaluator-emitted ordered window IDs for both report arms."""

    windows = _mapping(
        report.get("evaluation_windows"), label="candidate report.evaluation_windows"
    )
    result: dict[str, object] = {}
    for arm in ("preview", "final"):
        arm_payload = _mapping(
            windows.get(arm), label=f"candidate report.evaluation_windows.{arm}"
        )
        ids = arm_payload.get("window_ids")
        if not isinstance(ids, list) or not ids:
            raise CleanSelectionEvidenceError(
                f"candidate report.evaluation_windows.{arm}.window_ids must be non-empty"
            )
        for index, item in enumerate(ids):
            if isinstance(item, bool) or not isinstance(item, (int, str)):
                raise CleanSelectionEvidenceError(
                    f"candidate report.evaluation_windows.{arm}.window_ids[{index}] is invalid"
                )
        result[arm] = list(ids)
    return result


def build_evaluator_execution_provenance(
    *,
    report: Mapping[str, object],
    execution_receipt: Mapping[str, object],
    execution_receipt_sha256: str,
    repeat_index: int,
) -> dict[str, object]:
    """Derive evaluator-native clean-selection provenance for one report.

    The evaluator calls this only after it has run the exact candidate and
    produced the ordinary report, but before it writes the report's runtime
    manifest.  Its immutable receipt is supplied before evaluation starts;
    the resulting provenance binds the real run ID and emitted window order
    rather than accepting caller-provided scheduling claims.
    """

    receipt = _execution_receipt(execution_receipt)
    config = cast(Mapping[str, object], receipt["selection_config"])
    schedule = _mapping(config["schedule"], label="selection_config.schedule")
    expected_repeats = _positive_int(
        schedule["evaluation_repeats"],
        label="selection_config.schedule.evaluation_repeats",
    )
    if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
        raise CleanSelectionEvidenceError("repeat_index must be an integer")
    if repeat_index < 0 or repeat_index >= expected_repeats:
        raise CleanSelectionEvidenceError(
            "repeat_index is outside the selection schedule"
        )
    execution_digest = _sha256(
        execution_receipt_sha256, label="execution_receipt_sha256"
    )
    run_id = _text(report.get("run_id"), label="candidate report.run_id")
    ordered_schedule = _ordered_two_arm_schedule(report)
    max_examples = _positive_int(
        schedule["max_examples"], label="selection_config.schedule.max_examples"
    )
    for arm in ("preview", "final"):
        if len(cast(list[object], ordered_schedule[arm])) != max_examples:
            raise CleanSelectionEvidenceError(
                "candidate report ordered schedule does not contain max_examples per arm"
            )
    return {
        "schema": EVALUATOR_PROVENANCE_SCHEMA,
        "execution_receipt_sha256": execution_digest,
        "selection_config_sha256": canonical_json_sha256(config),
        "original_model_key": receipt["original_model_key"],
        "candidate_id": receipt["candidate_id"],
        "repeat_index": repeat_index,
        "report_run_id": run_id,
        "transformation": receipt["transformation"],
        "baseline_identity": receipt["baseline_identity"],
        "dataset": config["dataset"],
        "seed": config["seed"],
        "effective_schedule": dict(schedule),
        "ordered_two_arm_schedule_sha256": canonical_json_sha256(ordered_schedule),
    }


def _assert_report_native_execution_provenance(
    report: Mapping[str, object],
    *,
    execution_receipt_sha256: str,
    selection_config: Mapping[str, object],
    original_model_key: str,
    candidate_id: str,
    transformation: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    repeat_index: int,
) -> str:
    """Require evaluator-native dataset/schedule provenance, not binder output."""

    config = _selection_config(selection_config)
    schedule = _mapping(config["schedule"], label="selection_config.schedule")
    dataset_identity = _mapping(config["dataset"], label="selection_config.dataset")
    provenance = _mapping(report.get("provenance"), label="candidate report.provenance")
    native = _exact_mapping(
        provenance.get("clean_selection_execution"),
        label="candidate report.provenance.clean_selection_execution",
        fields=frozenset(
            {
                "schema",
                "execution_receipt_sha256",
                "selection_config_sha256",
                "original_model_key",
                "candidate_id",
                "repeat_index",
                "report_run_id",
                "transformation",
                "baseline_identity",
                "dataset",
                "seed",
                "effective_schedule",
                "ordered_two_arm_schedule_sha256",
            }
        ),
    )
    if native["schema"] != EVALUATOR_PROVENANCE_SCHEMA:
        raise CleanSelectionEvidenceError(
            "candidate report evaluator provenance schema is invalid"
        )
    config_digest = canonical_json_sha256(config)
    if native["execution_receipt_sha256"] != execution_receipt_sha256:
        raise CleanSelectionEvidenceError(
            "candidate report evaluator provenance execution receipt mismatch"
        )
    if native["selection_config_sha256"] != config_digest:
        raise CleanSelectionEvidenceError(
            "candidate report evaluator provenance config digest mismatch"
        )
    if (
        native["original_model_key"] != original_model_key
        or native["candidate_id"] != candidate_id
        or native["transformation"] != dict(transformation)
        or native["baseline_identity"] != dict(baseline_identity)
    ):
        raise CleanSelectionEvidenceError(
            "candidate report evaluator provenance identity mismatch"
        )
    if (
        isinstance(native["repeat_index"], bool)
        or not isinstance(native["repeat_index"], int)
        or native["repeat_index"] != repeat_index
    ):
        raise CleanSelectionEvidenceError(
            "candidate report evaluator provenance repeat index mismatch"
        )
    if native["report_run_id"] != report.get("run_id"):
        raise CleanSelectionEvidenceError(
            "candidate report evaluator provenance run id mismatch"
        )
    if native["dataset"] != dict(dataset_identity):
        raise CleanSelectionEvidenceError(
            "candidate report evaluator provenance dataset identity mismatch"
        )
    if native["seed"] != config["seed"] or native["effective_schedule"] != dict(
        schedule
    ):
        raise CleanSelectionEvidenceError(
            "candidate report evaluator provenance effective schedule mismatch"
        )
    meta = _mapping(report.get("meta"), label="candidate report.meta")
    if meta.get("seed") != config["seed"]:
        raise CleanSelectionEvidenceError(
            "candidate report seed does not match selection config"
        )
    report_dataset = _mapping(report.get("dataset"), label="candidate report.dataset")
    if (
        report_dataset.get("dataset_name") != dataset_identity["name"]
        or report_dataset.get("revision") != dataset_identity["revision"]
        or report_dataset.get("split") != dataset_identity["split"]
    ):
        raise CleanSelectionEvidenceError(
            "candidate report dataset fields do not match the immutable selection dataset"
        )
    dataset_hash = _mapping(
        report_dataset.get("hash"), label="candidate report.dataset.hash"
    )
    if dataset_hash.get("source") == "config_fallback":
        raise CleanSelectionEvidenceError(
            "candidate report dataset hash must not use config_fallback"
        )
    window_seed = _mapping(
        report_dataset.get("windows"), label="candidate report.dataset.windows"
    ).get("seed")
    if window_seed != config["seed"]:
        raise CleanSelectionEvidenceError(
            "candidate report dataset window seed does not match selection config"
        )
    ordered_schedule = _ordered_two_arm_schedule(report)
    max_examples = _positive_int(
        schedule["max_examples"], label="selection_config.schedule.max_examples"
    )
    for arm in ("preview", "final"):
        ids = cast(list[object], ordered_schedule[arm])
        if len(ids) != max_examples:
            raise CleanSelectionEvidenceError(
                "candidate report ordered schedule does not contain max_examples per arm"
            )
    schedule_digest = canonical_json_sha256(ordered_schedule)
    if native["ordered_two_arm_schedule_sha256"] != schedule_digest:
        raise CleanSelectionEvidenceError(
            "candidate report evaluator provenance ordered schedule digest mismatch"
        )
    return schedule_digest


def _assert_eligible_report(
    report: Mapping[str, object],
    *,
    model_key: str,
    candidate_id: str,
    transformation: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    artifact_identity: Mapping[str, str],
    selection_config_sha256: str,
    execution_receipt_sha256: str,
    selection_config: Mapping[str, object],
    repeat_index: int,
) -> float:
    quality_loss = _eligible_report_quality_loss(
        report,
        model_key=model_key,
        baseline_identity=baseline_identity,
        artifact_identity=artifact_identity,
    )
    binding = _exact_mapping(
        report.get("clean_selection"),
        label="candidate report.clean_selection",
        fields=frozenset(
            {
                "schema",
                "selection_config_sha256",
                "execution_receipt_sha256",
                "candidate_id",
                "original_model_key",
                "repeat_index",
                "transformation",
                "baseline_identity",
                "artifact_identity",
                "quality_loss",
            }
        ),
    )
    if binding["schema"] != REPORT_SELECTION_BINDING_SCHEMA:
        raise CleanSelectionEvidenceError(
            "candidate report selection binding schema is invalid"
        )
    if binding["selection_config_sha256"] != selection_config_sha256:
        raise CleanSelectionEvidenceError(
            "candidate report selection config digest mismatch"
        )
    if binding["execution_receipt_sha256"] != execution_receipt_sha256:
        raise CleanSelectionEvidenceError(
            "candidate report selection execution receipt digest mismatch"
        )
    if (
        binding["candidate_id"] != candidate_id
        or binding["original_model_key"] != model_key
        or binding["repeat_index"] != repeat_index
    ):
        raise CleanSelectionEvidenceError(
            "candidate report selection identity mismatch"
        )
    if binding["transformation"] != dict(transformation):
        raise CleanSelectionEvidenceError(
            "candidate report transformation binding mismatch"
        )
    if binding["baseline_identity"] != dict(baseline_identity) or binding[
        "artifact_identity"
    ] != dict(artifact_identity):
        raise CleanSelectionEvidenceError(
            "candidate report selection artifact binding mismatch"
        )
    if not math.isclose(
        _finite(
            binding["quality_loss"],
            label="candidate report clean_selection.quality_loss",
        ),
        quality_loss,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise CleanSelectionEvidenceError(
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
    model_key: str,
    candidate_id: str,
    transformation: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    repeat_index: int,
) -> None:
    """Authenticate a normal evaluator report against its strict runtime manifest."""

    from invarlock.runtime_verify import verify_runtime_manifest_snapshot

    report_path = Path(cast(str, report_reference["path"]))
    manifest_path = Path(cast(str, manifest_reference["path"]))
    result = verify_runtime_manifest_snapshot(
        report_bytes,
        dict(manifest),
        report=report_path,
        manifest=manifest_path,
        require_strict_runtime=True,
    )
    if not result.ok:
        detail = "; ".join(result.errors) or "unknown runtime-manifest failure"
        raise CleanSelectionEvidenceError(
            f"candidate report runtime manifest is not an eligible strict binding: {detail}"
        )
    runtime = _mapping(
        manifest.get("runtime"), label="candidate runtime manifest.runtime"
    )
    if runtime.get("allow_network") is not False:
        raise CleanSelectionEvidenceError(
            "candidate report runtime manifest must record allow_network=false"
        )
    context = _mapping(
        manifest.get("context"), label="candidate runtime manifest.context"
    )
    link = _exact_mapping(
        context.get("clean_selection_execution"),
        label="candidate runtime manifest.context.clean_selection_execution",
        fields=frozenset(
            {
                "execution_receipt_sha256",
                "selection_config_sha256",
                "original_model_key",
                "candidate_id",
                "repeat_index",
                "report_run_id",
                "transformation",
                "baseline_identity",
            }
        ),
    )
    if (
        link["execution_receipt_sha256"] != execution_receipt_sha256
        or link["selection_config_sha256"] != selection_config_sha256
        or link["original_model_key"] != model_key
        or link["candidate_id"] != candidate_id
        or link["repeat_index"] != repeat_index
        or link["report_run_id"] != report.get("run_id")
        or link["transformation"] != dict(transformation)
        or link["baseline_identity"] != dict(baseline_identity)
    ):
        raise CleanSelectionEvidenceError(
            "candidate runtime manifest clean-selection execution linkage mismatch"
        )


def _assert_candidate_target_manifest(
    replay: Mapping[str, object], *, transformation: Mapping[str, object]
) -> None:
    """Require candidate replay to carry a semantically valid v1 target list.

    Selection evidence can be retained independently of its final evidence
    pack, so it must not accept a self-consistent replay that points a text
    transformation at a visual, audio, MTP, or otherwise unsupported tensor.
    """

    raw_manifest = replay.get("target_manifest")
    if raw_manifest is None:
        raise CleanSelectionEvidenceError("candidate replay target_manifest is missing")
    try:
        manifest = validate_transformation_target_manifest(raw_manifest)
    except TransformationTargetManifestError as exc:
        raise CleanSelectionEvidenceError(
            "candidate replay target_manifest is invalid: " + str(exc)
        ) from exc
    digest = replay.get("target_manifest_sha256")
    if not isinstance(digest, str) or digest != transformation_target_manifest_sha256(
        manifest
    ):
        raise CleanSelectionEvidenceError(
            "candidate replay target_manifest_sha256 mismatch"
        )
    expected = {
        "edit_type": transformation["edit_type"],
        "algorithm": _ALGORITHMS[cast(str, transformation["edit_type"])],
        "parameters": transformation["parameters"],
        "scope": transformation["scope"],
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
    }
    for field, value in expected.items():
        if manifest.get(field) != value or replay.get(field) != value:
            raise CleanSelectionEvidenceError(
                f"candidate replay target_manifest {field} mismatch"
            )
    for field in ("model_type", "architecture", "config_sha256", "layer_count"):
        if replay.get(field) != manifest.get(field):
            raise CleanSelectionEvidenceError(
                f"candidate replay target_manifest {field} mismatch"
            )


def _assert_clean_load_diagnostics(value: object) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != _RUNTIME_LOAD_DIAGNOSTICS_FIELDS
        or value.get("schema") != RUNTIME_LOAD_DIAGNOSTICS_SCHEMA
    ):
        raise CleanSelectionEvidenceError(
            "candidate runtime load diagnostics are invalid"
        )
    reloads = value.get("reloads")
    if not isinstance(reloads, list) or len(reloads) != 2:
        raise CleanSelectionEvidenceError(
            "candidate runtime load diagnostics must bind exactly two reloads"
        )
    for index, diagnostic in enumerate(reloads):
        if (
            not isinstance(diagnostic, Mapping)
            or set(diagnostic) != _RUNTIME_LOAD_DIAGNOSTIC_FIELDS
        ):
            raise CleanSelectionEvidenceError(
                f"candidate runtime load diagnostics reload {index} is invalid"
            )
        for field in _RUNTIME_LOAD_DIAGNOSTIC_FIELDS:
            entries = diagnostic.get(field)
            if not isinstance(entries, list) or entries:
                raise CleanSelectionEvidenceError(
                    f"candidate runtime load diagnostics reload {index} reports {field}"
                )


def _assert_clean_storage_key_audit(value: object) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != _RUNTIME_STORAGE_KEY_AUDIT_ENVELOPE_FIELDS
        or value.get("schema") != RUNTIME_STORAGE_KEY_AUDIT_SCHEMA
    ):
        raise CleanSelectionEvidenceError(
            "candidate runtime storage-key audit is invalid"
        )
    reloads = value.get("reloads")
    if not isinstance(reloads, list) or len(reloads) != 2:
        raise CleanSelectionEvidenceError(
            "candidate runtime storage-key audit must bind exactly two reloads"
        )
    expected: dict[str, object] | None = None
    for index, audit in enumerate(reloads):
        if (
            not isinstance(audit, Mapping)
            or set(audit) != _RUNTIME_STORAGE_KEY_AUDIT_FIELDS
        ):
            raise CleanSelectionEvidenceError(
                f"candidate runtime storage-key audit reload {index} is invalid"
            )
        artifact_storage_key_count = _positive_int(
            audit.get("artifact_storage_key_count"),
            label=(
                "candidate runtime storage-key audit "
                f"reload {index} artifact_storage_key_count"
            ),
        )
        model_state_key_count = _positive_int(
            audit.get("model_state_key_count"),
            label=(
                "candidate runtime storage-key audit "
                f"reload {index} model_state_key_count"
            ),
        )
        if artifact_storage_key_count > model_state_key_count:
            raise CleanSelectionEvidenceError(
                "candidate runtime storage-key audit "
                f"reload {index} has more artifact storage keys than model state keys"
            )
        for field in (
            "artifact_storage_keys_sha256",
            "model_state_keys_sha256",
        ):
            _sha256(
                audit.get(field),
                label=f"candidate runtime storage-key audit reload {index} {field}",
            )
        if audit.get("unexpected_storage_keys") != []:
            raise CleanSelectionEvidenceError(
                f"candidate runtime storage-key audit reload {index} has unexpected storage keys"
            )
        normalized = dict(audit)
        if expected is None:
            expected = normalized
        elif normalized != expected:
            raise CleanSelectionEvidenceError(
                "candidate runtime storage-key audits disagree across reloads"
            )


def _assert_candidate_replay_runtime(
    replay: Mapping[str, object],
    runtime: Mapping[str, object],
    *,
    transformation: Mapping[str, object],
    baseline_identity: Mapping[str, str],
    artifact_identity: Mapping[str, str],
) -> None:
    """Validate the actual replay/runtime pair before report binding or use."""

    edit_type = cast(str, transformation["edit_type"])
    expected_spec = {
        "schema": TRANSFORMATION_PARAMETERS_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "edit_type": edit_type,
        "algorithm": _ALGORITHMS[edit_type],
        "parameters": transformation["parameters"],
    }
    if (
        replay.get("schema") != TRANSFORMATION_REPLAY_SCHEMA
        or replay.get("ok") is not True
        or replay.get("issues") != []
        or replay.get("edit_type") != edit_type
        or replay.get("transformation") != expected_spec
        or replay.get("parameters") != transformation["parameters"]
        or replay.get("scope") != transformation["scope"]
        or replay.get("algorithm") != _ALGORITHMS[edit_type]
        or replay.get("baseline_identity") != dict(baseline_identity)
        or replay.get("artifact_identity") != dict(artifact_identity)
    ):
        raise CleanSelectionEvidenceError(
            "candidate replay sidecar does not bind its selected transformation"
        )

    _assert_candidate_target_manifest(replay, transformation=transformation)

    def _runtime_shape(value: object) -> bool:
        return (
            isinstance(value, list)
            and bool(value)
            and all(
                isinstance(item, int) and not isinstance(item, bool) and item > 0
                for item in value
            )
        )

    if set(runtime) != _RUNTIME_RELOAD_PROOF_FIELDS:
        raise CleanSelectionEvidenceError(
            "candidate runtime sidecar has unbound or missing fields"
        )
    for field in ("prompt_sha256", "token_ids_sha256", "logits_sha256"):
        _sha256(runtime.get(field), label=f"candidate runtime.{field}")
    if (
        runtime.get("schema") != RUNTIME_RELOAD_PROOF_SCHEMA
        or runtime.get("ok") is not True
        or runtime.get("replay_schema") != TRANSFORMATION_REPLAY_SCHEMA
        or runtime.get("edit_type") != edit_type
        or runtime.get("artifact_identity") != dict(artifact_identity)
        or runtime.get("replay_artifact_identity") != dict(artifact_identity)
        or runtime.get("all_logits_finite") is not True
        or runtime.get("repeat_deterministic") is not True
        or runtime.get("prompt_sha256") != _RUNTIME_RELOAD_PROMPT_SHA256
        or not isinstance(runtime.get("device"), str)
        or _RUNTIME_DEVICE_RE.fullmatch(cast(str, runtime.get("device"))) is None
        or not isinstance(runtime.get("input_device"), str)
        or _RUNTIME_DEVICE_RE.fullmatch(cast(str, runtime.get("input_device"))) is None
        or runtime.get("reload_runs") != 2
        or not _runtime_shape(runtime.get("token_ids_shape"))
        or not _runtime_shape(runtime.get("logits_shape"))
    ):
        raise CleanSelectionEvidenceError(
            "candidate runtime sidecar is not an eligible two-reload proof"
        )
    _assert_clean_load_diagnostics(runtime.get("load_diagnostics"))
    _assert_clean_storage_key_audit(runtime.get("storage_key_audit"))


def validate_candidate_replay_runtime(
    *,
    replay: Mapping[str, object],
    runtime: Mapping[str, object],
    transformation: Mapping[str, object],
    baseline_identity: Mapping[str, str],
) -> dict[str, str]:
    """Validate a candidate's replay/runtime pair before evaluation starts.

    This is the evaluator-facing preflight counterpart to the final bundle
    verifier.  It prevents an expensive candidate evaluation from starting
    with absent, truncated, or identity-mismatched transformation proof.
    """

    normalized_transform = _transform(transformation, label="candidate transformation")
    normalized_baseline = _identity(
        baseline_identity, label="candidate baseline identity"
    )
    artifact_identity = _identity(
        replay.get("artifact_identity"), label="candidate replay.artifact_identity"
    )
    _assert_candidate_replay_runtime(
        replay,
        runtime,
        transformation=normalized_transform,
        baseline_identity=normalized_baseline,
        artifact_identity=artifact_identity,
    )
    return artifact_identity
