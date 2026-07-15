"""Magnitude-pruning replay and clean-selection validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from invarlock.clean_pruning_selection_common import CleanPruningSelectionEvidenceError
from invarlock.clean_pruning_selection_contracts.snapshot import (
    verify_clean_pruning_selection_snapshot_tree,
)
from invarlock.evidence_pack_edit_common import (
    _SHA256_RE,
    PRUNING_REPLAY_SIDECAR,
    RUNTIME_RELOAD_PROOF_SIDECAR,
    _expected_literal_pruning_params,
    _finite_number,
    _is_nonnegative_int,
    _same_finite_number,
    _sanitize_model_key,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    read_json_object_snapshot,
    sha256_prefixed,
)
from invarlock.pruning_contract import (
    PRUNING_ALGORITHM,
    PRUNING_REPLAY_SCHEMA,
    PRUNING_STORAGE_POLICY,
    PRUNING_TARGET_MANIFEST_SCHEMA,
    PruningContractError,
    pruning_target_manifest_sha256,
    validate_pruning_target_manifest,
)
from invarlock.pruning_contract import (
    PRUNING_SCOPE_POLICY_VERSION as PRUNING_SCOPE_POLICY,
)


def _is_clean_pruning_scenario(spec: dict[str, Any] | None) -> bool:
    if not isinstance(spec, dict):
        return False
    generation = spec.get("generation")
    edit_spec = generation.get("edit_spec") if isinstance(generation, dict) else ""
    if not isinstance(edit_spec, str):
        return False
    parts = edit_spec.split(":")
    return len(parts) >= 2 and parts[0] == "magnitude_prune" and parts[1] == "clean"


def _pruning_identity_errors(
    *,
    prefix: str,
    label: str,
    value: object,
) -> list[str]:
    if not isinstance(value, dict):
        return [prefix + f"pruning replay {label} must be an object"]
    kind = value.get("kind")
    digest = value.get("sha256")
    if not isinstance(kind, str) or not kind:
        return [prefix + f"pruning replay {label}.kind must be a non-empty string"]
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        return [prefix + f"pruning replay {label}.sha256 must be a sha256 digest"]
    return []


def _pruning_target_manifest_errors(
    *,
    prefix: str,
    payload: dict[str, Any],
) -> list[str]:
    """Validate the canonical target list instead of trusting a bare digest."""

    errors: list[str] = []
    manifest = payload.get("target_manifest")
    if not isinstance(manifest, dict):
        return [prefix + "pruning replay target_manifest must be an object"]
    digest = payload.get("target_manifest_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        errors.append(
            prefix + "pruning replay target_manifest_sha256 must be a sha256 digest"
        )
    try:
        canonical_manifest = validate_pruning_target_manifest(
            manifest,
            expected_scope=payload.get("scope")
            if isinstance(payload.get("scope"), str)
            else None,
        )
        canonical_digest = pruning_target_manifest_sha256(canonical_manifest)
    except PruningContractError as exc:
        errors.append(
            prefix + f"pruning replay target_manifest policy violation: {exc}"
        )
    else:
        if digest != canonical_digest:
            errors.append(prefix + "pruning replay target_manifest digest mismatch")

    expected_manifest = {
        "schema": PRUNING_TARGET_MANIFEST_SCHEMA,
        "scope": payload.get("scope"),
        "scope_policy": payload.get("scope_policy"),
        "pruning_algorithm": payload.get("pruning_algorithm"),
        "storage_policy": payload.get("storage_policy"),
        "model_type": payload.get("model_type"),
        "architecture": payload.get("architecture"),
        "config_sha256": payload.get("config_sha256"),
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            errors.append(prefix + f"pruning replay target_manifest {field} mismatch")

    targets = manifest.get("targets")
    if not isinstance(targets, list) or not targets:
        errors.append(
            prefix + "pruning replay target_manifest.targets must be a non-empty list"
        )
        return errors
    names: set[str] = set()
    selected_params = 0
    for index, target in enumerate(targets):
        path = f"pruning replay target_manifest.targets[{index}]"
        if not isinstance(target, dict):
            errors.append(prefix + path + " must be an object")
            continue
        name = target.get("name")
        dtype = target.get("dtype")
        shape = target.get("shape")
        numel = target.get("numel")
        if not isinstance(name, str) or not name:
            errors.append(prefix + path + ".name must be a non-empty string")
        elif name in names:
            errors.append(prefix + path + ".name is duplicated")
        else:
            names.add(name)
        if not isinstance(dtype, str) or not dtype:
            errors.append(prefix + path + ".dtype must be a non-empty string")
        if (
            not isinstance(shape, list)
            or not shape
            or not all(_is_nonnegative_int(dimension) for dimension in shape)
        ):
            errors.append(prefix + path + ".shape must be a non-empty integer list")
        if not isinstance(numel, int) or isinstance(numel, bool) or numel <= 0:
            errors.append(prefix + path + ".numel must be a positive int")
            continue
        if (
            isinstance(shape, list)
            and shape
            and all(_is_nonnegative_int(dimension) for dimension in shape)
        ):
            expected_numel = 1
            for dimension in shape:
                expected_numel *= int(dimension)
            if numel != expected_numel:
                errors.append(prefix + path + ".numel does not match shape")
        selected_params += numel

    if payload.get("selected_tensors") != len(targets):
        errors.append(
            prefix + "pruning replay selected_tensors does not match target manifest"
        )
    if payload.get("selected_params") != selected_params:
        errors.append(
            prefix + "pruning replay selected_params does not match target manifest"
        )
    return errors


def _pruning_contract_errors(
    *,
    prefix: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    spec: dict[str, Any] | None,
) -> tuple[float | None, list[str]]:
    errors: list[str] = []
    if payload.get("schema") != PRUNING_REPLAY_SCHEMA:
        errors.append(prefix + "pruning_replay.json has unrecognized schema")
    if payload.get("ok") is not True:
        errors.append(prefix + "pruning replay proof did not pass")
    if payload.get("edit_type") != "magnitude_prune":
        errors.append(prefix + "pruning replay edit_type mismatch")

    parameters = metadata.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    metadata_sparsity_value = _finite_number(parameters.get("target_sparsity"))
    proof_sparsity = _finite_number(payload.get("target_sparsity"))
    if proof_sparsity is None:
        errors.append(prefix + "pruning replay target_sparsity missing")
    elif not 0.0 < proof_sparsity < 1.0:
        errors.append(prefix + "pruning replay target_sparsity must be in (0, 1)")
    elif (
        metadata_sparsity_value is None
        or abs(metadata_sparsity_value - proof_sparsity) > 1e-12
    ):
        errors.append(prefix + "pruning replay target_sparsity metadata mismatch")
    scope = payload.get("scope")
    if not isinstance(scope, str) or not scope:
        errors.append(prefix + "pruning replay scope must be a non-empty string")
    if scope != metadata.get("scope"):
        errors.append(prefix + "pruning replay scope metadata mismatch")

    contract_values = {
        "scope_policy": PRUNING_SCOPE_POLICY,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
    }
    for field, expected in contract_values.items():
        if payload.get(field) != expected:
            errors.append(prefix + f"pruning replay {field} mismatch")
        if metadata.get(field) != payload.get(field):
            errors.append(prefix + f"pruning replay {field} metadata mismatch")
    for field, metadata_field in (
        ("model_type", "model_type"),
        ("architecture", "pruning_architecture"),
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            errors.append(prefix + f"pruning replay {field} must be a non-empty string")
        if metadata.get(metadata_field) != value:
            errors.append(prefix + f"pruning replay {field} metadata mismatch")
    config_sha256 = payload.get("config_sha256")
    if (
        not isinstance(config_sha256, str)
        or _SHA256_RE.fullmatch(config_sha256) is None
    ):
        errors.append(prefix + "pruning replay config_sha256 must be a sha256 digest")
    if metadata.get("config_sha256") != config_sha256:
        errors.append(prefix + "pruning replay config_sha256 metadata mismatch")

    if spec is not None:
        expected_sparsity, expected_scope, pruning_error = (
            _expected_literal_pruning_params(spec)
        )
        if pruning_error is not None:
            errors.append(prefix + pruning_error)
        if expected_scope is not None and scope != expected_scope:
            errors.append(prefix + "pruning replay scope scenario mismatch")
        if expected_sparsity is not None and (
            proof_sparsity is None
            or abs(float(proof_sparsity) - expected_sparsity) > 1e-12
        ):
            errors.append(prefix + "pruning replay target_sparsity scenario mismatch")
    return proof_sparsity, errors


def _pruning_coverage_errors(
    *,
    prefix: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    proof_sparsity: float | None,
) -> list[str]:
    errors: list[str] = []
    coverage_fields = (
        "checked_tensors",
        "selected_tensors",
        "selected_params",
        "total_params",
        "expected_pruned_params",
        "expected_changed_params",
        "observed_changed_params",
        "original_zero_params",
        "observed_zero_params",
        "out_of_scope_tensors_checked",
        "out_of_scope_bytes_checked",
        "support_files_checked",
    )
    for field in coverage_fields:
        if not _is_nonnegative_int(payload.get(field)):
            errors.append(prefix + f"pruning replay {field} must be a non-negative int")
    checked_tensors = payload.get("checked_tensors")
    selected_tensors = payload.get("selected_tensors")
    out_of_scope_tensors = payload.get("out_of_scope_tensors_checked")
    if selected_tensors == 0 or payload.get("selected_params") == 0:
        errors.append(prefix + "pruning replay selected no tensors")
    if (
        _is_nonnegative_int(checked_tensors)
        and _is_nonnegative_int(selected_tensors)
        and _is_nonnegative_int(out_of_scope_tensors)
        and cast(int, checked_tensors)
        != cast(int, selected_tensors) + cast(int, out_of_scope_tensors)
    ):
        errors.append(prefix + "pruning replay tensor coverage count mismatch")
    if (
        proof_sparsity is not None
        and payload.get("expected_pruned_params") == 0
        and proof_sparsity > 0
    ):
        errors.append(prefix + "pruning replay expected no pruned parameters")
    if payload.get("expected_changed_params") == 0:
        errors.append(prefix + "pruning replay made no effective parameter changes")
    if payload.get("observed_changed_params") != payload.get("expected_changed_params"):
        errors.append(prefix + "pruning replay changed parameter count mismatch")
    expected_pruned_params = payload.get("expected_pruned_params")
    selected_params = payload.get("selected_params")
    total_params = payload.get("total_params")
    if total_params == 0:
        errors.append(prefix + "pruning replay total parameter count must be positive")
    if (
        isinstance(total_params, int)
        and not isinstance(total_params, bool)
        and isinstance(selected_params, int)
        and not isinstance(selected_params, bool)
        and selected_params > total_params
    ):
        errors.append(prefix + "pruning replay total parameter count invalid")
    if (
        isinstance(expected_pruned_params, int)
        and not isinstance(expected_pruned_params, bool)
        and isinstance(selected_params, int)
        and not isinstance(selected_params, bool)
        and expected_pruned_params > selected_params
    ):
        errors.append(prefix + "pruning replay expected pruned parameter count invalid")
    expected_changed_params = payload.get("expected_changed_params")
    if (
        isinstance(expected_changed_params, int)
        and not isinstance(expected_changed_params, bool)
        and isinstance(expected_pruned_params, int)
        and not isinstance(expected_pruned_params, bool)
        and expected_changed_params > expected_pruned_params
    ):
        errors.append(
            prefix + "pruning replay expected changed parameter count invalid"
        )
    if payload.get("support_files_checked") == 0:
        errors.append(prefix + "pruning replay checked no support files")

    coverage = metadata.get("coverage")
    if not isinstance(coverage, dict):
        errors.append(prefix + "pruning replay metadata coverage must be an object")
    else:
        for field in ("edited_tensors", "edited_params", "total_params"):
            if not _is_nonnegative_int(coverage.get(field)):
                errors.append(
                    prefix
                    + f"pruning replay metadata coverage.{field} must be a non-negative int"
                )
        if payload.get("selected_tensors") != coverage.get("edited_tensors"):
            errors.append(prefix + "pruning replay edited tensor count mismatch")
        if selected_params != coverage.get("edited_params"):
            errors.append(prefix + "pruning replay edited parameter count mismatch")
        if total_params != coverage.get("total_params"):
            errors.append(
                prefix + "pruning replay metadata total parameter count mismatch"
            )
        coverage_ratio = _finite_number(coverage.get("coverage_ratio"))
        expected_ratio = (
            selected_params / total_params
            if isinstance(selected_params, int)
            and not isinstance(selected_params, bool)
            and isinstance(total_params, int)
            and not isinstance(total_params, bool)
            and total_params > 0
            else None
        )
        if coverage_ratio is None or coverage_ratio != expected_ratio:
            errors.append(
                prefix + "pruning replay metadata coverage.coverage_ratio mismatch"
            )

    errors.extend(_pruning_target_manifest_errors(prefix=prefix, payload=payload))
    if metadata.get("target_manifest") != payload.get("target_manifest"):
        errors.append(prefix + "pruning replay target_manifest metadata mismatch")
    if metadata.get("target_manifest_sha256") != payload.get("target_manifest_sha256"):
        errors.append(
            prefix + "pruning replay target_manifest digest metadata mismatch"
        )
    if metadata.get("effective_changed_params") != payload.get(
        "observed_changed_params"
    ):
        errors.append(
            prefix + "pruning replay effective_changed_params metadata mismatch"
        )
    return errors


def _pruning_report_identity_errors(
    *, prefix: str, report: dict[str, Any], payload: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    report_identity = (
        (report.get("meta") or {}).get("model_identity")
        if isinstance(report.get("meta"), dict)
        else None
    )
    baseline_ref = report.get("baseline_ref")
    report_baseline_identity = (
        baseline_ref.get("model_identity") if isinstance(baseline_ref, dict) else None
    )
    if not isinstance(report_identity, dict):
        errors.append(prefix + "evaluation subject identity missing")
    else:
        errors.extend(
            _pruning_identity_errors(
                prefix=prefix, label="artifact_identity", value=report_identity
            )
        )
        if payload.get("artifact_identity") != report_identity:
            errors.append(prefix + "pruning replay artifact identity mismatch")
    errors.extend(
        _pruning_identity_errors(
            prefix=prefix,
            label="artifact_identity",
            value=payload.get("artifact_identity"),
        )
    )
    if not isinstance(report_baseline_identity, dict):
        errors.append(prefix + "evaluation baseline identity missing")
    else:
        errors.extend(
            _pruning_identity_errors(
                prefix=prefix,
                label="baseline_identity",
                value=report_baseline_identity,
            )
        )
        if payload.get("baseline_identity") != report_baseline_identity:
            errors.append(prefix + "pruning replay baseline identity mismatch")
    errors.extend(
        _pruning_identity_errors(
            prefix=prefix,
            label="baseline_identity",
            value=payload.get("baseline_identity"),
        )
    )
    issues = payload.get("issues")
    if not isinstance(issues, list):
        errors.append(prefix + "pruning replay issues must be a list")
    elif issues:
        errors.append(prefix + "pruning replay issues must be empty when ok")
    return errors


def _pruning_replay_errors(
    *,
    scenario_id: str,
    report: dict[str, Any],
    metadata: dict[str, Any],
    payload: dict[str, Any],
    spec: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    prefix = f"{scenario_id}: "
    proof_sparsity, contract_errors = _pruning_contract_errors(
        prefix=prefix, payload=payload, metadata=metadata, spec=spec
    )
    errors.extend(contract_errors)

    errors.extend(
        _pruning_coverage_errors(
            prefix=prefix,
            payload=payload,
            metadata=metadata,
            proof_sparsity=proof_sparsity,
        )
    )

    errors.extend(
        _pruning_report_identity_errors(prefix=prefix, report=report, payload=payload)
    )
    return errors


def _clean_pruning_selection_errors(
    *,
    pack_dir: Path,
    scenario_id: str,
    report_path: Path,
    report_dir: Path,
    report_model_name: str | None,
    report: dict[str, Any],
    metadata: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    """Bind a final clean-pruning lane to one selected v1 campaign result.

    Unlike generated transformations, pruning does not attach a mutable
    selection receipt to its replay.  The pack owns an immutable snapshot of
    the complete candidate bundle instead, and the final lane must be the
    byte-identical selected candidate report plus its exact replay/artifact
    identity.  This closes the old v1 ``selected_by_*`` preset loophole.
    """

    prefix = f"{scenario_id}: "
    errors: list[str] = []
    if report_model_name is None:
        return [prefix + "clean pruning report has no model path"]
    legacy_fields = {"selection_receipt", "selection_receipt_sha256"} & set(payload)
    if legacy_fields:
        return [
            prefix
            + "clean pruning replay must not carry retired v1 selection fields: "
            + ", ".join(sorted(legacy_fields))
        ]
    selection_root = pack_dir / "metadata" / "clean_pruning_selection"
    try:
        selection_snapshot = verify_clean_pruning_selection_snapshot_tree(
            selection_root
        )
    except CleanPruningSelectionEvidenceError as exc:
        return [
            prefix
            + "clean magnitude-prune v1 selection snapshot is invalid: "
            + str(exc)
        ]
    entries = cast(list[dict[str, object]], selection_snapshot.bundle["entries"])
    matching_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("original_model_key"), str)
        and _sanitize_model_key(cast(str, entry["original_model_key"]))
        == report_model_name
    ]
    if len(matching_entries) != 1:
        return [
            prefix
            + "clean magnitude-prune v1 selection snapshot has no unique matching model entry"
        ]
    entry = matching_entries[0]
    selected = cast(dict[str, object], entry["selected_entry"])
    receipt = cast(dict[str, object], selected["selection_receipt"])
    selected_pruning = cast(dict[str, object], receipt["selected_pruning"])
    selected_evaluation = cast(dict[str, object], receipt["selected_evaluation"])
    model_key = cast(str, receipt["original_model_key"])
    baseline_identity = cast(dict[str, object], receipt["baseline_identity"])
    expected_scope = cast(str, selected_pruning["scope"])
    expected_sparsity = cast(float, _finite_number(selected_pruning["target_sparsity"]))
    if payload.get("scope") != expected_scope:
        errors.append(prefix + "clean magnitude-prune selected scope mismatch")
    if not _same_finite_number(payload.get("target_sparsity"), expected_sparsity):
        errors.append(prefix + "clean magnitude-prune selected sparsity mismatch")
    parameters = metadata.get("parameters")
    if not isinstance(parameters, dict) or not _same_finite_number(
        parameters.get("target_sparsity"), expected_sparsity
    ):
        errors.append(prefix + "clean magnitude-prune metadata sparsity mismatch")
    if metadata.get("scope") != expected_scope:
        errors.append(prefix + "clean magnitude-prune metadata scope mismatch")

    replay_reference = cast(dict[str, object], selected_evaluation["replay"])
    expected_artifact = cast(dict[str, object], replay_reference["artifact_identity"])
    if payload.get("artifact_identity") != expected_artifact:
        errors.append(
            prefix + "clean magnitude-prune selected artifact identity mismatch"
        )
    if payload.get("baseline_identity") != baseline_identity:
        errors.append(
            prefix + "clean magnitude-prune selected baseline identity mismatch"
        )
    report_meta = report.get("meta")
    if not isinstance(report_meta, dict) or report_meta.get("model_id") != model_key:
        errors.append(prefix + "clean magnitude-prune report model identity mismatch")
    if (
        not isinstance(report_meta, dict)
        or report_meta.get("model_identity") != expected_artifact
    ):
        errors.append(
            prefix + "clean magnitude-prune report artifact identity mismatch"
        )
    report_baseline = report.get("baseline_ref")
    if (
        not isinstance(report_baseline, dict)
        or report_baseline.get("model_identity") != baseline_identity
    ):
        errors.append(
            prefix + "clean magnitude-prune report baseline identity mismatch"
        )

    reports = cast(list[dict[str, object]], selected_evaluation["reports"])
    try:
        report_bytes, parsed_report = read_json_object_snapshot(
            report_path, label="clean magnitude-prune final report"
        )
    except (OSError, StrictJsonError) as exc:
        return errors + [
            prefix + "clean magnitude-prune final report is unavailable: " + str(exc)
        ]
    if parsed_report != report:
        errors.append(
            prefix + "clean magnitude-prune final report changed during verification"
        )
    manifest_path = report_dir / "runtime.manifest.json"
    try:
        manifest_bytes, _ = read_json_object_snapshot(
            manifest_path, label="clean magnitude-prune final runtime manifest"
        )
    except (OSError, StrictJsonError) as exc:
        manifest_bytes = b""
        errors.append(
            prefix
            + "clean magnitude-prune runtime manifest is unavailable: "
            + str(exc)
        )
    replay_path = report_dir / PRUNING_REPLAY_SIDECAR
    runtime_path = report_dir / RUNTIME_RELOAD_PROOF_SIDECAR
    try:
        replay_bytes, _ = read_json_object_snapshot(
            replay_path, label="clean magnitude-prune final replay"
        )
    except (OSError, StrictJsonError) as exc:
        replay_bytes = b""
        errors.append(
            prefix + "clean magnitude-prune final replay is unavailable: " + str(exc)
        )
    try:
        runtime_bytes, _ = read_json_object_snapshot(
            runtime_path, label="clean magnitude-prune final runtime proof"
        )
    except (OSError, StrictJsonError) as exc:
        runtime_bytes = b""
        errors.append(
            prefix
            + "clean magnitude-prune final runtime proof is unavailable: "
            + str(exc)
        )
    replay_reference = cast(dict[str, object], selected_evaluation["replay"])
    runtime_reference = cast(dict[str, object], selected_evaluation["runtime"])
    replay_relative = cast(str, replay_reference["path"])
    runtime_relative = cast(str, runtime_reference["path"])
    expected_replay_bytes = selection_snapshot.sidecar_bytes.get(replay_relative)
    expected_runtime_bytes = selection_snapshot.sidecar_bytes.get(runtime_relative)
    if replay_bytes != expected_replay_bytes:
        errors.append(
            prefix
            + "clean magnitude-prune final replay is not the selected candidate replay"
        )
    if runtime_bytes != expected_runtime_bytes:
        errors.append(
            prefix
            + "clean magnitude-prune final runtime proof is not the selected candidate proof"
        )
    matched_report = False
    for report_run in reports:
        report_reference = cast(dict[str, object], report_run["report"])
        manifest_reference = cast(dict[str, object], report_run["runtime_manifest"])
        report_relative = cast(str, report_reference["path"])
        manifest_relative = cast(str, manifest_reference["path"])
        expected_report_bytes = selection_snapshot.sidecar_bytes.get(report_relative)
        expected_manifest_bytes = selection_snapshot.sidecar_bytes.get(
            manifest_relative
        )
        if (
            expected_report_bytes == report_bytes
            and expected_manifest_bytes == manifest_bytes
            and report_reference.get("sha256") == sha256_prefixed(report_bytes)
            and manifest_reference.get("sha256") == sha256_prefixed(manifest_bytes)
        ):
            matched_report = True
            break
    if not matched_report:
        errors.append(
            prefix
            + "clean magnitude-prune final report/runtime manifest is not an exact selected candidate repeat"
        )
    return errors
