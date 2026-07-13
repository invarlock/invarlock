"""Generated-transformation replay and cross-sidecar binding validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from invarlock.evidence_pack_edit_common import (
    _SHA256_RE,
    TRANSFORMATION_MATERIALIZATION_RECEIPT,
    TRANSFORMATION_REPLAY_SCHEMA,
    TRANSFORMATION_SCOPE_POLICY,
    _expected_edit_type,
    _is_nonnegative_int,
    _load_json_sidecar,
    _sha256_file,
)
from invarlock.evidence_pack_transformation_contract import (
    _canonical_transformation_scope,
    _canonical_transformation_spec,
    _expected_literal_transformation,
    _is_clean_transformation_scenario,
    _is_exact_json_value,
    _transformation_identity_errors,
)
from invarlock.evidence_pack_transformation_validation import (
    _clean_transformation_selection_errors,
    _transformation_change_errors,
    _transformation_materialization_receipt_errors,
    _transformation_metadata_errors,
    _transformation_output_weights_errors,
    _transformation_shard_plan_errors,
    _transformation_target_manifest_errors,
)

_REPLAY_BASE_FIELDS = {
    "schema",
    "ok",
    "edit_type",
    "transformation",
    "algorithm",
    "parameters",
    "scope",
    "scope_policy",
    "model_type",
    "architecture",
    "config_sha256",
    "layer_count",
    "target_manifest",
    "target_manifest_sha256",
    "baseline_identity",
    "artifact_identity",
    "materialization_receipt_sha256",
    "edit_metadata_sha256",
    "source_shard_plan",
    "source_shard_plan_sha256",
    "output_shard_plan",
    "output_shard_plan_sha256",
    "max_output_shard_bytes",
    "output_weights",
    "execution_policy",
    "checked_tensors",
    "selected_tensors",
    "selected_params",
    "total_tensors",
    "total_params",
    "actual_changes",
    "out_of_scope_tensors_checked",
    "out_of_scope_bytes_checked",
    "support_files_checked",
    "issues",
}
_REPLAY_SELECTION_FIELDS = {"selection_receipt", "selection_receipt_sha256"}


def _replay_contract_errors(
    *, prefix: str, payload: dict[str, Any], spec: dict[str, Any] | None
) -> tuple[dict[str, object] | None, str | None, list[str]]:
    errors: list[str] = []
    allowed_fields = _REPLAY_BASE_FIELDS | _REPLAY_SELECTION_FIELDS
    if not set(payload) <= allowed_fields:
        errors.append(prefix + "transformation replay has unbound fields")
    if not _REPLAY_BASE_FIELDS <= set(payload):
        errors.append(prefix + "transformation replay has missing required fields")
    if payload.get("schema") != TRANSFORMATION_REPLAY_SCHEMA:
        errors.append(prefix + "transformation_replay.json has unrecognized schema")
    if payload.get("ok") is not True:
        errors.append(prefix + "transformation replay proof did not pass")

    edit_type = payload.get("edit_type")
    transformation, transformation_error = _canonical_transformation_spec(
        edit_type, payload.get("parameters")
    )
    if transformation is None:
        errors.append(
            prefix
            + "transformation replay canonical parameters invalid: "
            + (transformation_error or "unknown error")
        )
    else:
        if not _is_exact_json_value(
            payload.get("parameters"), transformation["parameters"]
        ):
            errors.append(prefix + "transformation replay parameters are not canonical")
        if not _is_exact_json_value(payload.get("transformation"), transformation):
            errors.append(
                prefix + "transformation replay transformation contract mismatch"
            )
        if payload.get("algorithm") != transformation["algorithm"]:
            errors.append(prefix + "transformation replay algorithm mismatch")

    scope, scope_error = _canonical_transformation_scope(payload.get("scope"))
    if scope is None or payload.get("scope") != scope:
        errors.append(
            prefix
            + "transformation replay scope is not canonical: "
            + (scope_error or "unknown error")
        )
    if payload.get("scope_policy") != TRANSFORMATION_SCOPE_POLICY:
        errors.append(prefix + "transformation replay scope_policy mismatch")
    if not isinstance(payload.get("model_type"), str) or not payload.get("model_type"):
        errors.append(
            prefix + "transformation replay model_type must be a non-empty string"
        )
    if not isinstance(payload.get("architecture"), str) or not payload.get(
        "architecture"
    ):
        errors.append(
            prefix + "transformation replay architecture must be a non-empty string"
        )
    config_sha256 = payload.get("config_sha256")
    if (
        not isinstance(config_sha256, str)
        or _SHA256_RE.fullmatch(config_sha256) is None
    ):
        errors.append(
            prefix + "transformation replay config_sha256 must be a sha256 digest"
        )
    layer_count = payload.get("layer_count")
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
    ):
        errors.append(
            prefix + "transformation replay layer_count must be a positive int"
        )

    if spec is not None:
        expected_edit_type = _expected_edit_type(spec)
        if payload.get("edit_type") != expected_edit_type:
            errors.append(prefix + "transformation replay edit_type scenario mismatch")
        expected_transformation, expected_scope, expected_error = (
            _expected_literal_transformation(spec)
        )
        if expected_error is not None:
            errors.append(prefix + expected_error)
        elif expected_transformation is not None:
            if transformation is None or not _is_exact_json_value(
                transformation, expected_transformation
            ):
                errors.append(
                    prefix + "transformation replay parameters scenario mismatch"
                )
            if scope != expected_scope:
                errors.append(prefix + "transformation replay scope scenario mismatch")
    return transformation, scope, errors


def _replay_target_source_coverage_errors(
    *, prefix: str, payload: dict[str, Any]
) -> list[str]:
    target_manifest = payload.get("target_manifest")
    source_plan = payload.get("source_shard_plan")
    target_names = (
        {
            target.get("name")
            for target in target_manifest.get("targets", [])
            if isinstance(target, dict) and isinstance(target.get("name"), str)
        }
        if isinstance(target_manifest, dict)
        else set()
    )
    source_tensor_names = (
        {
            name
            for shard in source_plan.get("source_shards", [])
            if isinstance(shard, dict)
            for name in shard.get("tensor_names", [])
            if isinstance(name, str)
        }
        if isinstance(source_plan, dict)
        else set()
    )
    if target_names and not target_names <= source_tensor_names:
        return [
            prefix
            + "transformation replay target manifest is not covered by source plan"
        ]
    return []


def _replay_counter_and_plan_errors(
    *,
    prefix: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    transformation: dict[str, object] | None,
    scope: str | None,
) -> list[str]:
    errors: list[str] = []
    counter_fields = (
        "checked_tensors",
        "selected_tensors",
        "selected_params",
        "total_tensors",
        "total_params",
        "out_of_scope_tensors_checked",
        "out_of_scope_bytes_checked",
        "support_files_checked",
    )
    for field in counter_fields:
        if not _is_nonnegative_int(payload.get(field)):
            errors.append(
                prefix + f"transformation replay {field} must be a non-negative int"
            )
    selected_tensors = payload.get("selected_tensors")
    selected_params = payload.get("selected_params")
    total_tensors = payload.get("total_tensors")
    total_params = payload.get("total_params")
    if selected_tensors == 0 or selected_params == 0:
        errors.append(prefix + "transformation replay selected no tensors")
    if payload.get("checked_tensors") != (
        (selected_tensors or 0) + (payload.get("out_of_scope_tensors_checked") or 0)
    ):
        errors.append(prefix + "transformation replay tensor coverage count mismatch")
    if (
        isinstance(total_tensors, int)
        and not isinstance(total_tensors, bool)
        and isinstance(selected_tensors, int)
        and not isinstance(selected_tensors, bool)
        and total_tensors < selected_tensors
    ):
        errors.append(prefix + "transformation replay total tensor count invalid")
    if (
        isinstance(total_params, int)
        and not isinstance(total_params, bool)
        and isinstance(selected_params, int)
        and not isinstance(selected_params, bool)
        and total_params < selected_params
    ):
        errors.append(prefix + "transformation replay total parameter count invalid")
    if payload.get("support_files_checked") == 0:
        errors.append(prefix + "transformation replay checked no support files")
    errors.extend(
        _transformation_change_errors(
            prefix=prefix, actual_changes=payload.get("actual_changes")
        )
    )
    actual_changes = payload.get("actual_changes")
    if (
        isinstance(actual_changes, dict)
        and isinstance(selected_tensors, int)
        and not isinstance(selected_tensors, bool)
        and isinstance(selected_params, int)
        and not isinstance(selected_params, bool)
    ):
        for field in ("value_changed_tensors", "byte_changed_tensors"):
            value = actual_changes.get(field)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                and value > selected_tensors
            ):
                errors.append(prefix + f"transformation replay {field} exceeds targets")
        for field in ("value_changed_params", "byte_changed_params"):
            value = actual_changes.get(field)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                and value > selected_params
            ):
                errors.append(prefix + f"transformation replay {field} exceeds targets")

    for field in ("materialization_receipt_sha256", "edit_metadata_sha256"):
        if (
            not isinstance(payload.get(field), str)
            or _SHA256_RE.fullmatch(str(payload.get(field))) is None
        ):
            errors.append(
                prefix + f"transformation replay {field} must be a sha256 digest"
            )
    max_output_shard_bytes = payload.get("max_output_shard_bytes")
    if (
        isinstance(max_output_shard_bytes, bool)
        or not isinstance(max_output_shard_bytes, int)
        or max_output_shard_bytes < 1024 * 1024
    ):
        errors.append(
            prefix
            + "transformation replay max_output_shard_bytes must be an integer of at least 1 MiB"
        )
    errors.extend(_transformation_shard_plan_errors(prefix=prefix, payload=payload))
    errors.extend(
        _transformation_output_weights_errors(
            prefix=prefix, output_weights=payload.get("output_weights")
        )
    )

    errors.extend(_replay_target_source_coverage_errors(prefix=prefix, payload=payload))
    if transformation is not None and scope is not None:
        errors.extend(
            _transformation_target_manifest_errors(
                prefix=prefix,
                payload=payload,
                transformation=transformation,
                scope=scope,
            )
        )
        errors.extend(
            _transformation_metadata_errors(
                prefix=prefix,
                metadata=metadata,
                payload=payload,
                transformation=transformation,
                scope=scope,
            )
        )
    return errors


def _replay_identity_and_issue_errors(
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
            _transformation_identity_errors(
                prefix=prefix, label="artifact_identity", value=report_identity
            )
        )
        if report_identity != payload.get("artifact_identity"):
            errors.append(prefix + "transformation replay artifact identity mismatch")
    errors.extend(
        _transformation_identity_errors(
            prefix=prefix,
            label="artifact_identity",
            value=payload.get("artifact_identity"),
        )
    )
    if not isinstance(report_baseline_identity, dict):
        errors.append(prefix + "evaluation baseline identity missing")
    else:
        errors.extend(
            _transformation_identity_errors(
                prefix=prefix,
                label="baseline_identity",
                value=report_baseline_identity,
            )
        )
        if report_baseline_identity != payload.get("baseline_identity"):
            errors.append(prefix + "transformation replay baseline identity mismatch")
    errors.extend(
        _transformation_identity_errors(
            prefix=prefix,
            label="baseline_identity",
            value=payload.get("baseline_identity"),
        )
    )
    issues = payload.get("issues")
    if not isinstance(issues, list):
        errors.append(prefix + "transformation replay issues must be a list")
    elif issues:
        errors.append(prefix + "transformation replay issues must be empty when ok")
    return errors


def _replay_report_sidecar_errors(
    *,
    prefix: str,
    payload: dict[str, Any],
    report_dir: Path | None,
    transformation: dict[str, object] | None,
    scope: str | None,
) -> list[str]:
    errors: list[str] = []
    if report_dir is None:
        return [
            prefix + "transformation replay cannot verify report-sidecar cross-links"
        ]

    metadata_path = report_dir / "edit_metadata.json"
    if metadata_path.is_file() and not metadata_path.is_symlink():
        try:
            if payload.get("edit_metadata_sha256") != _sha256_file(metadata_path):
                errors.append(
                    prefix + "transformation replay edit metadata digest mismatch"
                )
        except OSError:
            errors.append(prefix + "transformation replay edit metadata is unreadable")
    else:
        errors.append(prefix + "transformation replay edit metadata sidecar missing")

    receipt_path = report_dir / TRANSFORMATION_MATERIALIZATION_RECEIPT
    if not receipt_path.is_file() or receipt_path.is_symlink():
        errors.append(prefix + "transformation materialization receipt sidecar missing")
        return errors
    try:
        receipt_digest = _sha256_file(receipt_path)
    except OSError:
        receipt_digest = None
        errors.append(prefix + "transformation materialization receipt is unreadable")
    if (
        receipt_digest is not None
        and payload.get("materialization_receipt_sha256") != receipt_digest
    ):
        errors.append(
            prefix + "transformation replay materialization receipt digest mismatch"
        )
    receipt, receipt_error = _load_json_sidecar(receipt_path)
    if receipt_error is not None or receipt is None:
        errors.append(
            prefix + "transformation materialization receipt sidecar is invalid"
        )
    elif transformation is not None and scope is not None:
        errors.extend(
            _transformation_materialization_receipt_errors(
                prefix=prefix,
                receipt=receipt,
                payload=payload,
                transformation=transformation,
                scope=scope,
            )
        )
    return errors


def _transformation_replay_errors(
    *,
    scenario_id: str,
    report: dict[str, Any],
    metadata: dict[str, Any],
    payload: dict[str, Any],
    spec: dict[str, Any] | None = None,
    pack_dir: Path | None = None,
    report_dir: Path | None = None,
    report_model_name: str | None = None,
) -> list[str]:
    """Verify that a generated transformation report is a replay-bound receipt."""

    prefix = f"{scenario_id}: "
    errors: list[str] = []
    transformation, scope, contract_errors = _replay_contract_errors(
        prefix=prefix, payload=payload, spec=spec
    )
    errors.extend(contract_errors)

    errors.extend(
        _replay_counter_and_plan_errors(
            prefix=prefix,
            payload=payload,
            metadata=metadata,
            transformation=transformation,
            scope=scope,
        )
    )

    errors.extend(
        _replay_identity_and_issue_errors(prefix=prefix, report=report, payload=payload)
    )

    errors.extend(
        _replay_report_sidecar_errors(
            prefix=prefix,
            payload=payload,
            report_dir=report_dir,
            transformation=transformation,
            scope=scope,
        )
    )

    is_clean = _is_clean_transformation_scenario(spec)
    if is_clean:
        if (
            pack_dir is None
            or report_model_name is None
            or transformation is None
            or scope is None
        ):
            errors.append(
                prefix
                + "clean generated transformation selection cannot be verified without pack metadata"
            )
        else:
            errors.extend(
                _clean_transformation_selection_errors(
                    pack_dir=pack_dir,
                    scenario_id=scenario_id,
                    report_model_name=report_model_name,
                    payload=payload,
                    transformation=transformation,
                    scope=scope,
                )
            )
    elif _REPLAY_SELECTION_FIELDS & set(payload):
        errors.append(
            prefix
            + "non-clean transformation replay must not carry a selection receipt"
        )
    return errors
