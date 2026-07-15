"""Generated-transformation artifact, selection, and materialization validation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

from invarlock.clean_selection.common import (
    CLEAN_SELECTION_CONTRACT_VERSION,
    CleanSelectionEvidenceError,
)
from invarlock.clean_selection.common import (
    canonical_json_sha256 as _clean_selection_canonical_sha256,
)
from invarlock.clean_selection.snapshot import snapshot_selection_bundle_file
from invarlock.evidence_pack_edit_common import (
    _SHA256_RE,
    TRANSFORMATION_CONTRACT_VERSION,
    TRANSFORMATION_EXECUTION_POLICY,
    TRANSFORMATION_MATERIALIZATION_RECEIPT_SCHEMA,
    TRANSFORMATION_SCOPE_POLICY,
    TRANSFORMATION_SELECTION_RECEIPT_SCHEMA,
    TRANSFORMATION_SELECTION_SOURCE_PATH,
    TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
    _finite_number,
    _is_nonnegative_int,
    _sanitize_model_key,
)
from invarlock.evidence_pack_json import (
    sha256_prefixed,
)
from invarlock.evidence_pack_transformation_contract import (
    _canonical_json_sha256,
    _is_exact_json_value,
)
from invarlock.transformation_target_manifest import (
    TransformationTargetManifestError,
    validate_transformation_target_manifest,
)


def _transformation_target_entry_errors(
    *,
    prefix: str,
    index: int,
    target: object,
    expected_roles: set[str],
    qualifiers: dict[str, int],
) -> tuple[str | None, int, list[str]]:
    label = f"transformation replay target_manifest.targets[{index}]"
    if not isinstance(target, dict):
        return None, 0, [prefix + label + " must be an object"]
    errors: list[str] = []
    expected_target_keys = {"name", "dtype", "shape", "numel", "role", "layer"}
    if set(target) != expected_target_keys:
        errors.append(prefix + label + " has unbound fields")
    name = target.get("name")
    dtype = target.get("dtype")
    shape = target.get("shape")
    numel = target.get("numel")
    role = target.get("role")
    layer = target.get("layer")
    valid_name = name if isinstance(name, str) and name else None
    if valid_name is None:
        errors.append(prefix + label + ".name must be a non-empty string")
    if (
        not isinstance(dtype, str)
        or "float" not in dtype.lower()
        or "float8" in dtype.lower()
        or "mxfp" in dtype.lower()
    ):
        errors.append(prefix + label + ".dtype must be regular floating-point storage")
    valid_shape = (
        isinstance(shape, list)
        and len(shape) >= 2
        and all(
            isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and dimension > 0
            for dimension in shape
        )
    )
    if not valid_shape:
        errors.append(prefix + label + ".shape must be a positive matrix shape")
    selected_params = 0
    if isinstance(numel, bool) or not isinstance(numel, int) or numel <= 0:
        errors.append(prefix + label + ".numel must be a positive int")
    else:
        selected_params = numel
        if valid_shape and isinstance(shape, list) and numel != math.prod(shape):
            errors.append(prefix + label + ".numel does not match shape")
    if not isinstance(role, str) or role not in expected_roles:
        errors.append(prefix + label + ".role is outside the declared scope")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        errors.append(prefix + label + ".layer must be a non-negative int")
    elif "layers" in qualifiers and layer >= qualifiers["layers"]:
        errors.append(prefix + label + ".layer is outside the layers qualifier")
    elif "layer" in qualifiers and layer != qualifiers["layer"]:
        errors.append(prefix + label + ".layer is outside the layer qualifier")
    return valid_name, selected_params, errors


def _transformation_target_manifest_errors(
    *,
    prefix: str,
    payload: dict[str, Any],
    transformation: dict[str, object],
    scope: str,
) -> list[str]:
    """Validate an explicit, canonical target manifest before trusting its hash."""

    errors: list[str] = []
    manifest = payload.get("target_manifest")
    if not isinstance(manifest, dict):
        return [prefix + "transformation replay target_manifest must be an object"]
    try:
        validate_transformation_target_manifest(manifest)
    except TransformationTargetManifestError as exc:
        errors.append(
            prefix
            + "transformation replay target_manifest policy violation: "
            + str(exc)
        )
    digest = payload.get("target_manifest_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        errors.append(
            prefix
            + "transformation replay target_manifest_sha256 must be a sha256 digest"
        )
    canonical_digest = _canonical_json_sha256(manifest)
    if canonical_digest is None:
        errors.append(prefix + "transformation replay target_manifest is not JSON-safe")
    elif digest != canonical_digest:
        errors.append(prefix + "transformation replay target_manifest digest mismatch")

    expected_values: dict[str, object] = {
        "schema": TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        "edit_type": transformation["edit_type"],
        "algorithm": transformation["algorithm"],
        "parameters": transformation["parameters"],
        "scope": scope,
        "model_type": payload.get("model_type"),
        "architecture": payload.get("architecture"),
        "config_sha256": payload.get("config_sha256"),
        "layer_count": payload.get("layer_count"),
    }
    expected_keys = {*expected_values, "targets"}
    if set(manifest) != expected_keys:
        errors.append(
            prefix + "transformation replay target_manifest has unbound fields"
        )
    for field, expected in expected_values.items():
        if not _is_exact_json_value(manifest.get(field), expected):
            errors.append(
                prefix + f"transformation replay target_manifest {field} mismatch"
            )

    targets = manifest.get("targets")
    if not isinstance(targets, list) or not targets:
        errors.append(
            prefix
            + "transformation replay target_manifest.targets must be a non-empty list"
        )
        return errors

    expected_roles = (
        {"ffn", "attn", "router"}
        if scope.split("@", 1)[0] == "all"
        else {scope.split("@", 1)[0]}
    )
    qualifiers = {
        name: int(value)
        for name, value in (
            item.split("=", 1) for item in scope.partition("@")[2].split(",") if item
        )
    }
    names: list[str] = []
    selected_params = 0
    for index, target in enumerate(targets):
        name, target_params, target_errors = _transformation_target_entry_errors(
            prefix=prefix,
            index=index,
            target=target,
            expected_roles=expected_roles,
            qualifiers=qualifiers,
        )
        errors.extend(target_errors)
        if name is not None:
            names.append(name)
        selected_params += target_params

    if names != sorted(names) or len(names) != len(set(names)):
        errors.append(
            prefix
            + "transformation replay target_manifest targets must be sorted and unique"
        )
    if payload.get("selected_tensors") != len(targets):
        errors.append(
            prefix
            + "transformation replay selected_tensors does not match target manifest"
        )
    if payload.get("selected_params") != selected_params:
        errors.append(
            prefix
            + "transformation replay selected_params does not match target manifest"
        )
    return errors


def _transformation_output_weights_errors(
    *,
    prefix: str,
    output_weights: object,
) -> list[str]:
    if not isinstance(output_weights, dict):
        return [prefix + "transformation replay output_weights must be an object"]
    if set(output_weights) != {"sha256", "index_sha256", "shards"}:
        return [prefix + "transformation replay output_weights has unbound fields"]
    index_sha256 = output_weights.get("index_sha256")
    digest = output_weights.get("sha256")
    shards = output_weights.get("shards")
    errors: list[str] = []
    for field, value in (("sha256", digest), ("index_sha256", index_sha256)):
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            errors.append(
                prefix + f"transformation replay output_weights.{field} is invalid"
            )
    if not isinstance(shards, list) or not shards:
        return errors + [
            prefix + "transformation replay output_weights.shards must be non-empty"
        ]
    names: list[str] = []
    canonical_shards: list[dict[str, str]] = []
    for index, shard in enumerate(shards):
        label = f"transformation replay output_weights.shards[{index}]"
        if not isinstance(shard, dict) or set(shard) != {"name", "sha256"}:
            errors.append(prefix + label + " must contain only name and sha256")
            continue
        name = shard.get("name")
        shard_digest = shard.get("sha256")
        if (
            not isinstance(name, str)
            or not name.endswith(".safetensors")
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
        ):
            errors.append(prefix + label + ".name must be a safe safetensors filename")
        else:
            names.append(name)
        if (
            not isinstance(shard_digest, str)
            or _SHA256_RE.fullmatch(shard_digest) is None
        ):
            errors.append(prefix + label + ".sha256 must be a sha256 digest")
        if isinstance(name, str) and isinstance(shard_digest, str):
            canonical_shards.append({"name": name, "sha256": shard_digest})
    if names != sorted(names) or len(names) != len(set(names)):
        errors.append(
            prefix
            + "transformation replay output weight shards must be sorted and unique"
        )
    if not errors and isinstance(index_sha256, str):
        expected_digest = _canonical_json_sha256(
            {"index_sha256": index_sha256, "shards": canonical_shards}
        )
        if digest != expected_digest:
            errors.append(
                prefix + "transformation replay output_weights digest mismatch"
            )
    return errors


def _safe_checkpoint_relative_path(value: object) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\\" in value
    ):
        return False
    parts = value.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _source_shard_entry_errors(
    *, prefix: str, index: int, shard: object
) -> tuple[str | None, tuple[str, str, tuple[str, ...]] | None, list[str]]:
    label = f"transformation replay source_shard_plan.source_shards[{index}]"
    expected_fields = {"path", "sha256", "tensor_names", "byte_count"}
    if not isinstance(shard, dict) or set(shard) != expected_fields:
        return None, None, [prefix + label + " has unbound fields"]
    errors: list[str] = []
    path = shard.get("path")
    digest = shard.get("sha256")
    names = shard.get("tensor_names")
    byte_count = shard.get("byte_count")
    valid_path = (
        cast(str, path)
        if _safe_checkpoint_relative_path(path) and str(path).endswith(".safetensors")
        else None
    )
    if valid_path is None:
        errors.append(prefix + label + ".path is not a safe safetensors path")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        errors.append(prefix + label + ".sha256 must be a sha256 digest")
    valid_names = (
        isinstance(names, list)
        and bool(names)
        and all(isinstance(name, str) and name for name in names)
        and names == sorted(names)
        and len(names) == len(set(names))
    )
    if not valid_names:
        errors.append(prefix + label + ".tensor_names must be sorted and unique")
    if not _is_nonnegative_int(byte_count) or byte_count == 0:
        errors.append(prefix + label + ".byte_count must be positive")
    entry = (
        (path, digest, tuple(cast(list[str], names)))
        if isinstance(path, str)
        and isinstance(digest, str)
        and isinstance(names, list)
        and all(isinstance(name, str) for name in names)
        else None
    )
    return valid_path, entry, errors


def _output_chunk_errors(
    *,
    prefix: str,
    index: int,
    chunk: object,
    source_entries: dict[str, tuple[str, tuple[str, ...]]],
    max_output_shard_bytes: object,
) -> tuple[str | None, list[str], list[str]]:
    label = f"transformation replay output_shard_plan.chunks[{index}]"
    expected_fields = {
        "name",
        "source_path",
        "source_sha256",
        "tensor_names",
        "byte_count",
    }
    if not isinstance(chunk, dict) or set(chunk) != expected_fields:
        return None, [], [prefix + label + " has unbound fields"]
    errors: list[str] = []
    name = chunk.get("name")
    source_path = chunk.get("source_path")
    source_sha256 = chunk.get("source_sha256")
    tensor_names = chunk.get("tensor_names")
    byte_count = chunk.get("byte_count")
    valid_name = (
        name
        if isinstance(name, str)
        and name.endswith(".safetensors")
        and _safe_checkpoint_relative_path(name)
        and "/" not in name
        else None
    )
    if valid_name is None:
        errors.append(prefix + label + ".name is not a safe output shard name")
    expected_source = None
    if not _safe_checkpoint_relative_path(source_path) or not str(source_path).endswith(
        ".safetensors"
    ):
        errors.append(prefix + label + ".source_path is invalid")
    else:
        expected_source = source_entries.get(cast(str, source_path))
        if expected_source is None:
            errors.append(prefix + label + ".source_path is not in source plan")
        elif source_sha256 != expected_source[0]:
            errors.append(prefix + label + ".source_sha256 mismatch")
    valid_names = (
        isinstance(tensor_names, list)
        and bool(tensor_names)
        and all(isinstance(tensor, str) and tensor for tensor in tensor_names)
        and tensor_names == sorted(tensor_names)
        and len(tensor_names) == len(set(tensor_names))
    )
    output_names: list[str] = []
    if not valid_names:
        errors.append(prefix + label + ".tensor_names must be sorted and unique")
    elif expected_source is not None:
        source_names = set(expected_source[1])
        output_names = cast(list[str], tensor_names)
        if not set(output_names) <= source_names:
            errors.append(prefix + label + ".tensor_names are not in source shard")
    if not _is_nonnegative_int(byte_count) or byte_count == 0:
        errors.append(prefix + label + ".byte_count must be positive")
    elif (
        isinstance(max_output_shard_bytes, int)
        and not isinstance(max_output_shard_bytes, bool)
        and isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and byte_count > max_output_shard_bytes
        and isinstance(tensor_names, list)
        and len(tensor_names) != 1
    ):
        errors.append(
            prefix + label + ".byte_count exceeds the bound for a multi-tensor shard"
        )
    return valid_name, output_names, errors


def _transformation_shard_plan_errors(
    *,
    prefix: str,
    payload: dict[str, Any],
) -> list[str]:
    """Require inspectable source/output plans, not just opaque plan digests."""

    errors: list[str] = []
    source_plan = payload.get("source_shard_plan")
    output_plan = payload.get("output_shard_plan")
    source_digest = payload.get("source_shard_plan_sha256")
    output_digest = payload.get("output_shard_plan_sha256")
    if not isinstance(source_plan, dict) or set(source_plan) != {"source_shards"}:
        errors.append(prefix + "transformation replay source_shard_plan is invalid")
        source_shards: list[object] = []
    else:
        source_shards = source_plan.get("source_shards", [])
        if not isinstance(source_shards, list) or not source_shards:
            errors.append(
                prefix
                + "transformation replay source_shard_plan.source_shards is empty"
            )
            source_shards = []
    canonical_source_digest = _canonical_json_sha256(source_plan)
    if canonical_source_digest is None or source_digest != canonical_source_digest:
        errors.append(
            prefix + "transformation replay source_shard_plan digest mismatch"
        )

    source_entries: dict[str, tuple[str, tuple[str, ...]]] = {}
    source_paths: list[str] = []
    for index, shard in enumerate(source_shards):
        path, entry, entry_errors = _source_shard_entry_errors(
            prefix=prefix, index=index, shard=shard
        )
        errors.extend(entry_errors)
        if path is not None:
            source_paths.append(path)
        if entry is not None:
            entry_path, digest, names = entry
            source_entries[entry_path] = (digest, names)
    if source_paths != sorted(source_paths) or len(source_paths) != len(
        set(source_paths)
    ):
        errors.append(
            prefix
            + "transformation replay source shard paths must be sorted and unique"
        )

    if not isinstance(output_plan, dict) or set(output_plan) != {
        "source_shard_plan_sha256",
        "target_manifest_sha256",
        "chunks",
    }:
        errors.append(prefix + "transformation replay output_shard_plan is invalid")
        chunks: list[object] = []
    else:
        if output_plan.get("source_shard_plan_sha256") != source_digest:
            errors.append(
                prefix
                + "transformation replay output_shard_plan source plan digest mismatch"
            )
        if output_plan.get("target_manifest_sha256") != payload.get(
            "target_manifest_sha256"
        ):
            errors.append(
                prefix
                + "transformation replay output_shard_plan target manifest digest mismatch"
            )
        chunks = output_plan.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            errors.append(
                prefix + "transformation replay output_shard_plan.chunks is empty"
            )
            chunks = []
    canonical_output_digest = _canonical_json_sha256(output_plan)
    if canonical_output_digest is None or output_digest != canonical_output_digest:
        errors.append(
            prefix + "transformation replay output_shard_plan digest mismatch"
        )

    max_output_shard_bytes = payload.get("max_output_shard_bytes")
    chunk_names: list[str] = []
    output_tensor_names: list[str] = []
    for index, chunk in enumerate(chunks):
        name, tensor_names, chunk_errors = _output_chunk_errors(
            prefix=prefix,
            index=index,
            chunk=chunk,
            source_entries=source_entries,
            max_output_shard_bytes=max_output_shard_bytes,
        )
        errors.extend(chunk_errors)
        if name is not None:
            chunk_names.append(name)
        output_tensor_names.extend(tensor_names)
    if chunk_names != sorted(chunk_names) or len(chunk_names) != len(set(chunk_names)):
        errors.append(
            prefix
            + "transformation replay output shard names must be sorted and unique"
        )
    expected_output_names = sorted(
        name for _, (_, names) in source_entries.items() for name in names
    )
    if sorted(output_tensor_names) != expected_output_names or len(
        output_tensor_names
    ) != len(set(output_tensor_names)):
        errors.append(
            prefix
            + "transformation replay output shard plan does not cover source tensors exactly"
        )
    output_weights = payload.get("output_weights")
    output_weight_shards = (
        output_weights.get("shards") if isinstance(output_weights, dict) else None
    )
    output_weight_names = (
        [entry.get("name") for entry in output_weight_shards if isinstance(entry, dict)]
        if isinstance(output_weight_shards, list)
        else []
    )
    if output_weight_names != chunk_names:
        errors.append(
            prefix
            + "transformation replay output weights do not match output shard plan"
        )
    return errors


def _transformation_change_errors(
    *,
    prefix: str,
    actual_changes: object,
) -> list[str]:
    if not isinstance(actual_changes, dict):
        return [prefix + "transformation replay actual_changes must be an object"]
    expected_fields = {
        "value_changed_tensors",
        "value_changed_params",
        "byte_changed_tensors",
        "byte_changed_params",
    }
    errors: list[str] = []
    if set(actual_changes) != expected_fields:
        errors.append(
            prefix + "transformation replay actual_changes has unbound fields"
        )
    for field in sorted(expected_fields):
        value = actual_changes.get(field)
        if not _is_nonnegative_int(value):
            errors.append(
                prefix
                + f"transformation replay actual_changes.{field} must be a non-negative int"
            )
        elif isinstance(value, int) and not isinstance(value, bool) and value <= 0:
            errors.append(
                prefix
                + f"transformation replay actual_changes.{field} must be positive"
            )
    return errors


def _transformation_metadata_errors(
    *,
    prefix: str,
    metadata: dict[str, Any],
    payload: dict[str, Any],
    transformation: dict[str, object],
    scope: str,
) -> list[str]:
    expected_values: dict[str, object] = {
        "edit_type": transformation["edit_type"],
        "scope": scope,
        "parameters": transformation["parameters"],
        "transformation_contract": transformation,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        "model_type": payload.get("model_type"),
        "transformation_architecture": payload.get("architecture"),
        "config_sha256": payload.get("config_sha256"),
        "layer_count": payload.get("layer_count"),
        "target_manifest": payload.get("target_manifest"),
        "target_manifest_sha256": payload.get("target_manifest_sha256"),
        "max_output_shard_bytes": payload.get("max_output_shard_bytes"),
        "source_shard_plan": payload.get("source_shard_plan"),
        "source_shard_plan_sha256": payload.get("source_shard_plan_sha256"),
        "output_shard_plan": payload.get("output_shard_plan"),
        "output_shard_plan_sha256": payload.get("output_shard_plan_sha256"),
        "selected_tensors": payload.get("selected_tensors"),
        "selected_params": payload.get("selected_params"),
        "actual_changes": payload.get("actual_changes"),
        "materialization": "resumable_bounded_safetensors_v1",
        "execution_policy": TRANSFORMATION_EXECUTION_POLICY,
    }
    errors: list[str] = []
    for field, expected in expected_values.items():
        if not _is_exact_json_value(metadata.get(field), expected):
            errors.append(prefix + f"transformation replay metadata {field} mismatch")

    coverage = metadata.get("coverage")
    if not isinstance(coverage, dict):
        return errors + [prefix + "transformation replay metadata coverage missing"]
    expected_coverage = {
        "edited_tensors": payload.get("selected_tensors"),
        "edited_params": payload.get("selected_params"),
        "total_params": payload.get("total_params"),
    }
    for field, expected in expected_coverage.items():
        if not _is_exact_json_value(coverage.get(field), expected):
            errors.append(
                prefix + f"transformation replay metadata coverage.{field} mismatch"
            )
    total_params = payload.get("total_params")
    selected_params = payload.get("selected_params")
    expected_ratio = (
        float(selected_params) / float(total_params)
        if isinstance(total_params, int)
        and not isinstance(total_params, bool)
        and total_params > 0
        and isinstance(selected_params, int)
        and not isinstance(selected_params, bool)
        else None
    )
    ratio = _finite_number(coverage.get("coverage_ratio"))
    if (
        expected_ratio is None
        or ratio is None
        or not math.isclose(ratio, expected_ratio, abs_tol=1e-12)
    ):
        errors.append(
            prefix + "transformation replay metadata coverage.coverage_ratio mismatch"
        )
    return errors


def _transformation_materialization_receipt_errors(
    *,
    prefix: str,
    receipt: dict[str, Any],
    payload: dict[str, Any],
    transformation: dict[str, object],
    scope: str,
) -> list[str]:
    expected_fields = {
        "schema",
        "ok",
        "baseline_identity",
        "transformation",
        "scope",
        "scope_policy",
        "model_type",
        "architecture",
        "config_sha256",
        "layer_count",
        "target_manifest",
        "target_manifest_sha256",
        "max_output_shard_bytes",
        "source_shard_plan",
        "source_shard_plan_sha256",
        "output_shard_plan",
        "output_shard_plan_sha256",
        "output_weights",
        "execution_policy",
        "output_shards",
        "resume_count",
        "selected_tensors",
        "selected_params",
        "out_of_scope_tensors",
        "out_of_scope_params",
        "total_tensors",
        "total_params",
        "actual_changes",
    }
    errors: list[str] = []
    if set(receipt) != expected_fields:
        errors.append(
            prefix + "transformation materialization receipt has unbound fields"
        )
    replay_total_params = payload.get("total_params")
    replay_selected_params = payload.get("selected_params")
    out_of_scope_params = (
        replay_total_params - replay_selected_params
        if isinstance(replay_total_params, int)
        and not isinstance(replay_total_params, bool)
        and isinstance(replay_selected_params, int)
        and not isinstance(replay_selected_params, bool)
        else None
    )
    expected_values: dict[str, object] = {
        "schema": TRANSFORMATION_MATERIALIZATION_RECEIPT_SCHEMA,
        "ok": True,
        "baseline_identity": payload.get("baseline_identity"),
        "transformation": transformation,
        "scope": scope,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        "model_type": payload.get("model_type"),
        "architecture": payload.get("architecture"),
        "config_sha256": payload.get("config_sha256"),
        "layer_count": payload.get("layer_count"),
        "target_manifest": payload.get("target_manifest"),
        "target_manifest_sha256": payload.get("target_manifest_sha256"),
        "max_output_shard_bytes": payload.get("max_output_shard_bytes"),
        "source_shard_plan": payload.get("source_shard_plan"),
        "source_shard_plan_sha256": payload.get("source_shard_plan_sha256"),
        "output_shard_plan": payload.get("output_shard_plan"),
        "output_shard_plan_sha256": payload.get("output_shard_plan_sha256"),
        "output_weights": payload.get("output_weights"),
        "execution_policy": TRANSFORMATION_EXECUTION_POLICY,
        "selected_tensors": payload.get("selected_tensors"),
        "selected_params": payload.get("selected_params"),
        "out_of_scope_tensors": payload.get("out_of_scope_tensors_checked"),
        "out_of_scope_params": out_of_scope_params,
        "total_tensors": payload.get("total_tensors"),
        "total_params": payload.get("total_params"),
        "actual_changes": payload.get("actual_changes"),
    }
    for field, expected in expected_values.items():
        if not _is_exact_json_value(receipt.get(field), expected):
            errors.append(
                prefix + f"transformation materialization receipt {field} mismatch"
            )
    output_shards = receipt.get("output_shards")
    resume_count = receipt.get("resume_count")
    if (
        not isinstance(output_shards, int)
        or isinstance(output_shards, bool)
        or output_shards <= 0
    ):
        errors.append(
            prefix
            + "transformation materialization receipt output_shards must be positive"
        )
    elif isinstance(payload.get("output_weights"), dict):
        shards = payload["output_weights"].get("shards")
        if isinstance(shards, list) and output_shards != len(shards):
            errors.append(
                prefix + "transformation materialization receipt output_shards mismatch"
            )
    if not _is_nonnegative_int(resume_count):
        errors.append(
            prefix
            + "transformation materialization receipt resume_count must be a non-negative int"
        )
    return errors


def _clean_transformation_selection_errors(
    *,
    pack_dir: Path,
    scenario_id: str,
    report_model_name: str,
    payload: dict[str, Any],
    transformation: dict[str, object],
    scope: str,
) -> list[str]:
    """Bind a final clean replay to all retained v1 candidate evidence."""

    prefix = f"{scenario_id}: "
    source_path = pack_dir / TRANSFORMATION_SELECTION_SOURCE_PATH
    try:
        bundle_snapshot = snapshot_selection_bundle_file(
            source_path, evidence_root=source_path.parent
        )
        bundle = bundle_snapshot.bundle
        source_digest = sha256_prefixed(bundle_snapshot.bundle_bytes)
    except CleanSelectionEvidenceError as exc:
        return [
            prefix
            + "clean generated transformation v1 selection bundle invalid: "
            + str(exc)
        ]

    receipt = payload.get("selection_receipt")
    receipt_digest = payload.get("selection_receipt_sha256")
    if not isinstance(receipt, dict):
        return [prefix + "clean generated transformation selection_receipt is missing"]
    if (
        not isinstance(receipt_digest, str)
        or _SHA256_RE.fullmatch(receipt_digest) is None
    ):
        return [
            prefix
            + "clean generated transformation selection_receipt_sha256 must be a sha256 digest"
        ]
    if receipt_digest != _clean_selection_canonical_sha256(receipt):
        return [
            prefix + "clean generated transformation selection receipt digest mismatch"
        ]
    expected_fields = {
        "schema",
        "contract_version",
        "transformation_contract_version",
        "scenario_id",
        "selection_bundle_path",
        "selection_bundle_sha256",
        "original_model_key",
        "edit_type",
        "algorithm",
        "parameters",
        "scope",
        "selected_candidate_id",
        "candidate_set_sha256",
        "selected_entry_sha256",
        "baseline_identity",
        "artifact_identity",
    }
    if set(receipt) != expected_fields:
        return [
            prefix
            + "clean generated transformation selection receipt has unbound fields"
        ]
    model_key = receipt.get("original_model_key")
    if (
        not isinstance(model_key, str)
        or _sanitize_model_key(model_key) != report_model_name
    ):
        return [
            prefix
            + "clean generated transformation selection receipt original_model_key mismatch"
        ]
    entries = cast(list[dict[str, object]], bundle["entries"])
    matching_entries: list[dict[str, object]] = []
    for entry in entries:
        if entry.get("original_model_key") != model_key:
            continue
        selected = cast(dict[str, object], entry["selected_entry"])
        if selected.get("edit_type") == transformation.get("edit_type"):
            matching_entries.append(entry)
    if len(matching_entries) != 1:
        return [
            prefix
            + "clean generated transformation v1 selection bundle has no unique model/edit entry"
        ]
    selected_entry = matching_entries[0]
    selected = cast(dict[str, object], selected_entry["selected_entry"])
    candidate_receipt = cast(dict[str, object], selected["selection_receipt"])
    selected_transformation = cast(
        dict[str, object], candidate_receipt["selected_transformation"]
    )
    selected_evaluation = cast(
        dict[str, object], candidate_receipt["selected_evaluation"]
    )
    report_runs = cast(list[dict[str, object]], selected_evaluation["reports"])
    report_reference = cast(dict[str, object], report_runs[0]["report"])
    expected_values: dict[str, object] = {
        "schema": TRANSFORMATION_SELECTION_RECEIPT_SCHEMA,
        "contract_version": CLEAN_SELECTION_CONTRACT_VERSION,
        "transformation_contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "scenario_id": scenario_id,
        "selection_bundle_path": TRANSFORMATION_SELECTION_SOURCE_PATH,
        "selection_bundle_sha256": source_digest,
        "original_model_key": model_key,
        "edit_type": transformation.get("edit_type"),
        "algorithm": transformation.get("algorithm"),
        "parameters": transformation.get("parameters"),
        "scope": scope,
        "selected_candidate_id": candidate_receipt.get("selected_candidate_id"),
        "candidate_set_sha256": candidate_receipt.get("candidate_set_sha256"),
        "selected_entry_sha256": _clean_selection_canonical_sha256(selected_entry),
        "baseline_identity": payload.get("baseline_identity"),
        "artifact_identity": payload.get("artifact_identity"),
    }
    errors: list[str] = []
    for field, expected in expected_values.items():
        if not _is_exact_json_value(receipt.get(field), expected):
            errors.append(
                prefix
                + f"clean generated transformation selection receipt {field} mismatch"
            )
    expected_selected = {
        "edit_type": transformation.get("edit_type"),
        "parameters": transformation.get("parameters"),
        "scope": scope,
    }
    if not _is_exact_json_value(selected_transformation, expected_selected):
        errors.append(
            prefix
            + "clean generated transformation selected candidate differs from final replay"
        )
    if report_reference.get("baseline_identity") != payload.get("baseline_identity"):
        errors.append(
            prefix
            + "clean generated transformation selected candidate baseline identity mismatch"
        )
    if report_reference.get("artifact_identity") != payload.get("artifact_identity"):
        errors.append(
            prefix
            + "clean generated transformation selected candidate artifact identity mismatch"
        )
    return errors
