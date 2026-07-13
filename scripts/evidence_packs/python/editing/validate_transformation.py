"""Canonical transformation receipt and exact tensor-replay validation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifact_tensor_validation import (
    _canonical_json_sha256,
    _load_safetensor_tensor,
    _strict_json_object,
    _strict_safetensor_weight_map,
    _support_file_digests,
    _tensor_bytes_equal,
)
from .checkpoint_paths import CheckpointLayoutError, require_regular_checkpoint_tree
from .transformation_oracle import (
    CANONICAL_EXECUTION_POLICY,
    TRANSFORMATION_MATERIALIZATION_RECEIPT,
    TRANSFORMATION_MATERIALIZATION_RECEIPT_SCHEMA,
    TRANSFORMATION_REPLAY_SCHEMA,
    TRANSFORMATION_SCOPE_POLICY_VERSION,
    TransformationOracle,
    TransformationOracleError,
    build_transformation_oracle,
    checkpoint_contract,
)
from .transformation_shard_validation import (
    _expected_output_shard_plan,
    _source_shard_plan,
)

try:
    from safetensors import SafetensorError, safe_open
except ImportError:  # pragma: no cover
    safe_open = None
    SafetensorError = RuntimeError


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _transformation_change_stats(source: Any, artifact: Any) -> dict[str, int]:
    import torch

    if source.shape != artifact.shape or source.dtype != artifact.dtype:
        raise ValueError("transformation changed tensor shape or dtype")
    source = source.detach().to(device="cpu").contiguous()
    artifact = artifact.detach().to(device="cpu").contiguous()
    value_changed_params = int(torch.ne(source, artifact).sum().item())
    source_bytes = source.view(torch.uint8).reshape(
        source.numel(), source.element_size()
    )
    artifact_bytes = artifact.view(torch.uint8).reshape(
        artifact.numel(), artifact.element_size()
    )
    byte_changed_params = int(
        torch.any(source_bytes != artifact_bytes, dim=1).sum().item()
    )
    return {
        "value_changed_tensors": int(value_changed_params > 0),
        "value_changed_params": value_changed_params,
        "byte_changed_tensors": int(byte_changed_params > 0),
        "byte_changed_params": byte_changed_params,
    }


def _change_stat_errors(
    payload: object,
    *,
    expected: Mapping[str, int],
    label: str,
) -> list[str]:
    if not isinstance(payload, Mapping) or set(payload) != set(expected):
        return [f"{label} must contain exactly the replay change statistics"]
    errors: list[str] = []
    for field, expected_value in expected.items():
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{label}.{field} must be a non-negative integer")
        elif value != expected_value:
            errors.append(f"{label}.{field} does not match replay")
    return errors


def _output_weight_identity(
    artifact_dir: Path,
    *,
    weight_map: Mapping[str, Path],
    index_path: Path | None,
) -> dict[str, object] | None:
    if index_path is None:
        return None
    shards = sorted(set(weight_map.values()), key=lambda path: path.as_posix())
    payload = {
        "index_sha256": _file_sha256(index_path),
        "shards": [
            {
                "name": path.relative_to(artifact_dir).as_posix(),
                "sha256": _file_sha256(path),
            }
            for path in shards
        ],
    }
    return {"sha256": _canonical_json_sha256(payload), **payload}


def _transformation_metadata_errors(
    metadata: Mapping[str, object],
    *,
    spec: Mapping[str, object],
    scope: str,
    contract: Any,
    target_manifest: Mapping[str, object],
    target_manifest_sha256: str,
    max_output_shard_bytes: int,
    source_shard_plan: Mapping[str, object],
    source_shard_plan_sha256: str,
    output_shard_plan: Mapping[str, object],
    output_shard_plan_sha256: str,
    selected_tensors: int,
    selected_params: int,
    total_params: int,
    actual_changes: Mapping[str, int],
    execution_policy: str,
) -> list[str]:
    """Check every metadata field that claims a replay-relevant fact."""

    errors: list[str] = []
    expected_values: dict[str, object] = {
        "scope": scope,
        "parameters": spec["parameters"],
        "transformation_contract": dict(spec),
        "scope_policy": TRANSFORMATION_SCOPE_POLICY_VERSION,
        "model_type": contract.model_type,
        "transformation_architecture": contract.architecture,
        "config_sha256": contract.config_sha256,
        "layer_count": contract.layer_count,
        "target_manifest": dict(target_manifest),
        "target_manifest_sha256": target_manifest_sha256,
        "max_output_shard_bytes": max_output_shard_bytes,
        "source_shard_plan": dict(source_shard_plan),
        "source_shard_plan_sha256": source_shard_plan_sha256,
        "output_shard_plan": dict(output_shard_plan),
        "output_shard_plan_sha256": output_shard_plan_sha256,
        "selected_tensors": selected_tensors,
        "selected_params": selected_params,
        "materialization": "resumable_bounded_safetensors_v1",
        "execution_policy": execution_policy,
    }
    for field, expected in expected_values.items():
        if metadata.get(field) != expected:
            errors.append(f"edit_metadata.{field} does not match transformation replay")

    coverage = metadata.get("coverage")
    if not isinstance(coverage, Mapping):
        errors.append("edit_metadata.coverage must be an object")
    else:
        expected_coverage = {
            "edited_tensors": selected_tensors,
            "edited_params": selected_params,
            "total_params": total_params,
        }
        for field, expected in expected_coverage.items():
            value = coverage.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(
                    f"edit_metadata.coverage.{field} must be a non-negative integer"
                )
            elif value != expected:
                errors.append(
                    f"edit_metadata.coverage.{field} does not match transformation replay"
                )
        expected_ratio = selected_params / total_params if total_params else 0.0
        ratio = _finite_number(coverage.get("coverage_ratio"))
        if ratio is None or not math.isclose(ratio, expected_ratio, abs_tol=1e-12):
            errors.append(
                "edit_metadata.coverage.coverage_ratio does not match transformation replay"
            )
    errors.extend(
        _change_stat_errors(
            metadata.get("actual_changes"),
            expected=actual_changes,
            label="edit_metadata.actual_changes",
        )
    )
    return errors


def _transformation_receipt_errors(
    receipt: Mapping[str, object],
    *,
    receipt_schema: str,
    baseline_identity: Mapping[str, str] | None,
    spec: Mapping[str, object],
    scope: str,
    contract: Any,
    target_manifest: Mapping[str, object],
    target_manifest_sha256: str,
    max_output_shard_bytes: int,
    source_shard_plan: Mapping[str, object],
    source_shard_plan_sha256: str,
    output_shard_plan: Mapping[str, object],
    output_shard_plan_sha256: str,
    output_weights: Mapping[str, object] | None,
    selected_tensors: int,
    selected_params: int,
    total_tensors: int,
    total_params: int,
    actual_changes: Mapping[str, int],
    execution_policy: str,
) -> list[str]:
    """Require the materializer receipt to bind the replayed checkpoint exactly."""

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
        errors.append("transformation materialization receipt has unbound fields")

    expected_values: dict[str, object] = {
        "schema": receipt_schema,
        "ok": True,
        "baseline_identity": dict(baseline_identity)
        if baseline_identity is not None
        else None,
        "transformation": dict(spec),
        "scope": scope,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY_VERSION,
        "model_type": contract.model_type,
        "architecture": contract.architecture,
        "config_sha256": contract.config_sha256,
        "layer_count": contract.layer_count,
        "target_manifest": dict(target_manifest),
        "target_manifest_sha256": target_manifest_sha256,
        "max_output_shard_bytes": max_output_shard_bytes,
        "source_shard_plan": dict(source_shard_plan),
        "source_shard_plan_sha256": source_shard_plan_sha256,
        "output_shard_plan": dict(output_shard_plan),
        "output_shard_plan_sha256": output_shard_plan_sha256,
        "output_weights": dict(output_weights) if output_weights is not None else None,
        "execution_policy": execution_policy,
        "selected_tensors": selected_tensors,
        "selected_params": selected_params,
        "out_of_scope_tensors": total_tensors - selected_tensors,
        "out_of_scope_params": total_params - selected_params,
        "total_tensors": total_tensors,
        "total_params": total_params,
    }
    for field, expected in expected_values.items():
        if receipt.get(field) != expected:
            errors.append(
                f"transformation materialization receipt {field} does not match replay"
            )

    for field in ("output_shards", "resume_count"):
        value = receipt.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(
                f"transformation materialization receipt {field} must be a non-negative integer"
            )
    expected_chunk_count = len(output_shard_plan.get("chunks", []))
    if receipt.get("output_shards") != expected_chunk_count:
        errors.append(
            "transformation materialization receipt output_shards does not match canonical output plan"
        )
    errors.extend(
        _change_stat_errors(
            receipt.get("actual_changes"),
            expected=actual_changes,
            label="transformation materialization receipt actual_changes",
        )
    )
    return errors


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _replay_tensors(
    *,
    spec: dict[str, object] | None,
    normalized_scope: str | None,
    baseline_contract: Any | None,
    oracle: TransformationOracle | None,
    baseline_map: dict[str, Path],
    artifact_map: dict[str, Path],
    baseline_map_issues: list[str],
    artifact_map_issues: list[str],
    issues: list[str],
) -> dict[str, Any]:
    import torch

    baseline_keys = set(baseline_map)
    artifact_keys = set(artifact_map)
    selected_tensors = 0
    selected_params = 0
    total_tensors = 0
    total_params = 0
    checked_tensors = 0
    out_of_scope_tensors = 0
    out_of_scope_bytes_checked = 0
    selected_entries: list[dict[str, object]] = []
    tensor_bytes: dict[str, int] = {}
    actual_changes = {
        "value_changed_tensors": 0,
        "value_changed_params": 0,
        "byte_changed_tensors": 0,
        "byte_changed_params": 0,
    }

    can_replay = (
        spec is not None
        and normalized_scope is not None
        and baseline_contract is not None
        and oracle is not None
        and not baseline_map_issues
        and not artifact_map_issues
        and baseline_keys == artifact_keys
    )
    if can_replay:
        canonical_parameters = spec["parameters"]
        assert isinstance(canonical_parameters, Mapping)
        for key in sorted(baseline_keys):
            try:
                baseline_tensor = _load_safetensor_tensor(baseline_map[key], key)
                artifact_tensor = _load_safetensor_tensor(artifact_map[key], key)
            except (OSError, RuntimeError, SafetensorError) as exc:
                issues.append(f"{key}: tensor cannot be read for replay: {exc}")
                continue
            checked_tensors += 1
            total_tensors += 1
            total_params += int(baseline_tensor.numel())
            tensor_bytes[key] = int(
                baseline_tensor.numel() * baseline_tensor.element_size()
            )
            if baseline_tensor.shape != artifact_tensor.shape:
                issues.append(f"{key}: tensor shape mismatch")
                continue
            if baseline_tensor.dtype != artifact_tensor.dtype:
                issues.append(f"{key}: tensor dtype mismatch")
                continue
            for source_label, tensor in (
                ("baseline", baseline_tensor),
                ("artifact", artifact_tensor),
            ):
                if torch.is_floating_point(tensor) and not bool(
                    torch.isfinite(tensor).all().item()
                ):
                    issues.append(
                        f"{key}: {source_label} tensor contains non-finite values"
                    )
                if "float8" in str(tensor.dtype).lower():
                    issues.append(
                        f"{key}: {source_label} tensor uses unsupported FP8 storage"
                    )

            assert oracle is not None
            is_target = oracle.is_target(key, baseline_tensor)
            if not is_target:
                out_of_scope_tensors += 1
                out_of_scope_bytes_checked += tensor_bytes[key]
                if not _tensor_bytes_equal(artifact_tensor, baseline_tensor):
                    issues.append(f"{key}: out-of-scope tensor changed")
                continue

            if not torch.is_floating_point(baseline_tensor):
                issues.append(
                    f"{key}: transformation replay requires a floating-point target tensor"
                )
                continue
            selected_tensors += 1
            selected_params += int(baseline_tensor.numel())
            try:
                selected_entries.append(oracle.target_entry(key, baseline_tensor))
                expected_tensor = oracle.replay_tensor(baseline_tensor)
            except (RuntimeError, TransformationOracleError, ValueError) as exc:
                issues.append(f"{key}: independent transformation replay failed: {exc}")
                continue
            if not _tensor_bytes_equal(artifact_tensor, expected_tensor):
                issues.append(
                    f"{key}: artifact does not match exact transformation replay"
                )
            try:
                changes = _transformation_change_stats(baseline_tensor, artifact_tensor)
            except ValueError as exc:
                issues.append(f"{key}: change accounting failed: {exc}")
                continue
            for field, value in changes.items():
                actual_changes[field] += value

    return {
        "can_replay": can_replay,
        "selected_tensors": selected_tensors,
        "selected_params": selected_params,
        "total_tensors": total_tensors,
        "total_params": total_params,
        "checked_tensors": checked_tensors,
        "out_of_scope_tensors": out_of_scope_tensors,
        "out_of_scope_bytes_checked": out_of_scope_bytes_checked,
        "selected_entries": selected_entries,
        "tensor_bytes": tensor_bytes,
        "actual_changes": actual_changes,
    }


def _collect_transformation_inputs(
    owner: Any,
    *,
    artifact_dir: Path,
    baseline_dir: Path,
    edit_type: str,
    parameters: Mapping[str, object],
    scope: str,
    materialization_receipt_name: str,
    issues: list[str],
) -> dict[str, Any]:
    spec: dict[str, object] | None = None
    normalized_scope: str | None = None
    oracle: TransformationOracle | None = None

    for label, checkpoint in (("baseline", baseline_dir), ("artifact", artifact_dir)):
        try:
            require_regular_checkpoint_tree(checkpoint, label=f"{label} checkpoint")
        except CheckpointLayoutError as exc:
            issues.append(f"{label}: {exc}")

    baseline_contract: Any | None = None
    artifact_contract: Any | None = None
    try:
        oracle = build_transformation_oracle(
            baseline_dir,
            edit_type=edit_type,
            parameters=parameters,
            scope=scope,
        )
        spec = oracle.spec
        normalized_scope = oracle.normalized_scope
        baseline_contract = oracle.contract
    except TransformationOracleError as exc:
        issues.append(f"baseline: {exc}")
    try:
        artifact_contract = checkpoint_contract(artifact_dir)
    except TransformationOracleError as exc:
        issues.append(f"artifact: {exc}")
    if (
        baseline_contract is not None
        and artifact_contract is not None
        and artifact_contract != baseline_contract
    ):
        issues.append("artifact transformation configuration does not match baseline")

    expected_edit_type = (
        str(spec["edit_type"])
        if spec is not None and isinstance(spec.get("edit_type"), str)
        else edit_type
    )
    artifact_result = owner.validate_edit_artifact(
        artifact_dir,
        require_metadata=True,
        expected_edit_type=expected_edit_type,
        expected_artifact_class="validation_subject_checkpoint",
    )
    issues.extend(artifact_result.issues or [])

    metadata_path = artifact_dir / "edit_metadata.json"
    metadata, metadata_issues = _strict_json_object(
        metadata_path,
        label="edit_metadata.json",
    )
    issues.extend(metadata_issues)

    baseline_map, baseline_index, baseline_map_issues = _strict_safetensor_weight_map(
        baseline_dir,
        label="baseline checkpoint",
    )
    artifact_map, artifact_index, artifact_map_issues = _strict_safetensor_weight_map(
        artifact_dir,
        label="artifact checkpoint",
    )
    issues.extend(baseline_map_issues)
    issues.extend(artifact_map_issues)
    if artifact_index is None:
        issues.append("artifact transformation output must use a safetensors index")

    baseline_keys = set(baseline_map)
    artifact_keys = set(artifact_map)
    missing_tensors = sorted(baseline_keys - artifact_keys)
    extra_tensors = sorted(artifact_keys - baseline_keys)
    if missing_tensors:
        issues.append(f"artifact missing tensors: {missing_tensors[:5]}")
    if extra_tensors:
        issues.append(f"artifact has unexpected tensors: {extra_tensors[:5]}")

    generated_files = frozenset(
        {
            "edit_metadata.json",
            "model.safetensors.index.json",
            materialization_receipt_name,
        }
    )
    for generated_file in ("edit_metadata.json", materialization_receipt_name):
        if (baseline_dir / generated_file).exists():
            issues.append(
                f"baseline checkpoint must not contain generated transformation file: "
                f"{generated_file}"
            )
    baseline_support, baseline_support_issues = _support_file_digests(
        baseline_dir,
        weight_paths=set(baseline_map.values()),
        generated_files=generated_files,
    )
    artifact_support, artifact_support_issues = _support_file_digests(
        artifact_dir,
        weight_paths=set(artifact_map.values()),
        generated_files=generated_files,
    )
    issues.extend(f"baseline: {issue}" for issue in baseline_support_issues)
    issues.extend(f"artifact: {issue}" for issue in artifact_support_issues)
    missing_support = sorted(set(baseline_support) - set(artifact_support))
    extra_support = sorted(set(artifact_support) - set(baseline_support))
    if missing_support:
        issues.append(f"artifact missing support files: {missing_support[:5]}")
    if extra_support:
        issues.append(f"artifact has unexpected support files: {extra_support[:5]}")
    for relative in sorted(set(baseline_support) & set(artifact_support)):
        if baseline_support[relative] != artifact_support[relative]:
            issues.append(f"support file changed: {relative}")

    baseline_identity: dict[str, str] | None = None
    artifact_identity: dict[str, str] | None = None
    try:
        baseline_identity = {
            "kind": "local_checkpoint_tree",
            "sha256": owner.checkpoint_tree_sha256(baseline_dir),
        }
    except (OSError, ValueError) as exc:
        issues.append(f"baseline identity unavailable: {exc}")
    try:
        artifact_identity = {
            "kind": "local_checkpoint_tree",
            "sha256": owner.checkpoint_tree_sha256(artifact_dir),
        }
    except (OSError, ValueError) as exc:
        issues.append(f"artifact identity unavailable: {exc}")

    receipt_path = artifact_dir / materialization_receipt_name
    receipt, receipt_issues = _strict_json_object(
        receipt_path,
        label=materialization_receipt_name,
    )
    issues.extend(receipt_issues)
    receipt_sha256 = _file_sha256(receipt_path) if receipt_path.is_file() else None
    metadata_sha256 = _file_sha256(metadata_path) if metadata_path.is_file() else None

    return {
        "spec": spec,
        "normalized_scope": normalized_scope,
        "oracle": oracle,
        "baseline_contract": baseline_contract,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "baseline_map": baseline_map,
        "artifact_map": artifact_map,
        "baseline_index": baseline_index,
        "artifact_index": artifact_index,
        "baseline_map_issues": baseline_map_issues,
        "artifact_map_issues": artifact_map_issues,
        "baseline_keys": baseline_keys,
        "artifact_keys": artifact_keys,
        "baseline_support": baseline_support,
        "baseline_identity": baseline_identity,
        "artifact_identity": artifact_identity,
        "receipt": receipt,
        "receipt_sha256": receipt_sha256,
        "metadata_sha256": metadata_sha256,
    }


def _validate_bound_contracts(
    *,
    artifact_dir: Path,
    baseline_dir: Path,
    baseline_map: dict[str, Path],
    artifact_map: dict[str, Path],
    artifact_index: Path | None,
    baseline_keys: set[str],
    tensor_bytes: dict[str, int],
    receipt: object,
    metadata: object,
    spec: dict[str, object] | None,
    normalized_scope: str | None,
    baseline_contract: Any | None,
    target_manifest: dict[str, object] | None,
    target_manifest_sha256: str | None,
    selected_tensors: int,
    selected_params: int,
    total_tensors: int,
    total_params: int,
    actual_changes: dict[str, int],
    execution_policy: str,
    materialization_receipt_schema: str,
    baseline_identity: dict[str, str] | None,
    issues: list[str],
) -> dict[str, Any]:
    source_shard_plan: dict[str, object] | None = None
    source_shard_plan_sha256: str | None = None
    if baseline_keys and set(tensor_bytes) == baseline_keys:
        source_shard_plan = _source_shard_plan(
            baseline_dir,
            weight_map=baseline_map,
            tensor_bytes=tensor_bytes,
        )
        source_shard_plan_sha256 = _canonical_json_sha256(source_shard_plan)
    output_weights = _output_weight_identity(
        artifact_dir,
        weight_map=artifact_map,
        index_path=artifact_index,
    )

    max_output_shard_bytes: int | None = None
    output_shard_plan: dict[str, object] | None = None
    output_shard_plan_sha256: str | None = None
    expected_output_weight_map: dict[str, str] | None = None
    if isinstance(receipt, Mapping):
        raw_max_output_shard_bytes = receipt.get("max_output_shard_bytes")
        if (
            isinstance(raw_max_output_shard_bytes, bool)
            or not isinstance(raw_max_output_shard_bytes, int)
            or raw_max_output_shard_bytes < 1024 * 1024
        ):
            issues.append(
                "transformation materialization receipt max_output_shard_bytes is invalid"
            )
        else:
            max_output_shard_bytes = raw_max_output_shard_bytes
    if (
        source_shard_plan is not None
        and source_shard_plan_sha256 is not None
        and target_manifest_sha256 is not None
        and max_output_shard_bytes is not None
    ):
        output_shard_plan, expected_output_weight_map = _expected_output_shard_plan(
            baseline_dir,
            weight_map=baseline_map,
            tensor_bytes=tensor_bytes,
            source_shard_plan_sha256=source_shard_plan_sha256,
            target_manifest_sha256=target_manifest_sha256,
            max_output_shard_bytes=max_output_shard_bytes,
        )
        output_shard_plan_sha256 = _canonical_json_sha256(output_shard_plan)
        actual_output_weight_map = {
            name: path.relative_to(artifact_dir).as_posix()
            for name, path in artifact_map.items()
        }
        if actual_output_weight_map != expected_output_weight_map:
            issues.append(
                "artifact safetensors index does not match canonical output shard plan"
            )

    if (
        metadata is not None
        and spec is not None
        and normalized_scope is not None
        and baseline_contract is not None
        and target_manifest is not None
        and target_manifest_sha256 is not None
        and max_output_shard_bytes is not None
        and source_shard_plan is not None
        and source_shard_plan_sha256 is not None
        and output_shard_plan is not None
        and output_shard_plan_sha256 is not None
    ):
        issues.extend(
            _transformation_metadata_errors(
                metadata,
                spec=spec,
                scope=normalized_scope,
                contract=baseline_contract,
                target_manifest=target_manifest,
                target_manifest_sha256=target_manifest_sha256,
                max_output_shard_bytes=max_output_shard_bytes,
                source_shard_plan=source_shard_plan,
                source_shard_plan_sha256=source_shard_plan_sha256,
                output_shard_plan=output_shard_plan,
                output_shard_plan_sha256=output_shard_plan_sha256,
                selected_tensors=selected_tensors,
                selected_params=selected_params,
                total_params=total_params,
                actual_changes=actual_changes,
                execution_policy=execution_policy,
            )
        )
    if (
        isinstance(receipt, Mapping)
        and spec is not None
        and normalized_scope is not None
        and baseline_contract is not None
        and target_manifest is not None
        and target_manifest_sha256 is not None
        and max_output_shard_bytes is not None
        and source_shard_plan is not None
        and source_shard_plan_sha256 is not None
        and output_shard_plan is not None
        and output_shard_plan_sha256 is not None
    ):
        issues.extend(
            _transformation_receipt_errors(
                receipt,
                receipt_schema=materialization_receipt_schema,
                baseline_identity=baseline_identity,
                spec=spec,
                scope=normalized_scope,
                contract=baseline_contract,
                target_manifest=target_manifest,
                target_manifest_sha256=target_manifest_sha256,
                max_output_shard_bytes=max_output_shard_bytes,
                source_shard_plan=source_shard_plan,
                source_shard_plan_sha256=source_shard_plan_sha256,
                output_shard_plan=output_shard_plan,
                output_shard_plan_sha256=output_shard_plan_sha256,
                output_weights=output_weights,
                selected_tensors=selected_tensors,
                selected_params=selected_params,
                total_tensors=total_tensors,
                total_params=total_params,
                actual_changes=actual_changes,
                execution_policy=execution_policy,
            )
        )

    return {
        "source_shard_plan": source_shard_plan,
        "source_shard_plan_sha256": source_shard_plan_sha256,
        "output_weights": output_weights,
        "max_output_shard_bytes": max_output_shard_bytes,
        "output_shard_plan": output_shard_plan,
        "output_shard_plan_sha256": output_shard_plan_sha256,
    }


def validate_transformation_artifact(
    owner: Any,
    artifact_dir: Path,
    *,
    baseline_dir: Path,
    edit_type: str,
    parameters: Mapping[str, object],
    scope: str,
) -> dict[str, Any]:
    """Replay a verifier-grade generated transformation from its baseline.

    The numerical and target oracle is intentionally independent of the
    materializer: it neither imports the generator nor its target resolver.
    It rereads every baseline and artifact tensor, rejects alternate checkpoint
    topologies, and binds the replay result to immutable trees.  The resulting
    payload is the only form suitable for a later evidence-pack sidecar;
    callers must persist it outside either checkpoint tree.
    """

    issues: list[str] = []
    materialization_receipt_name = TRANSFORMATION_MATERIALIZATION_RECEIPT
    materialization_receipt_schema = TRANSFORMATION_MATERIALIZATION_RECEIPT_SCHEMA
    execution_policy = CANONICAL_EXECUTION_POLICY
    inputs = _collect_transformation_inputs(
        owner,
        artifact_dir=artifact_dir,
        baseline_dir=baseline_dir,
        edit_type=edit_type,
        parameters=parameters,
        scope=scope,
        materialization_receipt_name=materialization_receipt_name,
        issues=issues,
    )
    spec = inputs["spec"]
    normalized_scope = inputs["normalized_scope"]
    oracle = inputs["oracle"]
    baseline_contract = inputs["baseline_contract"]
    metadata = inputs["metadata"]
    baseline_map = inputs["baseline_map"]
    artifact_map = inputs["artifact_map"]
    artifact_index = inputs["artifact_index"]
    baseline_map_issues = inputs["baseline_map_issues"]
    artifact_map_issues = inputs["artifact_map_issues"]
    baseline_keys = inputs["baseline_keys"]
    baseline_support = inputs["baseline_support"]
    baseline_identity = inputs["baseline_identity"]
    artifact_identity = inputs["artifact_identity"]
    receipt = inputs["receipt"]
    receipt_sha256 = inputs["receipt_sha256"]
    metadata_sha256 = inputs["metadata_sha256"]

    replay = _replay_tensors(
        spec=spec,
        normalized_scope=normalized_scope,
        baseline_contract=baseline_contract,
        oracle=oracle,
        baseline_map=baseline_map,
        artifact_map=artifact_map,
        baseline_map_issues=baseline_map_issues,
        artifact_map_issues=artifact_map_issues,
        issues=issues,
    )
    can_replay = replay["can_replay"]
    selected_tensors = replay["selected_tensors"]
    selected_params = replay["selected_params"]
    total_tensors = replay["total_tensors"]
    total_params = replay["total_params"]
    checked_tensors = replay["checked_tensors"]
    out_of_scope_tensors = replay["out_of_scope_tensors"]
    out_of_scope_bytes_checked = replay["out_of_scope_bytes_checked"]
    selected_entries = replay["selected_entries"]
    tensor_bytes = replay["tensor_bytes"]
    actual_changes = replay["actual_changes"]

    target_manifest: dict[str, object] | None = None
    target_manifest_sha256: str | None = None
    if (
        spec is not None
        and normalized_scope is not None
        and oracle is not None
        and selected_entries
    ):
        try:
            target_manifest = oracle.target_manifest(selected_entries)
            target_manifest_sha256 = oracle.target_manifest_sha256(selected_entries)
        except TransformationOracleError as exc:
            issues.append(f"independent transformation target manifest invalid: {exc}")

    if can_replay and selected_tensors <= 0:
        issues.append("no tensors matched transformation scope")
    if can_replay and (
        actual_changes["value_changed_params"] <= 0
        or actual_changes["byte_changed_params"] <= 0
        or actual_changes["value_changed_tensors"] <= 0
        or actual_changes["byte_changed_tensors"] <= 0
    ):
        issues.append(
            "transformation replay observed no effective value and byte changes"
        )

    bound = _validate_bound_contracts(
        artifact_dir=artifact_dir,
        baseline_dir=baseline_dir,
        baseline_map=baseline_map,
        artifact_map=artifact_map,
        artifact_index=artifact_index,
        baseline_keys=baseline_keys,
        tensor_bytes=tensor_bytes,
        receipt=receipt,
        metadata=metadata,
        spec=spec,
        normalized_scope=normalized_scope,
        baseline_contract=baseline_contract,
        target_manifest=target_manifest,
        target_manifest_sha256=target_manifest_sha256,
        selected_tensors=selected_tensors,
        selected_params=selected_params,
        total_tensors=total_tensors,
        total_params=total_params,
        actual_changes=actual_changes,
        execution_policy=execution_policy,
        materialization_receipt_schema=materialization_receipt_schema,
        baseline_identity=baseline_identity,
        issues=issues,
    )
    source_shard_plan = bound["source_shard_plan"]
    source_shard_plan_sha256 = bound["source_shard_plan_sha256"]
    output_weights = bound["output_weights"]
    max_output_shard_bytes = bound["max_output_shard_bytes"]
    output_shard_plan = bound["output_shard_plan"]
    output_shard_plan_sha256 = bound["output_shard_plan_sha256"]

    # The initial identities bind the receipt to the exact input trees that
    # replay began with.  Hash them again after every tensor and sidecar read
    # so a concurrent replacement cannot turn a valid result into a sidecar
    # for a different checkpoint.
    final_baseline_identity: dict[str, str] | None = None
    final_artifact_identity: dict[str, str] | None = None
    try:
        final_baseline_identity = {
            "kind": "local_checkpoint_tree",
            "sha256": owner.checkpoint_tree_sha256(baseline_dir),
        }
    except (OSError, ValueError) as exc:
        issues.append(f"baseline final identity unavailable: {exc}")
    try:
        final_artifact_identity = {
            "kind": "local_checkpoint_tree",
            "sha256": owner.checkpoint_tree_sha256(artifact_dir),
        }
    except (OSError, ValueError) as exc:
        issues.append(f"artifact final identity unavailable: {exc}")
    if (
        baseline_identity is not None
        and final_baseline_identity is not None
        and final_baseline_identity != baseline_identity
    ):
        issues.append(
            "baseline checkpoint changed during transformation replay validation"
        )
    if (
        artifact_identity is not None
        and final_artifact_identity is not None
        and final_artifact_identity != artifact_identity
    ):
        issues.append(
            "artifact checkpoint changed during transformation replay validation"
        )
    if final_baseline_identity is not None:
        baseline_identity = final_baseline_identity
    if final_artifact_identity is not None:
        artifact_identity = final_artifact_identity

    return {
        "schema": TRANSFORMATION_REPLAY_SCHEMA,
        "ok": not issues,
        "edit_type": spec.get("edit_type") if spec is not None else edit_type,
        "transformation": spec,
        "algorithm": spec.get("algorithm") if spec is not None else None,
        "parameters": spec.get("parameters") if spec is not None else None,
        "scope": normalized_scope if normalized_scope is not None else scope,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY_VERSION,
        "model_type": baseline_contract.model_type if baseline_contract else None,
        "architecture": baseline_contract.architecture if baseline_contract else None,
        "config_sha256": baseline_contract.config_sha256 if baseline_contract else None,
        "layer_count": baseline_contract.layer_count if baseline_contract else None,
        "target_manifest": target_manifest,
        "target_manifest_sha256": target_manifest_sha256,
        "baseline_identity": baseline_identity,
        "artifact_identity": artifact_identity,
        "materialization_receipt_sha256": receipt_sha256,
        "edit_metadata_sha256": metadata_sha256,
        "max_output_shard_bytes": max_output_shard_bytes,
        "source_shard_plan": source_shard_plan,
        "source_shard_plan_sha256": source_shard_plan_sha256,
        "output_shard_plan": output_shard_plan,
        "output_shard_plan_sha256": output_shard_plan_sha256,
        "output_weights": output_weights,
        "execution_policy": execution_policy or None,
        "checked_tensors": checked_tensors,
        "selected_tensors": selected_tensors,
        "selected_params": selected_params,
        "total_tensors": total_tensors,
        "total_params": total_params,
        "actual_changes": actual_changes,
        "out_of_scope_tensors_checked": out_of_scope_tensors,
        "out_of_scope_bytes_checked": out_of_scope_bytes_checked,
        "support_files_checked": len(baseline_support),
        "issues": issues,
    }
