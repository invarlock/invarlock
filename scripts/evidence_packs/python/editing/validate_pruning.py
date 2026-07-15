"""Exact pruning receipt and storage-replay validation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from invarlock.pruning_contract import (
    PRUNING_ALGORITHM,
    PRUNING_REPLAY_SCHEMA,
    PRUNING_SCOPE_POLICY_VERSION,
    PRUNING_STORAGE_POLICY,
    PruningCheckpointContract,
    PruningContractError,
    checkpoint_pruning_contract,
    finite_pruning_sparsity,
    pruning_target_manifest,
    pruning_target_manifest_sha256,
    validate_pruning_scope,
)

from .artifact_tensor_validation import (
    _bounded_pruning_replay_settings,
    _pruning_replay_one_tensor,
    _PruningReplayResult,
    _safetensor_weight_map,
    _support_file_digests,
)
from .checkpoint_paths import CheckpointLayoutError, require_regular_checkpoint_tree
from .implementations import read_edit_metadata
from .validate_deployable import _load_json_object, _valid_digest

PRUNING_MATERIALIZATION_RECEIPT_SCHEMA = "invarlock/pruning-materialization-v1"
PRUNING_MATERIALIZATION_RECEIPT = "pruning_materialization.json"


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _read_pruning_metadata(
    artifact_dir: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        metadata = read_edit_metadata(artifact_dir / "edit_metadata.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, [f"edit_metadata.json invalid: {exc}"]
    return metadata, []


def _pruning_metadata_errors(
    metadata: dict[str, Any],
    *,
    scope: str,
    target_sparsity: float,
    contract: PruningCheckpointContract,
    target_manifest: dict[str, object],
    target_manifest_sha256: str,
    selected_tensors: int,
    selected_params: int,
    expected_pruned_params: int,
    expected_changed_params: int,
    observed_changed_params: int,
    observed_zero_params: int,
    total_params: int,
) -> list[str]:
    errors: list[str] = []
    if metadata.get("scope") != scope:
        errors.append("edit_metadata.scope does not match pruning replay scope")
    parameters = metadata.get("parameters")
    metadata_sparsity = (
        parameters.get("target_sparsity") if isinstance(parameters, dict) else None
    )
    if (
        not isinstance(metadata_sparsity, int | float)
        or isinstance(metadata_sparsity, bool)
        or not math.isfinite(float(metadata_sparsity))
        or abs(float(metadata_sparsity) - target_sparsity) > 1e-12
    ):
        errors.append("edit_metadata.target_sparsity does not match pruning replay")
    required_metadata = {
        "scope_policy": PRUNING_SCOPE_POLICY_VERSION,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "model_type": contract.model_type,
        "pruning_architecture": contract.architecture,
        "config_sha256": contract.config_sha256,
        "target_manifest_sha256": target_manifest_sha256,
    }
    for field, expected in required_metadata.items():
        if metadata.get(field) != expected:
            errors.append(f"edit_metadata.{field} does not match pruning contract")
    if not _valid_digest(metadata.get("shard_plan_sha256")):
        errors.append("edit_metadata.shard_plan_sha256 is invalid")
    raw_manifest = metadata.get("target_manifest")
    if not isinstance(raw_manifest, dict):
        errors.append("edit_metadata.target_manifest must be an object")
    else:
        try:
            metadata_manifest_sha256 = pruning_target_manifest_sha256(raw_manifest)
        except PruningContractError as exc:
            errors.append(f"edit_metadata.target_manifest invalid: {exc}")
        else:
            if metadata_manifest_sha256 != target_manifest_sha256:
                errors.append("edit_metadata.target_manifest does not match replay")
            if raw_manifest != target_manifest:
                errors.append(
                    "edit_metadata.target_manifest content does not match replay"
                )

    coverage = metadata.get("coverage")
    if not isinstance(coverage, dict):
        errors.append("edit_metadata.coverage must be an object")
    else:
        expected_coverage = {
            "edited_tensors": selected_tensors,
            "edited_params": selected_params,
            "total_params": total_params,
        }
        for field, expected in expected_coverage.items():
            value = coverage.get(field)
            if not _is_nonnegative_int(value):
                errors.append(
                    f"edit_metadata.coverage.{field} must be a non-negative int"
                )
            elif value != expected:
                errors.append(f"edit_metadata.coverage.{field} does not match replay")
        ratio = _finite_number(coverage.get("coverage_ratio"))
        expected_ratio = selected_params / total_params if total_params else 0.0
        if ratio is None or ratio != expected_ratio:
            errors.append("edit_metadata.coverage.coverage_ratio does not match replay")

    expected_extras = {
        "effective_changed_params": observed_changed_params,
    }
    for field, expected in expected_extras.items():
        if metadata.get(field) != expected:
            errors.append(f"edit_metadata.{field} does not match replay")
    if observed_changed_params != expected_changed_params:
        errors.append(
            "replay changed parameter count does not match exact transformation"
        )
    mask_sparsity = _finite_number(metadata.get("mask_sparsity"))
    expected_mask_sparsity = expected_pruned_params / selected_params
    if mask_sparsity is None or not math.isclose(
        mask_sparsity, expected_mask_sparsity, abs_tol=1e-12
    ):
        errors.append("edit_metadata.mask_sparsity does not match replay")
    actual_zero_fraction = _finite_number(metadata.get("actual_zero_fraction"))
    expected_zero_fraction = observed_zero_params / selected_params
    if actual_zero_fraction is None or not math.isclose(
        actual_zero_fraction, expected_zero_fraction, abs_tol=1e-12
    ):
        errors.append("edit_metadata.actual_zero_fraction does not match replay")
    return errors


def _pruning_materialization_receipt_errors(
    artifact_dir: Path,
    *,
    baseline_identity: dict[str, str] | None,
    contract: PruningCheckpointContract,
    scope: str,
    target_sparsity: float,
    target_manifest_sha256: str,
    metadata_shard_plan_sha256: str | None,
    selected_tensors: int,
    selected_params: int,
    expected_pruned_params: int,
    original_zero_params: int,
    observed_zero_params: int,
    observed_changed_params: int,
    total_params: int,
) -> list[str]:
    receipt_path = artifact_dir / PRUNING_MATERIALIZATION_RECEIPT
    if not receipt_path.is_file():
        return [f"{PRUNING_MATERIALIZATION_RECEIPT} missing"]
    payload = _load_json_object(receipt_path)
    if payload is None:
        return [f"{PRUNING_MATERIALIZATION_RECEIPT} is invalid"]
    errors: list[str] = []
    expected_values: dict[str, object] = {
        "schema": PRUNING_MATERIALIZATION_RECEIPT_SCHEMA,
        "ok": True,
        "baseline_identity": baseline_identity,
        "scope": scope,
        "target_sparsity": target_sparsity,
        "scope_policy": PRUNING_SCOPE_POLICY_VERSION,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "model_type": contract.model_type,
        "architecture": contract.architecture,
        "config_sha256": contract.config_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "selected_tensors": selected_tensors,
        "selected_params": selected_params,
        "expected_pruned_params": expected_pruned_params,
        "original_zero_params": original_zero_params,
        "observed_zero_params": observed_zero_params,
        "effective_changed_params": observed_changed_params,
        "total_params": total_params,
    }
    for field, expected in expected_values.items():
        if payload.get(field) != expected:
            errors.append(f"materialization receipt {field} does not match replay")
    for field in ("output_shards", "resume_count"):
        if not _is_nonnegative_int(payload.get(field)):
            errors.append(f"materialization receipt {field} must be a non-negative int")
    if payload.get("output_shards") == 0:
        errors.append("materialization receipt output_shards must be positive")
    receipt_shard_plan_sha256 = payload.get("shard_plan_sha256")
    if not _valid_digest(receipt_shard_plan_sha256):
        errors.append("materialization receipt shard_plan_sha256 is invalid")
    elif receipt_shard_plan_sha256 != metadata_shard_plan_sha256:
        errors.append(
            "materialization receipt shard_plan_sha256 does not match metadata"
        )
    return errors


def _pruning_receipt_preflight_errors(
    artifact_dir: Path,
    *,
    contract: PruningCheckpointContract | None,
    scope: str,
    target_sparsity: float,
) -> list[str]:
    """Reject retired or mismatched receipt contracts before tensor replay."""

    receipt_path = artifact_dir / PRUNING_MATERIALIZATION_RECEIPT
    if not receipt_path.is_file():
        # The complete validator reports absence after replay. Keeping replay
        # active here preserves adversarial diagnostics for copied metadata or
        # unchanged checkpoints that omit the receipt entirely.
        return []
    payload = _load_json_object(receipt_path)
    if payload is None:
        return [f"{PRUNING_MATERIALIZATION_RECEIPT} is invalid"]
    expected: dict[str, object] = {
        "schema": PRUNING_MATERIALIZATION_RECEIPT_SCHEMA,
        "ok": True,
        "scope": scope,
        "target_sparsity": target_sparsity,
        "scope_policy": PRUNING_SCOPE_POLICY_VERSION,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
    }
    if contract is not None:
        expected.update(
            {
                "model_type": contract.model_type,
                "architecture": contract.architecture,
                "config_sha256": contract.config_sha256,
            }
        )
    return [
        f"materialization receipt {field} does not match pruning preflight"
        for field, value in expected.items()
        if payload.get(field) != value
    ]


def validate_pruning_artifact(
    owner: Any,
    artifact_dir: Path,
    *,
    baseline_dir: Path,
    scope: str,
    target_sparsity: float,
    workers: int = 1,
    worker_threads: int = 0,
) -> dict[str, Any]:
    import torch

    issues: list[str] = []
    try:
        replay_workers, replay_worker_threads = _bounded_pruning_replay_settings(
            workers=workers, worker_threads=worker_threads
        )
    except ValueError as exc:
        replay_workers, replay_worker_threads = 1, 1
        issues.append(str(exc))
    for label, checkpoint in (("baseline", baseline_dir), ("artifact", artifact_dir)):
        try:
            require_regular_checkpoint_tree(checkpoint, label=f"{label} checkpoint")
        except CheckpointLayoutError as exc:
            issues.append(f"{label}: {exc}")
    try:
        normalized_scope = validate_pruning_scope(scope)
    except PruningContractError as exc:
        normalized_scope = str(scope)
        issues.append(str(exc))
    try:
        normalized_sparsity = finite_pruning_sparsity(target_sparsity)
    except PruningContractError as exc:
        normalized_sparsity = float("nan")
        issues.append(str(exc))
    baseline_contract: PruningCheckpointContract | None = None
    artifact_contract: PruningCheckpointContract | None = None
    try:
        baseline_contract = checkpoint_pruning_contract(baseline_dir)
    except PruningContractError as exc:
        issues.append(f"baseline: {exc}")
    try:
        artifact_contract = checkpoint_pruning_contract(artifact_dir)
    except PruningContractError as exc:
        issues.append(f"artifact: {exc}")
    if baseline_contract is not None and artifact_contract is not None:
        if artifact_contract != baseline_contract:
            issues.append("artifact pruning configuration does not match baseline")
    issues.extend(
        _pruning_receipt_preflight_errors(
            artifact_dir,
            contract=baseline_contract,
            scope=normalized_scope,
            target_sparsity=normalized_sparsity,
        )
    )
    artifact_result = owner.validate_edit_artifact(
        artifact_dir,
        require_metadata=True,
        expected_edit_type="magnitude_prune",
        expected_artifact_class="validation_subject_checkpoint",
    )
    issues.extend(artifact_result.issues or [])
    metadata, metadata_errors = _read_pruning_metadata(artifact_dir)
    issues.extend(metadata_errors)

    baseline_map, baseline_issues = _safetensor_weight_map(baseline_dir)
    artifact_map, artifact_issues = _safetensor_weight_map(artifact_dir)
    issues.extend(f"baseline: {issue}" for issue in baseline_issues)
    issues.extend(f"artifact: {issue}" for issue in artifact_issues)
    baseline_keys = set(baseline_map)
    artifact_keys = set(artifact_map)
    missing = sorted(baseline_keys - artifact_keys)
    extra = sorted(artifact_keys - baseline_keys)
    if missing:
        issues.append(f"artifact missing tensors: {missing[:5]}")
    if extra:
        issues.append(f"artifact has unexpected tensors: {extra[:5]}")

    baseline_support, baseline_support_issues = _support_file_digests(
        baseline_dir,
        weight_paths=set(baseline_map.values()),
    )
    artifact_support, artifact_support_issues = _support_file_digests(
        artifact_dir,
        weight_paths=set(artifact_map.values()),
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
            if len(issues) >= 20:
                break

    selected_tensors = 0
    selected_params = 0
    expected_pruned_params = 0
    expected_changed_params = 0
    observed_changed_params = 0
    original_zero_params = 0
    observed_zero_params = 0
    total_params = 0
    out_of_scope_tensors = 0
    out_of_scope_bytes_checked = 0
    checked_tensors = 0
    selected_entries: list[dict[str, object]] = []
    if not issues and baseline_contract is not None:
        keys = sorted(baseline_keys)
        original_threads = torch.get_num_threads()
        active_threads = replay_worker_threads or original_threads
        try:
            torch.set_num_threads(active_threads)

            def replay(key: str) -> _PruningReplayResult:
                return _pruning_replay_one_tensor(
                    key=key,
                    baseline_path=baseline_map[key],
                    artifact_path=artifact_map[key],
                    scope=normalized_scope,
                    sparsity=normalized_sparsity,
                    contract=baseline_contract,
                )

            executor: ThreadPoolExecutor | None = None
            replay_results: Iterable[_PruningReplayResult]
            if replay_workers == 1:
                replay_results = map(replay, keys)
            else:
                executor = ThreadPoolExecutor(
                    max_workers=replay_workers,
                    thread_name_prefix="pruning-replay",
                )
                replay_results = executor.map(replay, keys)
            try:
                for replayed in replay_results:
                    checked_tensors += replayed.checked_tensors
                    total_params += replayed.total_params
                    selected_tensors += replayed.selected_tensors
                    selected_params += replayed.selected_params
                    expected_pruned_params += replayed.expected_pruned_params
                    expected_changed_params += replayed.expected_changed_params
                    observed_changed_params += replayed.observed_changed_params
                    original_zero_params += replayed.original_zero_params
                    observed_zero_params += replayed.observed_zero_params
                    out_of_scope_tensors += replayed.out_of_scope_tensors
                    out_of_scope_bytes_checked += replayed.out_of_scope_bytes_checked
                    if replayed.selected_entry is not None:
                        selected_entries.append(replayed.selected_entry)
                    if replayed.issue is not None:
                        issues.append(replayed.issue)
            except Exception as exc:  # noqa: BLE001 - worker errors fail closed
                issues.append(
                    "pruning replay worker failed: "
                    f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
        finally:
            torch.set_num_threads(original_threads)

    if not issues and selected_tensors <= 0:
        issues.append("no tensors matched pruning scope")
    if not issues and (expected_pruned_params <= 0 or expected_changed_params <= 0):
        issues.append("pruning replay selected no effective parameter changes")

    target_manifest: dict[str, object] | None = None
    target_manifest_sha256: str | None = None
    if baseline_contract is not None and selected_entries:
        try:
            target_manifest = pruning_target_manifest(
                scope=normalized_scope,
                contract=baseline_contract,
                targets=selected_entries,
            )
            target_manifest_sha256 = pruning_target_manifest_sha256(target_manifest)
        except PruningContractError as exc:
            issues.append(f"pruning target manifest invalid: {exc}")

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

    if (
        metadata is not None
        and baseline_contract is not None
        and target_manifest is not None
        and target_manifest_sha256 is not None
    ):
        issues.extend(
            _pruning_metadata_errors(
                metadata,
                scope=normalized_scope,
                target_sparsity=normalized_sparsity,
                contract=baseline_contract,
                target_manifest=target_manifest,
                target_manifest_sha256=target_manifest_sha256,
                selected_tensors=selected_tensors,
                selected_params=selected_params,
                expected_pruned_params=expected_pruned_params,
                expected_changed_params=expected_changed_params,
                observed_changed_params=observed_changed_params,
                observed_zero_params=observed_zero_params,
                total_params=total_params,
            )
        )
        issues.extend(
            _pruning_materialization_receipt_errors(
                artifact_dir,
                baseline_identity=baseline_identity,
                contract=baseline_contract,
                scope=normalized_scope,
                target_sparsity=normalized_sparsity,
                target_manifest_sha256=target_manifest_sha256,
                metadata_shard_plan_sha256=metadata.get("shard_plan_sha256"),
                selected_tensors=selected_tensors,
                selected_params=selected_params,
                expected_pruned_params=expected_pruned_params,
                original_zero_params=original_zero_params,
                observed_zero_params=observed_zero_params,
                observed_changed_params=observed_changed_params,
                total_params=total_params,
            )
        )

    return {
        "schema": PRUNING_REPLAY_SCHEMA,
        "ok": not issues,
        "edit_type": "magnitude_prune",
        "scope": normalized_scope,
        "target_sparsity": normalized_sparsity,
        "scope_policy": PRUNING_SCOPE_POLICY_VERSION,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "model_type": baseline_contract.model_type if baseline_contract else None,
        "architecture": baseline_contract.architecture if baseline_contract else None,
        "config_sha256": baseline_contract.config_sha256 if baseline_contract else None,
        "target_manifest": target_manifest,
        "target_manifest_sha256": target_manifest_sha256,
        "baseline_identity": baseline_identity,
        "artifact_identity": artifact_identity,
        "checked_tensors": checked_tensors,
        "selected_tensors": selected_tensors,
        "selected_params": selected_params,
        "total_params": total_params,
        "expected_pruned_params": expected_pruned_params,
        "expected_changed_params": expected_changed_params,
        "observed_changed_params": observed_changed_params,
        "original_zero_params": original_zero_params,
        "observed_zero_params": observed_zero_params,
        "out_of_scope_tensors_checked": out_of_scope_tensors,
        "out_of_scope_bytes_checked": out_of_scope_bytes_checked,
        "support_files_checked": len(baseline_support),
        "issues": issues,
    }
