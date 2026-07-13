"""Shared edit-metadata schemas, scenario lookup, and JSON helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_json import (
    StrictJsonError,
    load_json,
)
from invarlock.evidence_pack_scenario_contract import (
    ScenarioContract,
    ScenarioContractError,
    parse_scenario_contract,
)

EDIT_METADATA_SCHEMA = "invarlock/evidence-pack-edit-metadata-v1"
VALIDATION_SUBJECT_CHECKPOINT = "validation_subject_checkpoint"
DEPLOYABLE_OPTIMIZED_SUBJECT = "deployable_optimized_subject"
FAULT_INJECTION_FIXTURE = "fault_injection_fixture"
PRUNING_REPLAY_SIDECAR = "pruning_replay.json"
PRUNING_SELECTION_SOURCE_PATH = "metadata/clean_pruning_selection/bundle.json"
TRANSFORMATION_REPLAY_SIDECAR = "transformation_replay.json"
TRANSFORMATION_REPLAY_SCHEMA = "invarlock/generated-transformation-replay-v1"
TRANSFORMATION_MATERIALIZATION_RECEIPT = "transformation_materialization.json"
TRANSFORMATION_MATERIALIZATION_RECEIPT_SCHEMA = (
    "invarlock/transformation-materialization-v1"
)
TRANSFORMATION_TARGET_MANIFEST_SCHEMA = "invarlock/transformation-target-manifest-v1"
TRANSFORMATION_SELECTION_RECEIPT_SCHEMA = (
    "invarlock/generated-transformation-selection-v1"
)
TRANSFORMATION_SELECTION_SOURCE_PATH = "metadata/clean_selection/selection_bundle.json"
TRANSFORMATION_CONTRACT_VERSION = "verifier-grade-transformation-v1"
TRANSFORMATION_PARAMETERS_SCHEMA = "invarlock/transformation-parameters-v1"
TRANSFORMATION_SCOPE_POLICY = "architecture-aware-transformation-v1"
TRANSFORMATION_EXECUTION_POLICY = "cpu-float32-or-float64-v1"
RUNTIME_RELOAD_PROOF_SIDECAR = "runtime_reload_proof.json"
RUNTIME_RELOAD_PROOF_SCHEMA = "invarlock/transformation-runtime-reload-proof-v1"
RUNTIME_LOAD_DIAGNOSTICS_SCHEMA = "invarlock/pretrained-load-diagnostics-v1"
RUNTIME_STORAGE_KEY_AUDIT_SCHEMA = "invarlock/safetensors-storage-key-audit-v1"
TRAINING_EVIDENCE_PROOF_SIDECAR = "training_evidence_proof.json"
TRAINING_RECEIPT_SIDECAR = "training_receipt.json"
TRAINING_PROFILE_SNAPSHOT_SCHEMA = (
    "invarlock/evidence-pack-training-profile-snapshot-v1"
)
_RUNTIME_RELOAD_PROOF_FIELDS = {
    "schema",
    "ok",
    "replay_schema",
    "edit_type",
    "artifact_identity",
    "replay_artifact_identity",
    "prompt_sha256",
    "device",
    "input_device",
    "reload_runs",
    "token_ids_sha256",
    "token_ids_shape",
    "logits_sha256",
    "logits_shape",
    "all_logits_finite",
    "repeat_deterministic",
    "load_diagnostics",
    "storage_key_audit",
}
_RUNTIME_LOAD_DIAGNOSTIC_FIELDS = frozenset(
    {"unexpected_keys", "missing_keys", "mismatched_keys", "error_msgs"}
)
_RUNTIME_LOAD_DIAGNOSTICS_FIELDS = frozenset({"schema", "reloads"})
_RUNTIME_STORAGE_KEY_AUDIT_FIELDS = frozenset(
    {
        "artifact_storage_key_count",
        "artifact_storage_keys_sha256",
        "model_state_key_count",
        "model_state_keys_sha256",
        "unexpected_storage_keys",
    }
)
_RUNTIME_STORAGE_KEY_AUDIT_ENVELOPE_FIELDS = frozenset({"schema", "reloads"})
_TRANSFORMATION_ALGORITHMS = {
    "quant_rtn": "groupwise_rtn_dequantized_per_row_v1",
    "synthetic_lowrank_delta": "deterministic_synthetic_lowrank_delta_v1",
    "synthetic_dense_update": "deterministic_synthetic_dense_update_v1",
}
_VERIFIER_GRADE_TRANSFORMATION_EDIT_TYPES = frozenset(_TRANSFORMATION_ALGORITHMS)
_MAX_SYNTHETIC_LOWRANK_RANK = 32
_MAX_SYNTHETIC_DENSE_UPDATE_ITERATIONS = 16
DEPLOYABLE_SIDECARS = (
    "deployable_artifact_validation.json",
    "runtime_deployability_validation.json",
    "backend_inventory.json",
    "memory_report.json",
    "load_smoke.json",
    "inference_smoke.json",
    "publication_commit.json",
)
DEPLOYABLE_SIDECAR_SCHEMAS = {
    "deployable_artifact_validation.json": (
        "invarlock/deployable-artifact-validation-v1"
    ),
    "runtime_deployability_validation.json": (
        "invarlock/deployable-artifact-validation-v1"
    ),
    "backend_inventory.json": "invarlock/backend-inventory-v1",
    "memory_report.json": "invarlock/deployable-memory-report-v1",
    "load_smoke.json": "invarlock/deployable-load-smoke-v1",
    "inference_smoke.json": "invarlock/deployable-inference-smoke-v1",
    "publication_commit.json": "invarlock/deployable-publication-commit-v1",
}
_PROOF_LEDGER_SIDECARS = (
    "backend_inventory.json",
    "memory_report.json",
    "load_smoke.json",
    "inference_smoke.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


EDIT_PROVENANCE_FAMILIES = frozenset(
    {
        "custom",
        "deployable_backend_quantization",
        "dynamic_adapter",
        "fault_injection",
        "fine_tune",
        "knowledge_edit",
        "lora_merge",
        "magnitude_prune",
        "noop",
        "pruning",
        "quantization",
        "quantization_dequantized",
        "self_edit",
        "synthetic_dense_update",
        "synthetic_lowrank_delta",
    }
)
EDIT_IMPACT_SCENARIO_TYPES = frozenset(
    {
        "target_success",
        "near_neighbor",
        "near_confuser",
        "unrelated_locality",
        "general_ability_sentinel",
        "multilingual_portability",
        "sequential_edit_stress",
    }
)
EDIT_TOPOLOGY_ARTIFACT_KINDS = frozenset(
    {
        "checkpoint",
        "adapter",
        "merged_adapter",
        "memory_module",
        "dynamic_weight_module",
        "runtime_config",
        "prompt_wrapper",
    }
)
DELTA_AVAILABILITY_VALUES = frozenset({"none", "private", "public", "hash_only"})
PRIVACY_SENSITIVITY_VALUES = frozenset(
    {
        "public",
        "internal",
        "customer_controlled",
        "sensitive",
    }
)
_SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_SYNTHETIC_EDIT_TYPES = {"synthetic_dense_update", "synthetic_lowrank_delta"}


def _sanitize_model_key(model_key: str) -> str:
    """Match evidence-pack model directory names without trusting a host path."""

    normalized = model_key.lower().replace("/", "__").replace(" ", "_")
    return re.sub(r"[^a-z0-9_-]", "", normalized)


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _same_finite_number(left: object, right: object) -> bool:
    """Compare JSON numbers without accepting bools or non-finite values."""

    left_value = _finite_number(left)
    right_value = _finite_number(right)
    return (
        left_value is not None and right_value is not None and left_value == right_value
    )


def edit_metadata_coverage_errors(
    metadata: dict[str, Any],
    *,
    require_positive: bool,
) -> list[str]:
    """Validate edit coverage without coercing or repairing producer claims.

    A proof-routed model edit must identify at least one changed tensor and
    parameter in a non-empty checkpoint. Evidence-only/no-model records may
    carry an all-zero coverage block, but their ratio must still be exact.
    """

    coverage = metadata.get("coverage")
    if not isinstance(coverage, dict):
        return ["coverage must be an object"]

    errors: list[str] = []
    required_fields = {
        "edited_tensors",
        "edited_params",
        "total_params",
        "coverage_ratio",
    }
    missing = sorted(required_fields - set(coverage))
    errors.extend(f"coverage.{field} missing" for field in missing)

    edited_tensors = coverage.get("edited_tensors")
    edited_params = coverage.get("edited_params")
    total_params = coverage.get("total_params")
    ratio = coverage.get("coverage_ratio")
    for field, value in (
        ("edited_tensors", edited_tensors),
        ("edited_params", edited_params),
        ("total_params", total_params),
    ):
        if not _is_nonnegative_int(value):
            errors.append(f"coverage.{field} must be a non-negative integer")

    ratio_value = _finite_number(ratio)
    if ratio_value is None or not 0.0 <= ratio_value <= 1.0:
        errors.append("coverage.coverage_ratio must be finite and between 0 and 1")

    counts_are_valid = all(
        _is_nonnegative_int(value)
        for value in (edited_tensors, edited_params, total_params)
    )
    if counts_are_valid:
        assert isinstance(edited_tensors, int)
        assert isinstance(edited_params, int)
        assert isinstance(total_params, int)
        if edited_params > total_params:
            errors.append("coverage.edited_params must not exceed total_params")
        expected_ratio = edited_params / total_params if total_params else 0.0
        if ratio_value is not None and ratio_value != expected_ratio:
            errors.append(
                "coverage.coverage_ratio must equal edited_params / total_params"
            )
        if require_positive:
            for field, value in (
                ("edited_tensors", edited_tensors),
                ("edited_params", edited_params),
                ("total_params", total_params),
            ):
                if value <= 0:
                    errors.append(
                        f"coverage.{field} must be positive for a proof-routed model edit"
                    )
    return errors


def _load_json(path: Path) -> Any:
    return load_json(path, label=f"evidence-pack JSON {path}")


def _json_load_error_types() -> tuple[type[BaseException], ...]:
    return (OSError, TypeError, ValueError, json.JSONDecodeError, StrictJsonError)


def _infer_scenario_artifact_class(spec: dict[str, Any]) -> str:
    artifact_class = spec.get("artifact_class")
    if isinstance(artifact_class, str) and artifact_class:
        return artifact_class
    generation = spec.get("generation")
    kind = generation.get("kind") if isinstance(generation, dict) else ""
    if kind == "error":
        return FAULT_INJECTION_FIXTURE
    if kind == "deployable_edit":
        return DEPLOYABLE_OPTIMIZED_SUBJECT
    return VALIDATION_SUBJECT_CHECKPOINT if kind == "edit" else ""


def _typed_scenario_index_from_pack(
    pack_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, ScenarioContract], list[str]]:
    """Load every scenario through the one closed dispatch contract.

    This boundary rejects malformed, non-runnable, alias, or
    kind/class-conflicting records before report metadata is examined, so a bad
    scenario cannot silently fall through to a less demanding verifier path.
    """

    scenarios_path = pack_dir / "metadata" / "scenarios.json"
    if not scenarios_path.is_file() or scenarios_path.is_symlink():
        return {}, {}, []
    try:
        payload = _load_json(scenarios_path)
    except _json_load_error_types() as exc:
        return {}, {}, [f"metadata/scenarios.json is invalid: {exc}"]
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        return {}, {}, ["metadata/scenarios.json must contain a scenarios list"]

    records: dict[str, dict[str, Any]] = {}
    contracts: dict[str, ScenarioContract] = {}
    errors: list[str] = []
    for index, raw_record in enumerate(payload["scenarios"]):
        if not isinstance(raw_record, dict):
            errors.append(
                f"metadata/scenarios.json scenarios[{index}] must be an object"
            )
            continue
        try:
            contract = parse_scenario_contract(raw_record)
        except ScenarioContractError as exc:
            raw_id = raw_record.get("id")
            label = raw_id if isinstance(raw_id, str) and raw_id else str(index)
            errors.append(f"scenario {label!r} fails closed dispatch: {exc}")
            continue
        if contract.scenario_id in contracts:
            errors.append(
                "metadata/scenarios.json has duplicate scenario id: "
                + contract.scenario_id
            )
            continue
        records[contract.scenario_id] = raw_record
        contracts[contract.scenario_id] = contract
    return records, contracts, errors


def _report_scenario_id(pack_dir: Path, report_path: Path) -> str | None:
    try:
        rel = report_path.relative_to(pack_dir)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4 or parts[0] != "reports":
        return None
    if parts[2] == "errors":
        return parts[3] if len(parts) > 3 else None
    return parts[2]


def _report_model_name(pack_dir: Path, report_path: Path) -> str | None:
    try:
        rel = report_path.relative_to(pack_dir)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4 or parts[0] != "reports" or parts[2] == "errors":
        return None
    return parts[1] or None


def _load_json_sidecar(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = _load_json(path)
    except _json_load_error_types() as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "JSON sidecar must contain an object"
    return payload, None


def _expected_edit_type(spec: dict[str, Any]) -> str:
    generation = spec.get("generation")
    edit_spec = generation.get("edit_spec") if isinstance(generation, dict) else ""
    if isinstance(edit_spec, str) and edit_spec:
        return edit_spec.split(":", 1)[0]
    failure_class = spec.get("failure_class")
    if isinstance(failure_class, str) and "." in failure_class:
        return failure_class.rsplit(".", 1)[-1]
    return ""


def _expected_literal_pruning_params(
    spec: dict[str, Any],
) -> tuple[float | None, str | None, str | None]:
    generation = spec.get("generation")
    edit_spec = generation.get("edit_spec") if isinstance(generation, dict) else ""
    if not isinstance(edit_spec, str):
        return None, None, None
    parts = edit_spec.split(":")
    if len(parts) < 3 or parts[0] != "magnitude_prune" or parts[1] == "clean":
        return None, None, None
    try:
        sparsity = float(parts[1])
    except (TypeError, ValueError):
        return None, None, "magnitude_prune scenario sparsity is invalid"
    if not 0.0 < sparsity < 1.0:
        return None, None, "magnitude_prune scenario sparsity must be in (0, 1)"
    scope = parts[2]
    if not scope:
        return None, None, "magnitude_prune scenario scope is missing"
    return sparsity, scope, None
