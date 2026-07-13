"""Deployable quantization sidecar and report binding validation."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_contracts.deployable_coverage import (
    require_inventory_logical_binding,
    require_inventory_runtime_facts,
    require_logical_coverage,
)
from invarlock.evidence_pack_edit_common import (
    _PROOF_LEDGER_SIDECARS,
    DEPLOYABLE_SIDECAR_SCHEMAS,
    _sha256_file,
)


def _logical_coverage_errors(
    *, prefix: str, sidecar: str, payload: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    try:
        require_inventory_runtime_facts(payload)
    except ValueError as exc:
        errors.append(prefix + f"{sidecar} packed runtime facts invalid: {exc}")
    try:
        require_logical_coverage(payload.get("logical_coverage"))
    except ValueError as exc:
        errors.append(prefix + f"{sidecar} logical coverage invalid: {exc}")
    else:
        try:
            require_inventory_logical_binding(payload, payload.get("logical_coverage"))
        except ValueError as exc:
            errors.append(prefix + f"{sidecar} logical coverage binding invalid: {exc}")
    return errors


def _validation_sidecar_errors(
    *, prefix: str, sidecar: str, payload: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if payload.get("ok") is not True:
        errors.append(prefix + f"deployable sidecar did not pass: {sidecar}")
    if payload.get("load_smoke") is not True:
        errors.append(prefix + f"{sidecar} load_smoke must be true")
    if payload.get("inference_smoke") is not True:
        errors.append(prefix + f"{sidecar} inference_smoke must be true")
    if sidecar == "deployable_artifact_validation.json":
        if payload.get("validation_scope") != "structural_only":
            errors.append(
                prefix + f"{sidecar} validation_scope must be structural_only"
            )
        if payload.get("runtime_proof_authoritative") is not False:
            errors.append(
                prefix + f"{sidecar} runtime_proof_authoritative must be false"
            )
        return errors
    if payload.get("validation_scope") != "runtime_reproof":
        errors.append(prefix + f"{sidecar} validation_scope must be runtime_reproof")
    if payload.get("runtime_proof_authoritative") is not True:
        errors.append(prefix + f"{sidecar} runtime_proof_authoritative must be true")
    if not isinstance(payload.get("runtime_proof"), dict):
        errors.append(prefix + f"{sidecar} runtime_proof must be an object")
    return errors


def _backend_inventory_errors(
    *, prefix: str, sidecar: str, payload: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if "ok" in payload and payload.get("ok") is not True:
        errors.append(prefix + f"deployable sidecar did not pass: {sidecar}")
    if payload.get("load_smoke") is not True:
        errors.append(prefix + f"{sidecar} load_smoke must be true")
    if payload.get("inference_smoke") is not True:
        errors.append(prefix + f"{sidecar} inference_smoke must be true")
    errors.extend(
        _logical_coverage_errors(prefix=prefix, sidecar=sidecar, payload=payload)
    )
    footprint = payload.get("memory_footprint")
    if not isinstance(footprint, dict):
        errors.append(prefix + f"{sidecar} memory_footprint must be an object")
    elif (
        not isinstance(footprint.get("reported_bytes"), int)
        or int(footprint.get("reported_bytes", 0)) <= 0
    ):
        errors.append(
            prefix + f"{sidecar} memory_footprint.reported_bytes must be positive"
        )
    return errors


def _memory_report_errors(
    *, prefix: str, sidecar: str, payload: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    baseline = payload.get("baseline_reported_bytes")
    quantized = payload.get("quantized_reported_bytes")
    reduction = payload.get("reduction_bytes")
    ratio = payload.get("reduction_ratio")
    if (
        isinstance(baseline, bool)
        or not isinstance(baseline, int)
        or isinstance(quantized, bool)
        or not isinstance(quantized, int)
        or isinstance(reduction, bool)
        or not isinstance(reduction, int)
        or baseline <= quantized
        or reduction != baseline - quantized
    ):
        errors.append(prefix + f"{sidecar} observed reduction is invalid")
    expected_ratio = (
        reduction / baseline
        if isinstance(baseline, int)
        and not isinstance(baseline, bool)
        and isinstance(reduction, int)
        and not isinstance(reduction, bool)
        and baseline > 0
        else None
    )
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, int | float)
        or not math.isfinite(float(ratio))
        or expected_ratio is None
        or float(ratio) != expected_ratio
    ):
        errors.append(
            prefix + f"{sidecar} reduction_ratio does not match observed reduction"
        )
    if payload.get("runtime_memory_reduction_observed") is not True:
        errors.append(prefix + f"{sidecar} reduction must be observed")
    return errors


def _inference_smoke_errors(
    *, prefix: str, sidecar: str, payload: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    digest = payload.get("logits_sha256")
    shape = payload.get("logits_shape")
    if payload.get("all_logits_finite") is not True:
        errors.append(prefix + f"{sidecar} finite logits proof missing")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        errors.append(prefix + f"{sidecar} logits digest invalid")
    if (
        not isinstance(shape, list)
        or not shape
        or not all(isinstance(value, int) and value > 0 for value in shape)
    ):
        errors.append(prefix + f"{sidecar} logits shape invalid")
    return errors


def _deployable_sidecar_consistency_errors(
    *,
    scenario_id: str,
    sidecar: str,
    payload: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    prefix = f"{scenario_id}: "
    expected_schema = DEPLOYABLE_SIDECAR_SCHEMAS.get(sidecar)
    if expected_schema and payload.get("schema") != expected_schema:
        errors.append(
            prefix
            + f"deployable sidecar schema mismatch ({sidecar}): "
            + f"expected {expected_schema!r}, got {payload.get('schema')!r}"
        )

    if sidecar in {
        "deployable_artifact_validation.json",
        "runtime_deployability_validation.json",
    }:
        errors.extend(
            _validation_sidecar_errors(prefix=prefix, sidecar=sidecar, payload=payload)
        )
    elif sidecar == "backend_inventory.json":
        errors.extend(
            _backend_inventory_errors(prefix=prefix, sidecar=sidecar, payload=payload)
        )
    elif sidecar == "publication_commit.json":
        if payload.get("committed") is not True:
            errors.append(prefix + f"{sidecar} committed must be true")
    else:
        if payload.get("ok") is not True:
            errors.append(prefix + f"deployable sidecar did not pass: {sidecar}")
        if sidecar == "memory_report.json":
            errors.extend(
                _memory_report_errors(prefix=prefix, sidecar=sidecar, payload=payload)
            )
        elif sidecar == "load_smoke.json":
            if payload.get("loaded_from_saved_checkpoint") is not True:
                errors.append(prefix + f"{sidecar} saved-checkpoint load missing")
            if payload.get("load_time_quantization_override") is not False:
                errors.append(prefix + f"{sidecar} load override must be false")
            errors.extend(
                _logical_coverage_errors(
                    prefix=prefix, sidecar=sidecar, payload=payload
                )
            )
        elif sidecar == "inference_smoke.json":
            errors.extend(
                _inference_smoke_errors(prefix=prefix, sidecar=sidecar, payload=payload)
            )
    return errors


def _deployable_backend_contract_errors(
    *,
    prefix: str,
    generation: dict[str, Any],
    metadata: dict[str, Any],
    sidecars: dict[str, dict[str, Any]],
) -> list[str]:
    """Bind the declared backend and bitwidth to every runtime proof surface."""

    errors: list[str] = []
    validation = sidecars.get("deployable_artifact_validation.json", {})
    runtime_validation = sidecars.get("runtime_deployability_validation.json", {})
    publication = sidecars.get("publication_commit.json", {})
    inventory = sidecars.get("backend_inventory.json", {})
    load_smoke = sidecars.get("load_smoke.json", {})
    logical = metadata.get("logical_coverage")
    try:
        logical = require_logical_coverage(logical)
    except ValueError as exc:
        errors.append(prefix + f"deployable metadata logical coverage invalid: {exc}")
        logical = None
    if logical is not None:
        expected_coverage = {
            "edited_tensors": logical["weight_tensor_count"],
            "edited_params": logical["parameter_elements"],
            "total_params": logical["total_unique_parameter_elements"],
            "coverage_ratio": logical["parameter_elements"]
            / logical["total_unique_parameter_elements"],
        }
        if metadata.get("coverage") != expected_coverage:
            errors.append(prefix + "deployable metadata coverage is not canonical")
        for label, payload in (("inventory", inventory), ("load smoke", load_smoke)):
            if payload.get("logical_coverage") != logical:
                errors.append(prefix + f"deployable {label} logical coverage mismatch")
    expected_backend = generation.get("backend")
    if not isinstance(expected_backend, str) or not expected_backend:
        errors.append(prefix + "deployable scenario backend missing")
    else:
        for label, value in (
            ("metadata", metadata.get("backend")),
            ("validation", validation.get("backend")),
            ("runtime validation", runtime_validation.get("backend")),
            ("inventory", inventory.get("backend")),
        ):
            if value != expected_backend:
                errors.append(prefix + f"deployable {label} backend mismatch")

    edit_spec = generation.get("edit_spec")
    expected_edit_type = str(edit_spec).split(":", 1)[0] if edit_spec else ""
    if not expected_edit_type or metadata.get("edit_type") != expected_edit_type:
        errors.append(prefix + "deployable edit type does not match scenario")
    match = re.fullmatch(r"bnb_(4|8)bit", expected_edit_type)
    expected_bits = int(match.group(1)) if match else None
    if expected_bits is None:
        errors.append(
            prefix + "deployable scenario edit type has no supported bitwidth"
        )
        return errors

    for label, payload in sidecars.items():
        if payload.get("bits") != expected_bits:
            errors.append(prefix + f"deployable {label} bitwidth mismatch")
    if publication.get("bits") != expected_bits:
        errors.append(prefix + "deployable publication bitwidth mismatch")
    quantization_config = inventory.get("quantization_config")
    expected_flag = "load_in_4bit" if expected_bits == 4 else "load_in_8bit"
    opposite_flag = "load_in_8bit" if expected_bits == 4 else "load_in_4bit"
    if not isinstance(quantization_config, dict) or (
        quantization_config.get(expected_flag) is not True
        or quantization_config.get(opposite_flag) is True
    ):
        errors.append(prefix + "deployable inventory quantization flags mismatch")
    expected_module = "linear4bit" if expected_bits == 4 else "linear8bitlt"
    module_types = inventory.get("quantized_module_types")
    if not isinstance(module_types, list) or not any(
        isinstance(value, str) and expected_module in value.lower()
        for value in module_types
    ):
        errors.append(prefix + "deployable inventory module types mismatch")
    return errors


def _deployable_binding_errors(
    *,
    scenario_id: str,
    spec: dict[str, Any],
    report: dict[str, Any],
    metadata: dict[str, Any],
    report_dir: Path,
    sidecars: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    prefix = f"{scenario_id}: "
    report_identity = (
        (report.get("meta") or {}).get("model_identity")
        if isinstance(report.get("meta"), dict)
        else None
    )
    baseline_ref = report.get("baseline_ref")
    report_baseline_identity = (
        baseline_ref.get("model_identity") if isinstance(baseline_ref, dict) else None
    )
    validation = sidecars.get("deployable_artifact_validation.json", {})
    runtime_validation = sidecars.get("runtime_deployability_validation.json", {})
    publication = sidecars.get("publication_commit.json", {})
    inventory = sidecars.get("backend_inventory.json", {})
    load_smoke = sidecars.get("load_smoke.json", {})
    if validation.get("validation_scope") != "structural_only":
        errors.append(prefix + "deployable structural validation scope is invalid")
    if validation.get("runtime_proof_authoritative") is not False:
        errors.append(
            prefix + "deployable structural validation must not claim runtime authority"
        )
    if runtime_validation.get("validation_scope") != "runtime_reproof":
        errors.append(prefix + "deployable runtime validation scope is invalid")
    if runtime_validation.get("runtime_proof_authoritative") is not True:
        errors.append(prefix + "deployable runtime validation is not authoritative")
    if publication.get("validation_scope") != "structural_only":
        errors.append(prefix + "deployable publication validation scope is invalid")
    if publication.get("runtime_proof_authoritative") is not False:
        errors.append(
            prefix + "deployable publication must not claim runtime authority"
        )
    proof_identity = runtime_validation.get("artifact_identity")
    if not isinstance(report_identity, dict) or report_identity != proof_identity:
        errors.append(
            prefix
            + "deployable proof artifact identity does not match evaluation subject identity"
        )
    if publication.get("artifact_identity") != proof_identity:
        errors.append(prefix + "deployable publication artifact identity mismatch")
    if validation.get("artifact_identity") != proof_identity:
        errors.append(
            prefix + "generated and runtime deployable artifact identities disagree"
        )
    if not isinstance(report_baseline_identity, dict):
        errors.append(prefix + "evaluation baseline identity missing")
    for label, payload in sidecars.items():
        if payload.get("baseline_identity") != report_baseline_identity:
            errors.append(prefix + f"deployable {label} baseline identity mismatch")
    generation = spec.get("generation")
    generation = generation if isinstance(generation, dict) else {}
    errors.extend(
        _deployable_backend_contract_errors(
            prefix=prefix,
            generation=generation,
            metadata=metadata,
            sidecars=sidecars,
        )
    )
    actual_ledger = {
        name: _sha256_file(report_dir / name)
        for name in _PROOF_LEDGER_SIDECARS
        if (report_dir / name).is_file()
    }
    if validation.get("sidecar_digests") != actual_ledger:
        errors.append(prefix + "deployable validation sidecar digest ledger mismatch")
    if runtime_validation.get("sidecar_digests") != actual_ledger:
        errors.append(
            prefix + "deployable runtime validation sidecar digest ledger mismatch"
        )
    if publication.get("sidecar_digests") != actual_ledger:
        errors.append(prefix + "deployable publication sidecar digest ledger mismatch")
    validation_path = report_dir / "deployable_artifact_validation.json"
    if validation_path.is_file() and publication.get(
        "proof_validation_sha256"
    ) != _sha256_file(validation_path):
        errors.append(prefix + "deployable publication validation digest mismatch")
    runtime_proof = runtime_validation.get("runtime_proof")
    if not isinstance(runtime_proof, dict):
        errors.append(prefix + "deployable runtime proof missing")
    else:
        inference_smoke = sidecars.get("inference_smoke.json", {})
        try:
            require_inventory_runtime_facts(runtime_proof)
        except ValueError as exc:
            errors.append(prefix + f"deployable runtime packed facts invalid: {exc}")
        try:
            require_logical_coverage(runtime_proof.get("logical_coverage"))
        except ValueError as exc:
            errors.append(
                prefix + f"deployable runtime logical coverage invalid: {exc}"
            )
        else:
            try:
                require_inventory_logical_binding(
                    runtime_proof, runtime_proof.get("logical_coverage")
                )
            except ValueError as exc:
                errors.append(
                    prefix
                    + f"deployable runtime logical coverage binding invalid: {exc}"
                )
        for label, payload in (("inventory", inventory), ("load smoke", load_smoke)):
            if payload.get("quantized_module_count") != runtime_proof.get(
                "quantized_module_count"
            ):
                errors.append(
                    prefix + f"deployable runtime module count disagrees with {label}"
                )
            if payload.get("quantized_module_types") != runtime_proof.get(
                "quantized_module_types"
            ):
                errors.append(
                    prefix + f"deployable runtime module types disagree with {label}"
                )
            for field in (
                "quantized_module_names",
                "quantized_module_names_sha256",
                "packed_weight_storage_elements",
                "logical_coverage",
            ):
                if payload.get(field) != runtime_proof.get(field):
                    errors.append(
                        prefix + f"deployable runtime {field} disagrees with {label}"
                    )
        for field in ("logits_sha256", "logits_shape", "all_logits_finite"):
            if inference_smoke.get(field) != runtime_proof.get(field):
                errors.append(
                    prefix + f"deployable runtime inference disagrees on {field}"
                )
        memory_report = sidecars.get("memory_report.json", {})
        for field in (
            "baseline_reported_bytes",
            "quantized_reported_bytes",
            "reduction_bytes",
            "reduction_ratio",
            "runtime_memory_reduction_observed",
        ):
            if memory_report.get(field) != runtime_proof.get(field):
                errors.append(
                    prefix + f"deployable runtime memory disagrees on {field}"
                )
    return errors
