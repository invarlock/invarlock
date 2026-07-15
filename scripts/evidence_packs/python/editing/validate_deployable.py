"""Deployable quantization artifact validation phases."""

from __future__ import annotations

import importlib.metadata
import json
import math
import re
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_contracts.deployable_coverage import (
    require_inventory_logical_binding,
    require_inventory_runtime_facts,
    require_logical_coverage,
)

from .implementations import DEPLOYABLE_OPTIMIZED_SUBJECT, read_edit_metadata

DEPLOYABLE_VALIDATION_SCHEMA = "invarlock/deployable-artifact-validation-v1"
DEPLOYABLE_STRUCTURAL_VALIDATION_SCOPE = "structural_only"
DEPLOYABLE_RUNTIME_REPROOF_SCOPE = "runtime_reproof"
REQUIRED_DEPLOYABLE_SIDECAR_SCHEMAS = {
    "backend_inventory.json": "invarlock/backend-inventory-v1",
    "memory_report.json": "invarlock/deployable-memory-report-v1",
    "load_smoke.json": "invarlock/deployable-load-smoke-v1",
    "inference_smoke.json": "invarlock/deployable-inference-smoke-v1",
}
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _deployable_metadata_issues(
    metadata: dict[str, Any], backend: str | None
) -> list[str]:
    issues: list[str] = []
    if metadata.get("artifact_class") != DEPLOYABLE_OPTIMIZED_SUBJECT:
        issues.append(
            "edit_metadata.artifact_class must be deployable_optimized_subject"
        )
    if metadata.get("optimized_deployment_backend") is not True:
        issues.append("edit_metadata.optimized_deployment_backend must be true")
    if metadata.get("packed_quantized_storage") is not True:
        issues.append("edit_metadata.packed_quantized_storage must be true")
    if backend and metadata.get("backend") != backend:
        issues.append(
            f"edit_metadata.backend mismatch: expected {backend!r}, "
            f"got {metadata.get('backend')!r}"
        )
    if not metadata.get("backend"):
        issues.append("edit_metadata.backend missing")
    try:
        logical = require_logical_coverage(metadata.get("logical_coverage"))
    except ValueError as exc:
        issues.append(f"edit_metadata.logical_coverage invalid: {exc}")
    else:
        coverage = metadata.get("coverage")
        expected = {
            "edited_tensors": logical["weight_tensor_count"],
            "edited_params": logical["parameter_elements"],
            "total_params": logical["total_unique_parameter_elements"],
            "coverage_ratio": logical["parameter_elements"]
            / logical["total_unique_parameter_elements"],
        }
        if coverage != expected:
            issues.append("edit_metadata.coverage does not bind logical coverage")
    return issues


def _backend_inventory_issues(
    payload: dict[str, Any], *, backend: str | None
) -> list[str]:
    issues: list[str] = []
    sidecar = "backend_inventory.json"
    if "ok" in payload and payload.get("ok") is not True:
        issues.append(f"{sidecar} ok must be true")
    if backend and payload.get("backend") != backend:
        issues.append(
            f"{sidecar} backend mismatch: expected {backend!r}, "
            f"got {payload.get('backend')!r}"
        )
    if payload.get("load_smoke") is not True:
        issues.append(f"{sidecar} load_smoke must be true")
    if payload.get("inference_smoke") is not True:
        issues.append(f"{sidecar} inference_smoke must be true")
    try:
        require_inventory_runtime_facts(payload)
    except ValueError as exc:
        issues.append(f"{sidecar} packed runtime facts invalid: {exc}")
    try:
        require_logical_coverage(payload.get("logical_coverage"))
    except ValueError as exc:
        issues.append(f"{sidecar} logical coverage invalid: {exc}")
    else:
        try:
            require_inventory_logical_binding(payload, payload.get("logical_coverage"))
        except ValueError as exc:
            issues.append(f"{sidecar} logical coverage binding invalid: {exc}")
    memory_footprint = payload.get("memory_footprint")
    if not isinstance(memory_footprint, dict):
        issues.append(f"{sidecar} memory_footprint must be an object")
    elif (
        not isinstance(memory_footprint.get("reported_bytes"), int)
        or int(memory_footprint.get("reported_bytes", 0)) <= 0
    ):
        issues.append(f"{sidecar} memory_footprint.reported_bytes must be positive")
    return issues


def _deployable_sidecar_issues(
    sidecar: str,
    payload: dict[str, Any],
    *,
    backend: str | None,
) -> list[str]:
    issues: list[str] = []
    expected_schema = REQUIRED_DEPLOYABLE_SIDECAR_SCHEMAS[sidecar]
    if payload.get("schema") != expected_schema:
        issues.append(
            f"{sidecar} schema mismatch: expected {expected_schema!r}, "
            f"got {payload.get('schema')!r}"
        )
    if sidecar == "backend_inventory.json":
        issues.extend(_backend_inventory_issues(payload, backend=backend))
        return issues

    if payload.get("ok") is not True:
        issues.append(f"{sidecar} ok must be true")
    if sidecar == "memory_report.json":
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
            or baseline <= 0
            or quantized <= 0
            or reduction != baseline - quantized
            or reduction <= 0
        ):
            issues.append(f"{sidecar} must record a valid positive observed reduction")
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
            issues.append(
                f"{sidecar} reduction_ratio must equal reduction_bytes / "
                "baseline_reported_bytes"
            )
        if payload.get("runtime_memory_reduction_observed") is not True:
            issues.append(f"{sidecar} runtime_memory_reduction_observed must be true")
    elif sidecar == "load_smoke.json":
        if payload.get("loaded_from_saved_checkpoint") is not True:
            issues.append(f"{sidecar} loaded_from_saved_checkpoint must be true")
        if payload.get("load_time_quantization_override") is not False:
            issues.append(f"{sidecar} load_time_quantization_override must be false")
        try:
            require_inventory_runtime_facts(payload)
        except ValueError as exc:
            issues.append(f"{sidecar} packed runtime facts invalid: {exc}")
        try:
            require_logical_coverage(payload.get("logical_coverage"))
        except ValueError as exc:
            issues.append(f"{sidecar} logical coverage invalid: {exc}")
        else:
            try:
                require_inventory_logical_binding(
                    payload, payload.get("logical_coverage")
                )
            except ValueError as exc:
                issues.append(f"{sidecar} logical coverage binding invalid: {exc}")
    elif sidecar == "inference_smoke.json":
        if payload.get("all_logits_finite") is not True:
            issues.append(f"{sidecar} all_logits_finite must be true")
        if not _valid_digest(payload.get("logits_sha256")):
            issues.append(f"{sidecar} logits_sha256 must be a sha256 digest")
        shape = payload.get("logits_shape")
        if (
            not isinstance(shape, list)
            or not shape
            or not all(isinstance(value, int) and value > 0 for value in shape)
        ):
            issues.append(f"{sidecar} logits_shape must contain positive dimensions")
    return issues


def _runtime_reproof(
    owner: Any,
    *,
    artifact_dir: Path,
    baseline_dir: Path | None,
    expected_bits: int | None,
    trust_remote_code: bool,
    expected_identity: dict[str, str] | None,
    expected_baseline_identity: dict[str, str] | None,
    sidecar_payloads: dict[str, dict[str, Any]],
    issues: list[str],
    smoke: bool,
) -> dict[str, Any] | None:
    runtime_proof: dict[str, Any] | None = None
    if smoke and expected_bits in {4, 8}:
        if baseline_dir is None:
            issues.append("runtime deployability smoke requires baseline checkpoint")
        else:
            try:
                runtime_proof = owner._runtime_bitsandbytes_proof(
                    artifact_dir,
                    baseline_dir=baseline_dir,
                    expected_bits=expected_bits,
                    trust_remote_code=trust_remote_code,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                issues.append(f"runtime deployability smoke failed: {exc}")
            else:
                if runtime_proof.get("artifact_identity") != expected_identity:
                    issues.append("runtime deployability artifact identity mismatch")
                if runtime_proof.get("baseline_identity") != expected_baseline_identity:
                    issues.append("runtime deployability baseline identity mismatch")
                if runtime_proof.get("trust_remote_code") is not trust_remote_code:
                    issues.append("runtime deployability trust_remote_code mismatch")
        if runtime_proof is not None:
            try:
                require_inventory_runtime_facts(runtime_proof)
            except ValueError as exc:
                issues.append(f"runtime packed facts invalid: {exc}")
            try:
                require_logical_coverage(runtime_proof.get("logical_coverage"))
            except ValueError as exc:
                issues.append(f"runtime logical coverage invalid: {exc}")
            else:
                try:
                    require_inventory_logical_binding(
                        runtime_proof, runtime_proof.get("logical_coverage")
                    )
                except ValueError as exc:
                    issues.append(f"runtime logical coverage binding invalid: {exc}")
            inventory_payload = sidecar_payloads.get("backend_inventory.json", {})
            load_payload = sidecar_payloads.get("load_smoke.json", {})
            inference_payload = sidecar_payloads.get("inference_smoke.json", {})
            observed_count = runtime_proof.get("quantized_module_count")
            observed_names = runtime_proof.get("quantized_module_names")
            observed_names_sha256 = runtime_proof.get("quantized_module_names_sha256")
            observed_types = runtime_proof.get("quantized_module_types")
            for label, payload in (
                ("backend inventory", inventory_payload),
                ("load smoke", load_payload),
            ):
                if payload.get("quantized_module_count") != observed_count:
                    issues.append(f"runtime packed module count disagrees with {label}")
                if payload.get("quantized_module_types") != observed_types:
                    issues.append(f"runtime packed module types disagree with {label}")
                if payload.get("quantized_module_names") != observed_names:
                    issues.append(f"runtime packed module names disagree with {label}")
                if (
                    payload.get("quantized_module_names_sha256")
                    != observed_names_sha256
                ):
                    issues.append(
                        f"runtime packed module names digest disagrees with {label}"
                    )
                if payload.get("packed_weight_storage_elements") != runtime_proof.get(
                    "packed_weight_storage_elements"
                ):
                    issues.append(
                        f"runtime packed storage elements disagree with {label}"
                    )
                if payload.get("logical_coverage") != runtime_proof.get(
                    "logical_coverage"
                ):
                    issues.append(f"runtime logical coverage disagrees with {label}")
            for field in ("logits_sha256", "logits_shape", "all_logits_finite"):
                if inference_payload.get(field) != runtime_proof.get(field):
                    issues.append(f"runtime inference proof disagrees on {field}")
            memory_payload = sidecar_payloads.get("memory_report.json", {})
            for field in (
                "baseline_reported_bytes",
                "quantized_reported_bytes",
                "reduction_bytes",
                "reduction_ratio",
                "runtime_memory_reduction_observed",
            ):
                if memory_payload.get(field) != runtime_proof.get(field):
                    issues.append(f"runtime memory proof disagrees on {field}")

    return runtime_proof


def _publication_issues(
    owner: Any,
    *,
    require_publication: bool,
    report_dir: Path | None,
    expected_identity: dict[str, str] | None,
    baseline_identities: set[str],
    expected_bits: int | None,
    trust_remote_code: bool,
    sidecar_digests: dict[str, str],
    issues: list[str],
) -> None:
    if require_publication:
        publication = (
            _load_json_object(report_dir / "publication_commit.json")
            if report_dir is not None
            else None
        )
        prior_validation = (
            _load_json_object(report_dir / "deployable_artifact_validation.json")
            if report_dir is not None
            else None
        )
        if publication is None:
            issues.append("missing deployable publication commit")
        else:
            if (
                publication.get("schema")
                != "invarlock/deployable-publication-commit-v1"
                or publication.get("committed") is not True
            ):
                issues.append("deployable publication commit is invalid")
            if (
                publication.get("validation_scope")
                != DEPLOYABLE_STRUCTURAL_VALIDATION_SCOPE
            ):
                issues.append(
                    "deployable publication validation scope must be structural_only"
                )
            if publication.get("runtime_proof_authoritative") is not False:
                issues.append(
                    "deployable publication must not claim runtime proof authority"
                )
            if publication.get("artifact_identity") != expected_identity:
                issues.append("deployable publication artifact identity mismatch")
            publication_baseline = publication.get("baseline_identity")
            if (
                json.dumps(publication_baseline, sort_keys=True)
                not in baseline_identities
            ):
                issues.append("deployable publication baseline identity mismatch")
            if publication.get("bits") != expected_bits:
                issues.append("deployable publication bitwidth mismatch")
            if publication.get("trust_remote_code") is not trust_remote_code:
                issues.append("deployable publication trust_remote_code mismatch")
            if publication.get("sidecar_digests") != sidecar_digests:
                issues.append("deployable publication sidecar digest ledger mismatch")
            validation_path = (
                report_dir / "deployable_artifact_validation.json"
                if report_dir is not None
                else None
            )
            if (
                validation_path is None
                or not validation_path.is_file()
                or publication.get("proof_validation_sha256")
                != owner._file_sha256(validation_path)
            ):
                issues.append("deployable publication validation digest mismatch")
        if prior_validation is None:
            issues.append("missing published deployable artifact validation")
        else:
            if (
                prior_validation.get("validation_scope")
                != DEPLOYABLE_STRUCTURAL_VALIDATION_SCOPE
            ):
                issues.append(
                    "published deployable validation scope must be structural_only"
                )
            if prior_validation.get("runtime_proof_authoritative") is not False:
                issues.append(
                    "published deployable validation must not claim runtime proof authority"
                )
            if prior_validation.get("artifact_identity") != expected_identity:
                issues.append(
                    "published deployable validation artifact identity mismatch"
                )
            if (
                json.dumps(prior_validation.get("baseline_identity"), sort_keys=True)
                not in baseline_identities
            ):
                issues.append(
                    "published deployable validation baseline identity mismatch"
                )
            if prior_validation.get("sidecar_digests") != sidecar_digests:
                issues.append("published deployable validation sidecar ledger mismatch")
            if prior_validation.get("trust_remote_code") is not trust_remote_code:
                issues.append(
                    "published deployable validation trust_remote_code mismatch"
                )


def _collect_deployable_inputs(
    owner: Any,
    *,
    artifact_dir: Path,
    backend: str | None,
    report_dir: Path | None,
    expected_bits: int | None,
    trust_remote_code: bool,
    baseline_dir: Path | None,
    issues: list[str],
) -> tuple[
    dict[str, Any],
    int | None,
    str,
    str | None,
    dict[str, dict[str, Any]],
    dict[str, str] | None,
    dict[str, str] | None,
    set[str],
]:
    metadata_path = artifact_dir / "edit_metadata.json"
    metadata: dict[str, Any] = {}

    artifact_result = owner.validate_edit_artifact(
        artifact_dir,
        require_metadata=True,
        expected_artifact_class=DEPLOYABLE_OPTIMIZED_SUBJECT,
    )
    issues.extend(artifact_result.issues or [])

    if metadata_path.is_file():
        try:
            metadata = read_edit_metadata(metadata_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            issues.append(f"edit_metadata.json invalid: {exc}")
        else:
            issues.extend(_deployable_metadata_issues(metadata, backend))

    metadata_bits = (
        (metadata.get("parameters") or {}).get("bits")
        if isinstance(metadata.get("parameters"), dict)
        else None
    )
    if expected_bits is None and isinstance(metadata_bits, int):
        expected_bits = metadata_bits
    if expected_bits not in {4, 8}:
        issues.append("deployable validation requires expected bitwidth 4 or 8")
    elif metadata.get("edit_type") != f"bnb_{expected_bits}bit":
        issues.append("edit_metadata.edit_type does not match expected bitwidth")
    elif metadata_bits != expected_bits:
        issues.append("edit_metadata.parameters.bits does not match expected bitwidth")

    resolved_backend = backend or str(metadata.get("backend") or "")
    backend_version = _package_version(resolved_backend) if resolved_backend else None
    if resolved_backend and backend_version is None:
        issues.append(f"backend package not importable: {resolved_backend}")

    sidecar_payloads: dict[str, dict[str, Any]] = {}
    if report_dir is None:
        issues.append("deployable validation requires --report-dir sidecars")
    else:
        for sidecar in REQUIRED_DEPLOYABLE_SIDECAR_SCHEMAS:
            payload = _load_json_object(report_dir / sidecar)
            if payload is None:
                issues.append(f"missing or invalid report sidecar: {sidecar}")
                continue
            sidecar_payloads[sidecar] = payload
            issues.extend(
                _deployable_sidecar_issues(
                    sidecar, payload, backend=resolved_backend or None
                )
            )

    metadata_logical = metadata.get("logical_coverage")
    for sidecar in ("backend_inventory.json", "load_smoke.json"):
        payload = sidecar_payloads.get(sidecar)
        if (
            isinstance(payload, dict)
            and payload.get("logical_coverage") != metadata_logical
        ):
            issues.append(f"{sidecar} logical coverage does not bind edit_metadata")

    artifact_digest: str | None = None
    try:
        artifact_digest = owner.checkpoint_tree_sha256(artifact_dir)
    except (OSError, ValueError) as exc:
        issues.append(f"artifact identity unavailable: {exc}")
    expected_identity = (
        {
            "kind": "local_checkpoint_tree",
            "sha256": artifact_digest,
        }
        if artifact_digest
        else None
    )
    expected_baseline_identity: dict[str, str] | None = None
    if baseline_dir is not None:
        try:
            expected_baseline_identity = {
                "kind": "local_checkpoint_tree",
                "sha256": owner.checkpoint_tree_sha256(baseline_dir),
            }
        except (OSError, ValueError) as exc:
            issues.append(f"baseline identity unavailable: {exc}")
    for sidecar, payload in sidecar_payloads.items():
        if payload.get("artifact_identity") != expected_identity:
            issues.append(f"{sidecar} artifact_identity mismatch")
        baseline_identity = payload.get("baseline_identity")
        if (
            not isinstance(baseline_identity, dict)
            or baseline_identity.get("kind") != "local_checkpoint_tree"
            or not _valid_digest(baseline_identity.get("sha256"))
        ):
            issues.append(f"{sidecar} baseline_identity missing")
        elif (
            expected_baseline_identity is not None
            and baseline_identity != expected_baseline_identity
        ):
            issues.append(f"{sidecar} baseline_identity mismatch")
        if expected_bits in {4, 8} and payload.get("bits") != expected_bits:
            issues.append(f"{sidecar} bits mismatch")
        if payload.get("trust_remote_code") is not trust_remote_code:
            issues.append(f"{sidecar} trust_remote_code mismatch")

    baseline_identities = {
        json.dumps(payload.get("baseline_identity"), sort_keys=True)
        for payload in sidecar_payloads.values()
        if isinstance(payload.get("baseline_identity"), dict)
    }
    if len(baseline_identities) > 1:
        issues.append("deployable proof sidecars disagree on baseline_identity")
    issues.extend(
        _quantization_inventory_issues(
            sidecar_payloads.get("backend_inventory.json", {}), expected_bits
        )
    )

    return (
        metadata,
        expected_bits,
        resolved_backend,
        backend_version,
        sidecar_payloads,
        expected_identity,
        expected_baseline_identity,
        baseline_identities,
    )


def _quantization_inventory_issues(
    inventory: dict[str, Any], expected_bits: int | None
) -> list[str]:
    if expected_bits not in {4, 8}:
        return []
    expected_flag = "load_in_4bit" if expected_bits == 4 else "load_in_8bit"
    opposite_flag = "load_in_8bit" if expected_bits == 4 else "load_in_4bit"
    expected_type = "linear4bit" if expected_bits == 4 else "linear8bitlt"
    quant_config = inventory.get("quantization_config")
    module_types = inventory.get("quantized_module_types")
    issues: list[str] = []
    if not isinstance(quant_config, dict) or (
        quant_config.get(expected_flag) is not True
        or quant_config.get(opposite_flag) is True
    ):
        issues.append("backend_inventory.json quantization bit flags mismatch")
    if not isinstance(module_types, list) or not any(
        isinstance(value, str) and expected_type in value.lower()
        for value in module_types
    ):
        issues.append("backend_inventory.json module types mismatch expected bitwidth")
    return issues


def validate_deployable_artifact(
    owner: Any,
    artifact_dir: Path,
    *,
    backend: str | None = None,
    report_dir: Path | None = None,
    smoke: bool = False,
    expected_bits: int | None = None,
    trust_remote_code: bool = False,
    require_publication: bool = False,
    baseline_dir: Path | None = None,
) -> dict[str, Any]:
    trust_remote_code = owner._resolve_remote_code_request(trust_remote_code)
    issues: list[str] = []
    (
        metadata,
        expected_bits,
        resolved_backend,
        backend_version,
        sidecar_payloads,
        expected_identity,
        expected_baseline_identity,
        baseline_identities,
    ) = _collect_deployable_inputs(
        owner,
        artifact_dir=artifact_dir,
        backend=backend,
        report_dir=report_dir,
        expected_bits=expected_bits,
        trust_remote_code=trust_remote_code,
        baseline_dir=baseline_dir,
        issues=issues,
    )

    runtime_proof = _runtime_reproof(
        owner,
        artifact_dir=artifact_dir,
        baseline_dir=baseline_dir,
        expected_bits=expected_bits,
        trust_remote_code=trust_remote_code,
        expected_identity=expected_identity,
        expected_baseline_identity=expected_baseline_identity,
        sidecar_payloads=sidecar_payloads,
        issues=issues,
        smoke=smoke,
    )

    # This validator is intentionally conservative. Heavy reload/inference smoke
    # should be produced by backend-specific generators and passed as sidecars.
    load_smoke = (
        sidecar_payloads.get("load_smoke.json", {}).get("ok") is True
        if report_dir is not None
        else False
    )
    inference_smoke = (
        sidecar_payloads.get("inference_smoke.json", {}).get("ok") is True
        if report_dir is not None
        else False
    )
    if smoke and report_dir is None:
        issues.append(
            "--smoke requires --report-dir sidecars for deterministic validation"
        )

    sidecar_digests = {
        name: owner._file_sha256(report_dir / name)
        for name in REQUIRED_DEPLOYABLE_SIDECAR_SCHEMAS
        if report_dir is not None and (report_dir / name).is_file()
    }
    _publication_issues(
        owner,
        require_publication=require_publication,
        report_dir=report_dir,
        expected_identity=expected_identity,
        baseline_identities=baseline_identities,
        expected_bits=expected_bits,
        trust_remote_code=trust_remote_code,
        sidecar_digests=sidecar_digests,
        issues=issues,
    )

    ok = not issues
    validation_scope = (
        DEPLOYABLE_RUNTIME_REPROOF_SCOPE
        if smoke
        else DEPLOYABLE_STRUCTURAL_VALIDATION_SCOPE
    )
    runtime_proof_authoritative = bool(smoke and isinstance(runtime_proof, dict) and ok)
    proof_baseline_identity = expected_baseline_identity or next(
        (
            payload.get("baseline_identity")
            for payload in sidecar_payloads.values()
            if isinstance(payload.get("baseline_identity"), dict)
        ),
        None,
    )
    return {
        "schema": DEPLOYABLE_VALIDATION_SCHEMA,
        "ok": ok,
        "backend": resolved_backend or None,
        "backend_version": backend_version,
        "artifact_class": DEPLOYABLE_OPTIMIZED_SUBJECT,
        "artifact_identity": expected_identity,
        "baseline_identity": proof_baseline_identity,
        "bits": expected_bits,
        "trust_remote_code": trust_remote_code,
        "validation_scope": validation_scope,
        "runtime_proof_authoritative": runtime_proof_authoritative,
        "runtime_proof": runtime_proof,
        "sidecar_digests": sidecar_digests,
        "load_smoke": load_smoke,
        "inference_smoke": inference_smoke,
        "packed_quantized_storage": metadata.get("packed_quantized_storage") is True,
        "runtime_memory_reduction_observed": bool(
            sidecar_payloads.get("memory_report.json", {}).get(
                "runtime_memory_reduction_observed"
            )
            or metadata.get("runtime_memory_reduction")
        ),
        "issues": issues,
    }
