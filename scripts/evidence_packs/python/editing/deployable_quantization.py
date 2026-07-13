#!/usr/bin/env python3
"""Materialize and prove a packed, reloadable bitsandbytes checkpoint.

This module is deliberately separate from the dense RTN quantize/dequantize
fixtures.  A successful run means that packed backend modules were observed,
the checkpoint was saved and reloaded without supplying a load-time
quantization override, inference completed, and runtime model footprint fell.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.evidence_pack_contracts.deployable_coverage import (
    dense_parameter_catalog,
    inspect_bitsandbytes_modules,
    logical_coverage_from_inventory,
    require_inventory_logical_binding,
    require_inventory_runtime_facts,
    require_logical_coverage,
)

try:
    from ..runtime_tools import require_remote_code_opt_in
    from .implementations import (
        DEPLOYABLE_OPTIMIZED_SUBJECT,
        EDIT_SEMANTICS_DEPLOYABLE,
        build_edit_metadata,
        write_edit_metadata,
    )
    from .validate_artifact import (
        DEPLOYABLE_RUNTIME_REPROOF_SCOPE,
        DEPLOYABLE_SMOKE_PROMPT,
        DEPLOYABLE_STRUCTURAL_VALIDATION_SCOPE,
        validate_deployable_artifact,
    )
except ImportError:  # pragma: no cover - direct CLI/importlib execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime_tools import require_remote_code_opt_in

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from implementations import (
        DEPLOYABLE_OPTIMIZED_SUBJECT,
        EDIT_SEMANTICS_DEPLOYABLE,
        build_edit_metadata,
        write_edit_metadata,
    )
    from validate_artifact import (
        DEPLOYABLE_RUNTIME_REPROOF_SCOPE,
        DEPLOYABLE_SMOKE_PROMPT,
        DEPLOYABLE_STRUCTURAL_VALIDATION_SCOPE,
        validate_deployable_artifact,
    )


BACKEND = "bitsandbytes"
PROOF_SIDECARS = (
    "backend_inventory.json",
    "memory_report.json",
    "load_smoke.json",
    "inference_smoke.json",
)


def _resolve_remote_code_request(requested: bool) -> bool:
    """Require both the command flag and the evidence-pack authorization."""

    if not requested:
        return False
    return require_remote_code_opt_in("deployable_quantization.py")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"publication staging tree contains symlink: {path}")
        if path.is_dir():
            directories.append(path)
            continue
        if not path.is_file():
            raise RuntimeError(f"publication staging tree contains non-file: {path}")
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required backend package is unavailable: {name}") from exc


def require_fresh_outputs(output_path: Path, report_dir: Path) -> None:
    for path in (output_path, report_dir):
        if path.exists():
            raise FileExistsError(
                f"refusing to replace existing quantization evidence output: {path}"
            )


def require_memory_reduction(
    baseline_footprint: int, quantized_footprint: int
) -> dict[str, Any]:
    if (
        isinstance(baseline_footprint, bool)
        or not isinstance(baseline_footprint, int)
        or isinstance(quantized_footprint, bool)
        or not isinstance(quantized_footprint, int)
        or baseline_footprint <= 0
        or quantized_footprint <= 0
    ):
        raise RuntimeError("runtime model footprints must be positive")
    if quantized_footprint >= baseline_footprint:
        raise RuntimeError(
            "packed quantization did not reduce runtime model footprint "
            f"({quantized_footprint} >= {baseline_footprint})"
        )
    reduction = baseline_footprint - quantized_footprint
    return {
        "schema": "invarlock/deployable-memory-report-v1",
        "ok": True,
        "measurement_method": "transformers.get_memory_footprint",
        "baseline_reported_bytes": baseline_footprint,
        "quantized_reported_bytes": quantized_footprint,
        "reduction_bytes": reduction,
        "reduction_ratio": reduction / baseline_footprint,
        "runtime_memory_reduction_observed": True,
    }


def build_bitsandbytes_metadata(
    *,
    bits: int = 8,
    logical_coverage: dict[str, Any],
    runtime_memory_reduction: bool,
) -> dict[str, Any]:
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in {4, 8}:
        raise ValueError("bits must be exactly 4 or 8")
    if runtime_memory_reduction is not True:
        raise ValueError("deployable metadata requires observed memory reduction")
    coverage = require_logical_coverage(logical_coverage)
    return dict(
        build_edit_metadata(
            edit_type=f"bnb_{bits}bit",
            scope="all_backend_supported_linear_modules",
            artifact_class=DEPLOYABLE_OPTIMIZED_SUBJECT,
            edit_semantics=EDIT_SEMANTICS_DEPLOYABLE,
            optimized_deployment_backend=True,
            backend=BACKEND,
            storage_format=f"bitsandbytes_{bits}bit_packed",
            actual_storage_format=f"bitsandbytes_{bits}bit_packed",
            packed_quantized_storage=True,
            runtime_memory_reduction=runtime_memory_reduction,
            runtime_memory_reduction_expected=True,
            parameters={"bits": bits, "quantization_method": BACKEND},
            coverage={
                "edited_tensors": coverage["weight_tensor_count"],
                "edited_params": coverage["parameter_elements"],
                "total_params": coverage["total_unique_parameter_elements"],
                "coverage_ratio": coverage["parameter_elements"]
                / coverage["total_unique_parameter_elements"],
            },
            edit_provenance={
                "edit_family": "deployable_backend_quantization",
                "edit_method": f"transformers_bitsandbytes_{bits}bit_checkpoint",
                "edit_count": coverage["weight_tensor_count"],
                "dynamic_runtime_required": True,
                "synthetic": False,
            },
            extra={
                "quantization_mode": "packed_backend_checkpoint",
                "backend_adapter": "hf_bnb",
                "logical_coverage": dict(coverage),
            },
        )
    )


def write_deployable_sidecars(
    report_dir: Path,
    *,
    backend_version: str,
    transformers_version: str,
    inventory: dict[str, Any],
    logical_coverage: dict[str, Any],
    quantized_footprint: int,
    memory: dict[str, Any],
    load_details: dict[str, Any],
    inference_details: dict[str, Any],
    artifact_identity: dict[str, str],
    baseline_identity: dict[str, str],
    trust_remote_code: bool = False,
    bits: int = 8,
) -> None:
    require_inventory_runtime_facts(
        {
            "quantized_module_count": inventory.get("count"),
            "quantized_module_names": inventory.get("names"),
            "quantized_module_names_sha256": inventory.get("names_sha256"),
            "quantized_module_types": inventory.get("types"),
            "packed_weight_storage_elements": inventory.get(
                "packed_weight_storage_elements"
            ),
        }
    )
    logical_coverage = require_logical_coverage(logical_coverage)
    require_inventory_logical_binding(
        {
            "quantized_module_count": inventory["count"],
            "quantized_module_names": inventory["names"],
            "quantized_module_names_sha256": inventory["names_sha256"],
            "quantized_module_types": inventory["types"],
            "packed_weight_storage_elements": inventory[
                "packed_weight_storage_elements"
            ],
        },
        logical_coverage,
    )
    if memory.get("quantized_reported_bytes") != quantized_footprint:
        raise RuntimeError(
            "memory report and backend inventory quantized footprints disagree"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    binding = {
        "artifact_identity": artifact_identity,
        "baseline_identity": baseline_identity,
        "bits": bits,
        "trust_remote_code": trust_remote_code,
    }
    _write_json(
        report_dir / "backend_inventory.json",
        {
            "schema": "invarlock/backend-inventory-v1",
            "adapter": "hf_bnb",
            "backend": BACKEND,
            "backend_version": backend_version,
            "transformers_version": transformers_version,
            "quantization_config": {
                "quant_method": BACKEND,
                "load_in_8bit": bits == 8,
                "load_in_4bit": bits == 4,
            },
            "quantized_module_count": inventory["count"],
            "quantized_module_names": list(inventory["names"]),
            "quantized_module_names_sha256": inventory["names_sha256"],
            "quantized_module_types": list(inventory["types"]),
            "packed_weight_storage_elements": inventory[
                "packed_weight_storage_elements"
            ],
            "logical_coverage": dict(logical_coverage),
            "device_map": "auto",
            "memory_footprint": {
                "reported_bytes": quantized_footprint,
                "method": "get_memory_footprint",
            },
            "load_smoke": True,
            "inference_smoke": True,
            **binding,
        },
    )
    _write_json(report_dir / "memory_report.json", {**memory, **binding})
    _write_json(
        report_dir / "load_smoke.json",
        {
            "schema": "invarlock/deployable-load-smoke-v1",
            "ok": True,
            "backend": BACKEND,
            "quantized_module_count": inventory["count"],
            "quantized_module_names": list(inventory["names"]),
            "quantized_module_names_sha256": inventory["names_sha256"],
            "quantized_module_types": list(inventory["types"]),
            "packed_weight_storage_elements": inventory[
                "packed_weight_storage_elements"
            ],
            "logical_coverage": dict(logical_coverage),
            **load_details,
            **binding,
        },
    )
    _write_json(
        report_dir / "inference_smoke.json",
        {
            "schema": "invarlock/deployable-inference-smoke-v1",
            "ok": True,
            "backend": BACKEND,
            **inference_details,
            **binding,
        },
    )


def _model_footprint(model: Any) -> int:
    method = getattr(model, "get_memory_footprint", None)
    if not callable(method):
        raise RuntimeError("model does not expose get_memory_footprint")
    footprint = int(method())
    if footprint <= 0:
        raise RuntimeError("model reported a non-positive runtime footprint")
    return footprint


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _config_quantization_payload(model: Any, *, bits: int = 8) -> dict[str, Any]:
    payload = getattr(getattr(model, "config", None), "quantization_config", None)
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
    if not isinstance(payload, dict):
        raise RuntimeError("reloaded checkpoint has no serialized quantization_config")
    method_value = payload.get("quant_method")
    method = str(getattr(method_value, "value", method_value) or "").lower()
    bit_flag = "load_in_8bit" if bits == 8 else "load_in_4bit"
    opposite_flag = "load_in_4bit" if bits == 8 else "load_in_8bit"
    if (
        method != BACKEND
        or payload.get(bit_flag) is not True
        or payload.get(opposite_flag) is True
    ):
        raise RuntimeError(
            "reloaded checkpoint quantization_config does not identify "
            f"bitsandbytes {bits}-bit packed storage"
        )
    return payload


def _inference_smoke(model: Any, tokenizer: Any, prompt: str) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        output = model(**inputs)
    logits = output.logits.detach().float().cpu().contiguous()
    if logits.numel() <= 0 or not torch.isfinite(logits).all():
        raise RuntimeError("quantized checkpoint inference returned invalid logits")
    digest = hashlib.sha256(logits.numpy().tobytes()).hexdigest()
    return {
        "prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode()).hexdigest(),
        "logits_sha256": "sha256:" + digest,
        "logits_shape": list(logits.shape),
        "all_logits_finite": True,
    }


def promote_staged_outputs(
    artifact_stage: Path,
    report_stage: Path,
    output_path: Path,
    report_dir: Path,
) -> None:
    publication_path = report_stage / "publication_commit.json"
    try:
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "deployable publication commit marker is missing or invalid"
        ) from exc
    if not isinstance(publication, dict) or publication.get("committed") is not True:
        raise RuntimeError("deployable publication commit marker is not committed")
    if (
        publication.get("validation_scope") != DEPLOYABLE_STRUCTURAL_VALIDATION_SCOPE
        or publication.get("runtime_proof_authoritative") is not False
    ):
        raise RuntimeError(
            "deployable publication commit marker must be structural and "
            "non-authoritative"
        )
    _fsync_tree(artifact_stage)
    _fsync_tree(report_stage)
    _fsync_directory(output_path.parent)
    if report_dir.parent != output_path.parent:
        _fsync_directory(report_dir.parent)
    artifact_promoted = False
    try:
        artifact_stage.rename(output_path)
        artifact_promoted = True
        _fsync_directory(output_path.parent)
        report_stage.rename(report_dir)
        _fsync_directory(report_dir.parent)
    except Exception:
        if artifact_promoted and output_path.exists():
            try:
                output_path.rename(artifact_stage)
                _fsync_directory(output_path.parent)
            except Exception:
                shutil.rmtree(output_path, ignore_errors=True)
        raise


def recover_interrupted_publication(
    output_path: Path,
    report_dir: Path,
    *,
    baseline_path: Path,
    bits: int,
    trust_remote_code: bool,
) -> dict[str, Any] | None:
    """Recover only after a fresh runtime reproof of a staged publication."""

    trust_remote_code = _resolve_remote_code_request(trust_remote_code)

    if not output_path.exists() and not report_dir.exists():
        return None
    if not output_path.is_dir():
        raise FileExistsError(
            f"quantized artifact path is not a directory: {output_path}"
        )

    candidates: list[Path]
    if report_dir.is_dir():
        candidates = [report_dir]
    elif report_dir.exists():
        raise FileExistsError(f"quantized proof path is not a directory: {report_dir}")
    else:
        candidates = sorted(report_dir.parent.glob(f".{report_dir.name}.staging-*"))
    valid: list[tuple[Path, dict[str, Any]]] = []
    requested_baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": checkpoint_tree_sha256(baseline_path),
    }
    for candidate in candidates:
        marker_path = candidate / "publication_commit.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(marker, dict) or marker.get("committed") is not True:
            continue
        observed_identity = {
            "kind": "local_checkpoint_tree",
            "sha256": checkpoint_tree_sha256(output_path),
        }
        if (
            marker.get("artifact_identity") != observed_identity
            or marker.get("baseline_identity") != requested_baseline_identity
            or marker.get("bits") != bits
            or marker.get("trust_remote_code") is not trust_remote_code
        ):
            continue
        structural_validation = validate_deployable_artifact(
            output_path,
            backend=BACKEND,
            report_dir=candidate,
            smoke=False,
            expected_bits=bits,
            trust_remote_code=trust_remote_code,
            require_publication=True,
            baseline_dir=baseline_path,
        )
        if (
            structural_validation.get("ok") is not True
            or structural_validation.get("validation_scope")
            != DEPLOYABLE_STRUCTURAL_VALIDATION_SCOPE
            or structural_validation.get("runtime_proof_authoritative") is not False
        ):
            continue
        runtime_validation = validate_deployable_artifact(
            output_path,
            backend=BACKEND,
            report_dir=candidate,
            smoke=True,
            expected_bits=bits,
            trust_remote_code=trust_remote_code,
            require_publication=True,
            baseline_dir=baseline_path,
        )
        if (
            runtime_validation.get("ok") is True
            and runtime_validation.get("validation_scope")
            == DEPLOYABLE_RUNTIME_REPROOF_SCOPE
            and runtime_validation.get("runtime_proof_authoritative") is True
            and isinstance(runtime_validation.get("runtime_proof"), dict)
        ):
            valid.append((candidate, runtime_validation))

    if len(valid) != 1:
        raise FileExistsError(
            "refusing ambiguous or unproven interrupted quantization publication"
        )
    recovered_report, validation = valid[0]
    _write_json_atomically(
        recovered_report / "runtime_deployability_validation.json", validation
    )
    if recovered_report != report_dir:
        _fsync_tree(output_path)
        _fsync_tree(recovered_report)
        recovered_report.rename(report_dir)
        _fsync_directory(report_dir.parent)
    return validation


def materialize_bitsandbytes_checkpoint(
    baseline_path: Path,
    output_path: Path,
    report_dir: Path,
    *,
    trust_remote_code: bool,
    bits: int = 8,
) -> dict[str, Any]:
    trust_remote_code = _resolve_remote_code_request(trust_remote_code)
    recovered = recover_interrupted_publication(
        output_path,
        report_dir,
        baseline_path=baseline_path,
        bits=bits,
        trust_remote_code=trust_remote_code,
    )
    if recovered is not None:
        return recovered
    require_fresh_outputs(output_path, report_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("bitsandbytes deployable quantization requires CUDA")
    backend_version = _package_version(BACKEND)
    transformers_version = _package_version("transformers")
    if bits not in {4, 8}:
        raise ValueError("bits must be 4 or 8")
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "transformers bitsandbytes integration is unavailable"
        ) from exc

    artifact_stage = output_path.with_name(f".{output_path.name}.staging-{os.getpid()}")
    report_stage = report_dir.with_name(f".{report_dir.name}.staging-{os.getpid()}")
    for path in (artifact_stage, report_stage):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    try:
        baseline_identity = {
            "kind": "local_checkpoint_tree",
            "sha256": checkpoint_tree_sha256(baseline_path),
        }
        tokenizer = AutoTokenizer.from_pretrained(
            baseline_path, trust_remote_code=trust_remote_code
        )
        dense_model = AutoModelForCausalLM.from_pretrained(
            baseline_path,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        ).eval()
        dense_catalog = dense_parameter_catalog(dense_model)
        baseline_footprint = _model_footprint(dense_model)
        del dense_model
        _clear_cuda()

        quant_config = BitsAndBytesConfig(
            load_in_8bit=bits == 8,
            load_in_4bit=bits == 4,
        )
        quantized_model = AutoModelForCausalLM.from_pretrained(
            baseline_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
            quantization_config=quant_config,
        ).eval()
        inventory = inspect_bitsandbytes_modules(quantized_model, bits=bits)
        logical_coverage = logical_coverage_from_inventory(dense_catalog, inventory)
        quantized_footprint = _model_footprint(quantized_model)
        require_memory_reduction(baseline_footprint, quantized_footprint)
        metadata = build_bitsandbytes_metadata(
            bits=bits,
            logical_coverage=logical_coverage,
            runtime_memory_reduction=True,
        )
        tokenizer.save_pretrained(artifact_stage)
        quantized_model.save_pretrained(artifact_stage, safe_serialization=True)
        write_edit_metadata(artifact_stage / "edit_metadata.json", metadata)
        artifact_identity = {
            "kind": "local_checkpoint_tree",
            "sha256": checkpoint_tree_sha256(artifact_stage),
        }
        del quantized_model
        _clear_cuda()

        reloaded = AutoModelForCausalLM.from_pretrained(
            artifact_stage,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        ).eval()
        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            artifact_stage, trust_remote_code=trust_remote_code
        )
        reloaded_inventory = inspect_bitsandbytes_modules(reloaded, bits=bits)
        reloaded_logical_coverage = logical_coverage_from_inventory(
            dense_catalog, reloaded_inventory
        )
        if reloaded_logical_coverage != logical_coverage:
            raise RuntimeError(
                "reloaded packed modules changed logical quantization coverage"
            )
        quant_payload = _config_quantization_payload(reloaded, bits=bits)
        inference_details = _inference_smoke(
            reloaded, reloaded_tokenizer, DEPLOYABLE_SMOKE_PROMPT
        )
        reloaded_footprint = _model_footprint(reloaded)
        reloaded_memory = require_memory_reduction(
            baseline_footprint, reloaded_footprint
        )

        write_deployable_sidecars(
            report_stage,
            backend_version=backend_version,
            transformers_version=transformers_version,
            inventory=reloaded_inventory,
            logical_coverage=reloaded_logical_coverage,
            quantized_footprint=reloaded_footprint,
            memory=reloaded_memory,
            load_details={
                "loaded_from_saved_checkpoint": True,
                "load_time_quantization_override": False,
                "config_quant_method": quant_payload.get("quant_method"),
            },
            inference_details=inference_details,
            artifact_identity=artifact_identity,
            baseline_identity=baseline_identity,
            trust_remote_code=trust_remote_code,
            bits=bits,
        )
        validation = validate_deployable_artifact(
            artifact_stage,
            backend=BACKEND,
            report_dir=report_stage,
            smoke=False,
            expected_bits=bits,
            trust_remote_code=trust_remote_code,
        )
        if validation.get("ok") is not True:
            raise RuntimeError(
                "deployable checkpoint contract failed: "
                + "; ".join(str(item) for item in validation.get("issues", []))
            )
        if (
            validation.get("validation_scope") != DEPLOYABLE_STRUCTURAL_VALIDATION_SCOPE
            or validation.get("runtime_proof_authoritative") is not False
        ):
            raise RuntimeError(
                "prepublication deployable validation must be structural and "
                "non-authoritative"
            )
        validation["sidecar_digests"] = {
            name: _file_sha256(report_stage / name) for name in PROOF_SIDECARS
        }
        validation["baseline_identity"] = baseline_identity
        _write_json(report_stage / "deployable_artifact_validation.json", validation)
        _write_json(
            report_stage / "publication_commit.json",
            {
                "schema": "invarlock/deployable-publication-commit-v1",
                "committed": True,
                "validation_scope": DEPLOYABLE_STRUCTURAL_VALIDATION_SCOPE,
                "runtime_proof_authoritative": False,
                "artifact_identity": artifact_identity,
                "baseline_identity": baseline_identity,
                "bits": bits,
                "trust_remote_code": trust_remote_code,
                "proof_validation_sha256": _file_sha256(
                    report_stage / "deployable_artifact_validation.json"
                ),
                "sidecar_digests": validation["sidecar_digests"],
            },
        )
        del reloaded
        _clear_cuda()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_dir.parent.mkdir(parents=True, exist_ok=True)
        promote_staged_outputs(
            artifact_stage,
            report_stage,
            output_path,
            report_dir,
        )
        return dict(validation)
    except Exception:
        for path in (artifact_stage, report_stage):
            if path.exists():
                shutil.rmtree(path)
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and prove a reloadable bitsandbytes 8-bit checkpoint."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    validation = materialize_bitsandbytes_checkpoint(
        Path(args.baseline),
        Path(args.output),
        Path(args.report_dir),
        trust_remote_code=bool(args.trust_remote_code),
        bits=int(args.bits),
    )
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
