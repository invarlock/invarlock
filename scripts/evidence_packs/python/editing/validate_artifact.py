from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.evidence_pack_contracts.deployable_coverage import (
    dense_parameter_catalog,
    inspect_bitsandbytes_modules,
    logical_coverage_from_inventory,
)

try:
    from . import artifact_staging
    from .artifact_tensor_validation import _has_tokenizer, _has_valid_weights
    from .validate_deployable import (
        validate_deployable_artifact as _validate_deployable_artifact,
    )
    from .validate_pruning import (
        validate_pruning_artifact as _validate_pruning_artifact,
    )
    from .validate_transformation import (
        validate_transformation_artifact as _validate_transformation_artifact,
    )
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from editing import artifact_staging
    from editing.artifact_tensor_validation import _has_tokenizer, _has_valid_weights
    from editing.validate_deployable import (
        validate_deployable_artifact as _validate_deployable_artifact,
    )
    from editing.validate_pruning import (
        validate_pruning_artifact as _validate_pruning_artifact,
    )
    from editing.validate_transformation import (
        validate_transformation_artifact as _validate_transformation_artifact,
    )


try:
    from ..runtime_tools import require_remote_code_opt_in
except ImportError:  # pragma: no cover - top-level editing package/direct load
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime_tools import require_remote_code_opt_in

try:
    from .implementations import (
        read_edit_metadata,
        validate_edit_metadata,
        write_edit_metadata,
    )
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from editing.implementations import (
        read_edit_metadata,
        validate_edit_metadata,
        write_edit_metadata,
    )

try:
    from safetensors import SafetensorError, safe_open
except ImportError:  # pragma: no cover - optional at import time
    safe_open = None  # type: ignore[assignment]
    SafetensorError = RuntimeError  # type: ignore[misc,assignment]

staging_path_for = artifact_staging.staging_path_for
_remove_staging = artifact_staging.remove_staging
_replace_output = artifact_staging.replace_output
_reset_staging = artifact_staging.reset_staging

DEPLOYABLE_VALIDATION_SCHEMA = "invarlock/deployable-artifact-validation-v1"
DEPLOYABLE_STRUCTURAL_VALIDATION_SCOPE = "structural_only"
DEPLOYABLE_RUNTIME_REPROOF_SCOPE = "runtime_reproof"
PRUNING_MATERIALIZATION_RECEIPT_SCHEMA = "invarlock/pruning-materialization-v1"
PRUNING_MATERIALIZATION_RECEIPT = "pruning_materialization.json"
BACKEND_INVENTORY_SCHEMA = "invarlock/backend-inventory-v1"
MEMORY_REPORT_SCHEMA = "invarlock/deployable-memory-report-v1"
LOAD_SMOKE_SCHEMA = "invarlock/deployable-load-smoke-v1"
INFERENCE_SMOKE_SCHEMA = "invarlock/deployable-inference-smoke-v1"
DEPLOYABLE_SMOKE_PROMPT = "InvarLock quantized checkpoint verification"

REQUIRED_DEPLOYABLE_SIDECAR_SCHEMAS = {
    "backend_inventory.json": BACKEND_INVENTORY_SCHEMA,
    "memory_report.json": MEMORY_REPORT_SCHEMA,
    "load_smoke.json": LOAD_SMOKE_SCHEMA,
    "inference_smoke.json": INFERENCE_SMOKE_SCHEMA,
}

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_PRUNING_REPLAY_WORKERS = 8
_MAX_PRUNING_REPLAY_THREADS = 8


def _resolve_remote_code_request(requested: bool) -> bool:
    """Require both the command flag and the evidence-pack authorization."""

    if not requested:
        return False
    return require_remote_code_opt_in("validate_artifact.py deployable")


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _runtime_bitsandbytes_proof(
    artifact_dir: Path,
    *,
    baseline_dir: Path,
    expected_bits: int,
    trust_remote_code: bool,
) -> dict[str, Any]:
    """Reload the saved checkpoint and observe packed modules plus finite inference."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("deployable runtime smoke requires CUDA")
    before = checkpoint_tree_sha256(artifact_dir)
    baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": checkpoint_tree_sha256(baseline_dir),
    }
    baseline_model = AutoModelForCausalLM.from_pretrained(
        baseline_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    ).eval()
    dense_catalog = dense_parameter_catalog(baseline_model)
    baseline_footprint = int(baseline_model.get_memory_footprint())
    if baseline_footprint <= 0:
        raise RuntimeError("baseline model reported a non-positive runtime footprint")
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(
        artifact_dir, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        artifact_dir,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    ).eval()
    quantized_footprint = int(model.get_memory_footprint())
    if quantized_footprint <= 0 or quantized_footprint >= baseline_footprint:
        raise RuntimeError(
            "runtime smoke did not independently observe memory reduction"
        )
    inventory = inspect_bitsandbytes_modules(model, bits=expected_bits)
    logical_coverage = logical_coverage_from_inventory(dense_catalog, inventory)

    raw_quant = getattr(getattr(model, "config", None), "quantization_config", None)
    to_dict = getattr(raw_quant, "to_dict", None)
    if callable(to_dict):
        raw_quant = to_dict()
    if not isinstance(raw_quant, dict):
        raise RuntimeError("reloaded artifact has no serialized quantization config")
    expected_flag = "load_in_4bit" if expected_bits == 4 else "load_in_8bit"
    opposite_flag = "load_in_8bit" if expected_bits == 4 else "load_in_4bit"
    method = raw_quant.get("quant_method")
    method = str(getattr(method, "value", method) or "").lower()
    if (
        method != "bitsandbytes"
        or raw_quant.get(expected_flag) is not True
        or raw_quant.get(opposite_flag) is True
    ):
        raise RuntimeError("reloaded artifact quantization config bit flags mismatch")

    encoded = tokenizer(DEPLOYABLE_SMOKE_PROMPT, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits.detach().float().cpu().contiguous()
    if logits.numel() <= 0 or not torch.isfinite(logits).all():
        raise RuntimeError("reloaded artifact inference did not produce finite logits")
    after = checkpoint_tree_sha256(artifact_dir)
    if after != before:
        raise RuntimeError("deployable artifact tree changed during runtime smoke")
    return {
        "artifact_identity": {"kind": "local_checkpoint_tree", "sha256": after},
        "baseline_identity": baseline_identity,
        "trust_remote_code": trust_remote_code,
        "quantized_module_count": inventory["count"],
        "quantized_module_names": inventory["names"],
        "quantized_module_names_sha256": inventory["names_sha256"],
        "quantized_module_types": inventory["types"],
        "packed_weight_storage_elements": inventory["packed_weight_storage_elements"],
        "logical_coverage": logical_coverage,
        "logits_sha256": "sha256:"
        + hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
        "logits_shape": list(logits.shape),
        "all_logits_finite": True,
        "load_time_quantization_override": False,
        "baseline_reported_bytes": baseline_footprint,
        "quantized_reported_bytes": quantized_footprint,
        "reduction_bytes": baseline_footprint - quantized_footprint,
        "reduction_ratio": (baseline_footprint - quantized_footprint)
        / baseline_footprint,
        "runtime_memory_reduction_observed": True,
    }


@dataclass
class EditArtifactValidationResult:
    ok: bool
    has_config: bool = False
    has_weights: bool = False
    has_tokenizer: bool = False
    has_metadata: bool = False
    artifact_class: str | None = None
    edit_type: str | None = None
    issues: list[str] | None = None

    def __bool__(self) -> bool:
        return self.ok

    def to_json_payload(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "has_config": self.has_config,
            "has_weights": self.has_weights,
            "has_tokenizer": self.has_tokenizer,
            "has_metadata": self.has_metadata,
            "artifact_class": self.artifact_class,
            "edit_type": self.edit_type,
            "issues": list(self.issues or []),
        }


def validate_pruning_artifact(
    artifact_dir: Path,
    *,
    baseline_dir: Path,
    scope: str,
    target_sparsity: float,
    workers: int = 1,
    worker_threads: int = 0,
) -> dict[str, Any]:
    return _validate_pruning_artifact(
        sys.modules[__name__],
        artifact_dir,
        baseline_dir=baseline_dir,
        scope=scope,
        target_sparsity=target_sparsity,
        workers=workers,
        worker_threads=worker_threads,
    )


def validate_transformation_artifact(
    artifact_dir: Path,
    *,
    baseline_dir: Path,
    edit_type: str,
    parameters: Mapping[str, object],
    scope: str,
) -> dict[str, Any]:
    return _validate_transformation_artifact(
        sys.modules[__name__],
        artifact_dir,
        baseline_dir=baseline_dir,
        edit_type=edit_type,
        parameters=parameters,
        scope=scope,
    )


def validate_edit_artifact(
    edit_path: Path,
    *,
    require_metadata: bool = False,
    expected_edit_type: str | None = None,
    expected_artifact_class: str | None = None,
) -> EditArtifactValidationResult:
    issues: list[str] = []
    if not edit_path.is_dir():
        return EditArtifactValidationResult(
            ok=False,
            issues=[f"edit artifact directory not found: {edit_path}"],
        )

    has_config = (edit_path / "config.json").is_file()
    has_tokenizer = _has_tokenizer(edit_path)
    has_weights = _has_valid_weights(edit_path)

    if not has_config:
        issues.append("config.json missing")
    if not has_tokenizer:
        issues.append("tokenizer files missing")
    if not has_weights:
        issues.append("model weights missing or invalid")

    metadata_path = edit_path / "edit_metadata.json"
    has_metadata = metadata_path.is_file()
    artifact_class: str | None = None
    edit_type: str | None = None
    if require_metadata and not has_metadata:
        issues.append("edit_metadata.json missing")
    if has_metadata:
        try:
            metadata = read_edit_metadata(metadata_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            issues.append(f"edit_metadata.json invalid: {exc}")
        else:
            artifact_class_raw = metadata.get("artifact_class")
            edit_type_raw = metadata.get("edit_type")
            artifact_class = (
                artifact_class_raw if isinstance(artifact_class_raw, str) else None
            )
            edit_type = edit_type_raw if isinstance(edit_type_raw, str) else None
            issues.extend(
                validate_edit_metadata(
                    metadata,
                    expected_edit_type=expected_edit_type,
                    expected_artifact_class=expected_artifact_class,
                )
            )

    return EditArtifactValidationResult(
        ok=not issues,
        has_config=has_config,
        has_weights=has_weights,
        has_tokenizer=has_tokenizer,
        has_metadata=has_metadata,
        artifact_class=artifact_class,
        edit_type=edit_type,
        issues=issues,
    )


def save_edited_subject_artifact(
    *,
    model: Any,
    tokenizer: Any,
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    staging_path = staging_path_for(output_path)
    _reset_staging(staging_path)
    staging_path.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer.save_pretrained(staging_path)
        model.save_pretrained(staging_path, safe_serialization=True)
        write_edit_metadata(staging_path / "edit_metadata.json", metadata)
        result = validate_edit_artifact(
            staging_path,
            require_metadata=True,
            expected_edit_type=str(metadata.get("edit_type") or ""),
            expected_artifact_class=str(metadata.get("artifact_class") or ""),
        )
        if not result.ok:
            raise RuntimeError(
                "saved edit artifact failed validation: "
                + "; ".join(result.issues or [])
            )
        _replace_output(staging_path, output_path)
    except Exception:
        _remove_staging(staging_path, output_path)
        raise


def validate_deployable_artifact(
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
    return _validate_deployable_artifact(
        sys.modules[__name__],
        artifact_dir,
        backend=backend,
        report_dir=report_dir,
        smoke=smoke,
        expected_bits=expected_bits,
        trust_remote_code=trust_remote_code,
        require_publication=require_publication,
        baseline_dir=baseline_dir,
    )


def _validate_deployable_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate a deployable edit artifact.")
    parser.add_argument("artifact_dir")
    parser.add_argument("--backend")
    parser.add_argument("--report-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--expected-bits", type=int, choices=(4, 8))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--require-publication", action="store_true")
    parser.add_argument("--baseline")
    parser.add_argument("--out")
    parser.add_argument("--json", action="store_true", dest="json_out")
    args = parser.parse_args(argv)

    payload = validate_deployable_artifact(
        Path(args.artifact_dir),
        backend=args.backend,
        report_dir=Path(args.report_dir) if args.report_dir else None,
        smoke=bool(args.smoke),
        expected_bits=args.expected_bits,
        trust_remote_code=bool(args.trust_remote_code),
        require_publication=bool(args.require_publication),
        baseline_dir=Path(args.baseline) if args.baseline else None,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.json_out or not args.out:
        print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("ok") is True else 1


def _validate_pruning_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a magnitude-pruned artifact."
    )
    parser.add_argument("artifact_dir")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--target-sparsity", required=True, type=float)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="bounded concurrent tensor replay workers (default: 1; maximum: 8)",
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=0,
        help="Torch intra-op threads per replay task; 0 preserves the process default.",
    )
    parser.add_argument("--out")
    parser.add_argument("--json", action="store_true", dest="json_out")
    args = parser.parse_args(argv)

    payload = validate_pruning_artifact(
        Path(args.artifact_dir),
        baseline_dir=Path(args.baseline),
        scope=str(args.scope),
        target_sparsity=float(args.target_sparsity),
        workers=int(args.workers),
        worker_threads=int(args.worker_threads),
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.json_out or not args.out:
        print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("ok") is True else 1


def _parse_cli_json_object(raw: str, *, argument_name: str) -> dict[str, object]:
    def no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-standard JSON constant {value!r}")

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=no_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{argument_name} must be a strict JSON object: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{argument_name} must be a JSON object")
    return payload


def _write_transformation_replay_sidecar(
    output_path: Path,
    *,
    payload: Mapping[str, object],
    artifact_dir: Path,
    baseline_dir: Path,
) -> None:
    """Atomically persist replay evidence without invalidating its identities."""

    if output_path.exists() and (output_path.is_symlink() or not output_path.is_file()):
        raise ValueError("transformation replay --out must be a regular file path")
    try:
        resolved_output = output_path.resolve(strict=False)
        protected_roots = (
            artifact_dir.resolve(strict=True),
            baseline_dir.resolve(strict=True),
        )
    except OSError as exc:
        raise ValueError(
            f"could not resolve transformation replay --out path: {exc}"
        ) from exc
    for root in protected_roots:
        try:
            resolved_output.relative_to(root)
        except ValueError:
            continue
        raise ValueError(
            "transformation replay sidecar must be outside the baseline and artifact "
            "trees so its recorded identities remain valid"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists() or temporary_path.is_symlink():
        raise ValueError(
            "transformation replay sidecar temporary path is unexpectedly occupied"
        )
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with temporary_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
        temporary_path.replace(output_path)
    except OSError as exc:
        raise ValueError(
            f"could not write transformation replay sidecar: {exc}"
        ) from exc
    finally:
        if temporary_path.exists() or temporary_path.is_symlink():
            temporary_path.unlink()


def _validate_transformation_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Replay-validate a verifier-grade generated transformation."
    )
    parser.add_argument("artifact_dir")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--edit-type", required=True)
    parser.add_argument(
        "--parameters-json",
        "--parameters",
        dest="parameters_json",
        required=True,
        help="canonical transformation parameters as a strict JSON object",
    )
    parser.add_argument("--scope", required=True)
    parser.add_argument("--out")
    parser.add_argument("--json", action="store_true", dest="json_out")
    args = parser.parse_args(argv)
    try:
        parameters = _parse_cli_json_object(
            str(args.parameters_json), argument_name="--parameters-json"
        )
    except ValueError as exc:
        parser.error(str(exc))

    artifact_dir = Path(args.artifact_dir)
    baseline_dir = Path(args.baseline)
    payload = validate_transformation_artifact(
        artifact_dir,
        baseline_dir=baseline_dir,
        edit_type=str(args.edit_type),
        parameters=parameters,
        scope=str(args.scope),
    )
    if args.out:
        try:
            _write_transformation_replay_sidecar(
                Path(args.out),
                payload=payload,
                artifact_dir=artifact_dir,
                baseline_dir=baseline_dir,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.json_out or not args.out:
        print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0 if payload.get("ok") is True else 1


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "deployable":
        return _validate_deployable_cli(argv[2:])
    if len(argv) > 1 and argv[1] == "pruning":
        return _validate_pruning_cli(argv[2:])
    if len(argv) > 1 and argv[1] == "transform":
        return _validate_transformation_cli(argv[2:])

    parser = argparse.ArgumentParser(
        description="Validate an evidence-pack edit artifact."
    )
    parser.add_argument("edit_path")
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument("--expected-edit-type")
    parser.add_argument("--expected-artifact-class")
    parser.add_argument("--json", action="store_true", dest="json_out")
    args = parser.parse_args(argv[1:])

    result = validate_edit_artifact(
        Path(args.edit_path),
        require_metadata=bool(args.require_metadata),
        expected_edit_type=args.expected_edit_type,
        expected_artifact_class=args.expected_artifact_class,
    )
    if args.json_out:
        print(json.dumps(result.to_json_payload(), sort_keys=True))
    elif not result.ok:
        for issue in result.issues or []:
            print(issue, file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
