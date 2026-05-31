from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .implementations import (
        DEPLOYABLE_OPTIMIZED_SUBJECT,
        read_edit_metadata,
        validate_edit_metadata,
        write_edit_metadata,
    )
except ImportError:  # pragma: no cover - direct module load under pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from editing.implementations import (
        DEPLOYABLE_OPTIMIZED_SUBJECT,
        read_edit_metadata,
        validate_edit_metadata,
        write_edit_metadata,
    )

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - optional at import time
    safe_open = None  # type: ignore[assignment]

DEPLOYABLE_VALIDATION_SCHEMA = "invarlock/deployable-artifact-validation-v1"
BACKEND_INVENTORY_SCHEMA = "invarlock/backend-inventory-v1"
MEMORY_REPORT_SCHEMA = "invarlock/deployable-memory-report-v1"
LOAD_SMOKE_SCHEMA = "invarlock/deployable-load-smoke-v1"
INFERENCE_SMOKE_SCHEMA = "invarlock/deployable-inference-smoke-v1"

REQUIRED_DEPLOYABLE_SIDECAR_SCHEMAS = {
    "backend_inventory.json": BACKEND_INVENTORY_SCHEMA,
    "memory_report.json": MEMORY_REPORT_SCHEMA,
    "load_smoke.json": LOAD_SMOKE_SCHEMA,
    "inference_smoke.json": INFERENCE_SMOKE_SCHEMA,
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


def _has_tokenizer(edit_path: Path) -> bool:
    return any(
        (edit_path / name).is_file()
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "special_tokens_map.json",
        )
    )


def _validate_safetensors(path: Path) -> bool:
    if safe_open is None:
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return any(True for _ in handle.keys())
    except Exception:
        return False


def _validate_index_shards(edit_path: Path, index_path: Path) -> bool:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False

    shard_names = sorted({str(name) for name in weight_map.values() if str(name)})
    if not shard_names:
        return False

    for shard_name in shard_names:
        shard_path = edit_path / shard_name
        if not shard_path.is_file():
            return False
        if shard_path.suffix == ".safetensors" and not _validate_safetensors(
            shard_path
        ):
            return False
    return True


def _has_valid_weights(edit_path: Path) -> bool:
    single_safe = edit_path / "model.safetensors"
    safe_index = edit_path / "model.safetensors.index.json"
    single_bin = edit_path / "pytorch_model.bin"
    bin_index = edit_path / "pytorch_model.bin.index.json"

    if single_safe.is_file():
        return _validate_safetensors(single_safe)
    if safe_index.is_file():
        return _validate_index_shards(edit_path, safe_index)
    if single_bin.is_file():
        return True
    if bin_index.is_file():
        return _validate_index_shards(edit_path, bin_index)
    return False


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


def staging_path_for(output_path: Path) -> Path:
    return output_path.parent / f".{output_path.name}.tmp"


def backup_path_for(output_path: Path) -> Path:
    return output_path.parent / f".{output_path.name}.bak"


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _replace_output(staging_path: Path, output_path: Path) -> None:
    backup_path = backup_path_for(output_path)
    if backup_path.exists():
        _remove_path(backup_path)

    moved_existing = False
    try:
        if output_path.exists():
            output_path.rename(backup_path)
            moved_existing = True
        staging_path.rename(output_path)
    except Exception:
        if output_path.exists():
            _remove_path(output_path)
        if moved_existing and backup_path.exists():
            backup_path.rename(output_path)
        raise

    if backup_path.exists():
        _remove_path(backup_path)


def _remove_staging(staging_path: Path, output_path: Path) -> None:
    if staging_path.exists():
        shutil.rmtree(staging_path)
    backup_path = backup_path_for(output_path)
    if backup_path.exists() and not output_path.exists():
        backup_path.rename(output_path)


def _reset_staging(staging_path: Path) -> None:
    if staging_path.exists():
        shutil.rmtree(staging_path)


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
                "saved edit artifact failed validation: " + "; ".join(result.issues)
            )
        _replace_output(staging_path, output_path)
    except Exception:
        _remove_staging(staging_path, output_path)
        raise


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
        quantized_count = payload.get("quantized_module_count")
        if not isinstance(quantized_count, int) or quantized_count < 0:
            issues.append(f"{sidecar} quantized_module_count must be non-negative int")
        module_types = payload.get("quantized_module_types")
        if not isinstance(module_types, list):
            issues.append(f"{sidecar} quantized_module_types must be a list")
        memory_footprint = payload.get("memory_footprint")
        if not isinstance(memory_footprint, dict):
            issues.append(f"{sidecar} memory_footprint must be an object")
        return issues

    if payload.get("ok") is not True:
        issues.append(f"{sidecar} ok must be true")
    return issues


def validate_deployable_artifact(
    artifact_dir: Path,
    *,
    backend: str | None = None,
    report_dir: Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    issues: list[str] = []
    metadata_path = artifact_dir / "edit_metadata.json"
    metadata: dict[str, Any] = {}

    artifact_result = validate_edit_artifact(
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

    ok = not issues
    return {
        "schema": DEPLOYABLE_VALIDATION_SCHEMA,
        "ok": ok,
        "backend": resolved_backend or None,
        "backend_version": backend_version,
        "artifact_class": DEPLOYABLE_OPTIMIZED_SUBJECT,
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


def _validate_deployable_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate a deployable edit artifact.")
    parser.add_argument("artifact_dir")
    parser.add_argument("--backend")
    parser.add_argument("--report-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--json", action="store_true", dest="json_out")
    args = parser.parse_args(argv)

    payload = validate_deployable_artifact(
        Path(args.artifact_dir),
        backend=args.backend,
        report_dir=Path(args.report_dir) if args.report_dir else None,
        smoke=bool(args.smoke),
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.json_out or not args.out:
        print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("ok") is True else 1


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "deployable":
        return _validate_deployable_cli(argv[2:])

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
