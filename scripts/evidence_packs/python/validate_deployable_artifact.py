from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

try:
    from edit_metadata import DEPLOYABLE_OPTIMIZED_SUBJECT, read_edit_metadata
    from validate_edit_artifact import validate_edit_artifact
except ImportError:  # pragma: no cover - direct module load under pytest
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from edit_metadata import DEPLOYABLE_OPTIMIZED_SUBJECT, read_edit_metadata
    from validate_edit_artifact import validate_edit_artifact

DEPLOYABLE_VALIDATION_SCHEMA = "invarlock/deployable-artifact-validation-v1"


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


def _metadata_issues(metadata: dict[str, Any], backend: str | None) -> list[str]:
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
            issues.extend(_metadata_issues(metadata, backend))

    resolved_backend = backend or str(metadata.get("backend") or "")
    backend_version = _package_version(resolved_backend) if resolved_backend else None
    if resolved_backend and backend_version is None:
        issues.append(f"backend package not importable: {resolved_backend}")

    sidecar_checks: dict[str, bool] = {}
    if report_dir is not None:
        for sidecar in (
            "backend_inventory.json",
            "memory_report.json",
            "load_smoke.json",
            "inference_smoke.json",
        ):
            payload = _load_json_object(report_dir / sidecar)
            sidecar_checks[sidecar] = payload is not None
            if payload is None:
                issues.append(f"missing or invalid report sidecar: {sidecar}")

    # This validator is intentionally conservative. Heavy reload/inference smoke
    # should be produced by backend-specific generators and passed as sidecars.
    load_smoke = (
        bool(sidecar_checks.get("load_smoke.json")) if report_dir else not smoke
    )
    inference_smoke = (
        bool(sidecar_checks.get("inference_smoke.json")) if report_dir else not smoke
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
            metadata.get("runtime_memory_reduction")
        ),
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
