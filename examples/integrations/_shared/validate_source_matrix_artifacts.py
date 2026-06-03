from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MATRIX = Path("examples/integrations/source_matrix.json")
BACKEND_INVENTORY_SCHEMA = "invarlock/backend-inventory-v1"


@dataclass(frozen=True)
class ValidationIssue:
    target: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"target": self.target, "path": self.path, "message": self.message}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_dir(repo_root: Path, entry: dict[str, Any]) -> Path:
    readme_parent = (repo_root / str(entry["readme"])).parent
    report_path = str(entry["report_path"]).replace(
        "<artifact-lane>", str(entry["lane"])
    )
    return readme_parent / report_path


def _verify_status(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    if summary.get("status"):
        return str(summary["status"])
    if summary.get("reason"):
        return str(summary["reason"])
    if summary.get("ok") is True:
        return "ok"
    return None


def _runtime_provenance(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    results = payload.get("results")
    if not isinstance(results, list):
        return {}
    for result in results:
        if not isinstance(result, dict):
            continue
        verification = result.get("verification")
        if not isinstance(verification, dict):
            continue
        runtime = verification.get("runtime_provenance")
        if isinstance(runtime, dict):
            return runtime
    return {}


def _summary_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def _read_required_json_object(
    *,
    target: str,
    path: Path,
    artifact_name: str,
) -> tuple[dict[str, Any] | None, ValidationIssue | None]:
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as exc:
        return None, ValidationIssue(
            target=target,
            path=str(path),
            message=f"{artifact_name} is invalid JSON: {exc.msg}",
        )
    if not isinstance(payload, dict):
        return None, ValidationIssue(
            target=target,
            path=str(path),
            message=f"{artifact_name} must contain a JSON object",
        )
    return payload, None


def _validate_backend_inventory(
    *,
    target: str,
    path: Path,
    payload: dict[str, Any],
    expected_adapter: str | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if payload.get("schema") != BACKEND_INVENTORY_SCHEMA:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message=(
                    "backend inventory schema mismatch: "
                    f"expected {BACKEND_INVENTORY_SCHEMA!r}, "
                    f"got {payload.get('schema')!r}"
                ),
            )
        )
    if expected_adapter and payload.get("adapter") != expected_adapter:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message=(
                    "backend inventory adapter mismatch: "
                    f"expected {expected_adapter!r}, got {payload.get('adapter')!r}"
                ),
            )
        )
    quantized_count = payload.get("quantized_module_count")
    if not isinstance(quantized_count, int) or quantized_count < 0:
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message=(
                    "backend inventory quantized_module_count must be a "
                    "non-negative integer"
                ),
            )
        )
    return issues


def _validate_runtime_manifest(
    *,
    target: str,
    path: Path,
    payload: dict[str, Any],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message="runtime manifest runtime field must contain an object",
            )
        )
        runtime = {}
    if not str(runtime.get("image_digest") or "").strip():
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message="runtime manifest runtime.image_digest must be present",
            )
        )
    if not str(runtime.get("image_ref") or "").strip():
        issues.append(
            ValidationIssue(
                target=target,
                path=str(path),
                message="runtime manifest runtime.image_ref must be present",
            )
        )
    return issues


def validate_entry(repo_root: Path, entry: dict[str, Any]) -> list[ValidationIssue]:
    target = str(entry["target"])
    issues: list[ValidationIssue] = []
    report_dir = _report_dir(repo_root, entry)
    expected = entry.get("expected", {})

    if not report_dir.is_dir():
        return [
            ValidationIssue(
                target=target,
                path=str(report_dir),
                message="report directory is missing",
            )
        ]

    artifacts = entry.get("required_artifacts", [])
    if not isinstance(artifacts, list):
        return [
            ValidationIssue(
                target=target,
                path=str(report_dir),
                message="required_artifacts is not a list",
            )
        ]

    for artifact in artifacts:
        artifact_path = report_dir / str(artifact)
        if not artifact_path.is_file():
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(artifact_path),
                    message="required artifact is missing",
                )
            )
            continue
        if artifact_path.stat().st_size <= 0:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(artifact_path),
                    message="required artifact is empty",
                )
            )
            continue
        if artifact_path.suffix == ".json":
            try:
                _read_json(artifact_path)
            except json.JSONDecodeError as exc:
                issues.append(
                    ValidationIssue(
                        target=target,
                        path=str(artifact_path),
                        message=f"required JSON artifact is invalid: {exc.msg}",
                    )
                )

    lane_artifact_path = report_dir / "lane_artifact.json"
    if lane_artifact_path.is_file():
        lane_label: str | None = None
        lane_artifact, lane_error = _read_required_json_object(
            target=target,
            path=lane_artifact_path,
            artifact_name="lane artifact",
        )
        if lane_error is not None:
            issues.append(lane_error)
        elif lane_artifact is not None:
            lane_label = (
                str(lane_artifact["lane_artifact_label"])
                if "lane_artifact_label" in lane_artifact
                else None
            )
        if lane_error is None and lane_label != expected.get("lane_artifact_label"):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(lane_artifact_path),
                    message=(
                        "lane artifact label mismatch: "
                        f"expected {expected.get('lane_artifact_label')!r}, "
                        f"got {lane_label!r}"
                    ),
                )
            )

    summary_path = report_dir / "run_summary.txt"
    if summary_path.is_file():
        summary = _summary_fields(summary_path)
        if summary.get("status") != "success":
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(summary_path),
                    message="run summary does not record status: success",
                )
            )
        if summary.get("lane_artifact_label") != expected.get("lane_artifact_label"):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(summary_path),
                    message="run summary lane label does not match matrix",
                )
            )

    verify_path = report_dir / "verify.json"
    if verify_path.is_file():
        verify_payload, verify_error = _read_required_json_object(
            target=target,
            path=verify_path,
            artifact_name="verify artifact",
        )
        if verify_error is not None or verify_payload is None:
            if verify_error is not None:
                issues.append(verify_error)
            verify_payload = {}
        status = _verify_status(verify_payload)
        if status != expected.get("verify_status"):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(verify_path),
                    message=(
                        "verify status mismatch: "
                        f"expected {expected.get('verify_status')!r}, got {status!r}"
                    ),
                )
            )

        runtime = _runtime_provenance(verify_payload)
        declared = runtime.get("declared_mode")
        if declared != expected.get("runtime_provenance_declared"):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(verify_path),
                    message=(
                        "runtime provenance declared mode mismatch: "
                        f"expected {expected.get('runtime_provenance_declared')!r}, "
                        f"got {declared!r}"
                    ),
                )
            )
        if runtime.get("verified") != expected.get("runtime_provenance_verified"):
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(verify_path),
                    message=(
                        "runtime provenance verified flag mismatch: "
                        f"expected {expected.get('runtime_provenance_verified')!r}, "
                        f"got {runtime.get('verified')!r}"
                    ),
                )
            )

    backend_inventory_path = report_dir / "backend_inventory.json"
    if backend_inventory_path.is_file():
        expected_adapter = entry.get("subject_adapter")
        backend_inventory, backend_error = _read_required_json_object(
            target=target,
            path=backend_inventory_path,
            artifact_name="backend inventory",
        )
        if backend_error is not None:
            issues.append(backend_error)
        elif backend_inventory is not None:
            issues.extend(
                _validate_backend_inventory(
                    target=target,
                    path=backend_inventory_path,
                    payload=backend_inventory,
                    expected_adapter=(
                        expected_adapter
                        if isinstance(expected_adapter, str)
                        else None
                    ),
                )
            )

    runtime_manifest_path = report_dir / "runtime.manifest.json"
    runtime_image = entry.get("runtime_image")
    if (
        runtime_manifest_path.is_file()
        and isinstance(runtime_image, dict)
        and runtime_image.get("digest_source") == "runtime.manifest.json"
    ):
        runtime_manifest, runtime_error = _read_required_json_object(
            target=target,
            path=runtime_manifest_path,
            artifact_name="runtime manifest",
        )
        if runtime_error is not None:
            issues.append(runtime_error)
        elif runtime_manifest is not None:
            issues.extend(
                _validate_runtime_manifest(
                    target=target,
                    path=runtime_manifest_path,
                    payload=runtime_manifest,
                )
            )

    return issues


def validate_matrix(
    *,
    repo_root: Path,
    matrix_path: Path,
    targets: set[str] | None = None,
) -> tuple[list[str], list[ValidationIssue]]:
    payload = _read_json(matrix_path)
    if payload.get("schema") != "invarlock.integration_source_matrix.v1":
        raise ValueError("unsupported source matrix schema")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("source matrix entries must be a list")

    selected: list[str] = []
    issues: list[ValidationIssue] = []
    for entry in entries:
        target = str(entry.get("target", ""))
        if targets is not None and target not in targets:
            continue
        selected.append(target)
        issues.extend(validate_entry(repo_root, entry))

    if targets is not None:
        missing = sorted(targets - set(selected))
        for target in missing:
            issues.append(
                ValidationIssue(
                    target=target,
                    path=str(matrix_path),
                    message="target is not present in source matrix",
                )
            )

    return selected, issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate generated integration run artifacts against "
            "examples/integrations/source_matrix.json."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing examples/integrations.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Path to source_matrix.json. Defaults under --repo-root.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional matrix targets to validate. Defaults to all entries.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable validation output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    matrix_path = (
        args.matrix.resolve()
        if args.matrix is not None
        else (repo_root / DEFAULT_MATRIX).resolve()
    )
    targets = set(args.targets) if args.targets is not None else None
    selected, issues = validate_matrix(
        repo_root=repo_root,
        matrix_path=matrix_path,
        targets=targets,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not issues,
                    "matrix": str(matrix_path),
                    "targets": selected,
                    "issues": [issue.as_dict() for issue in issues],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        if issues:
            print("Source matrix artifact validation failed:")
            for issue in issues:
                print(f"- {issue.target}: {issue.path}: {issue.message}")
        else:
            print("Source matrix artifact validation passed for " + ", ".join(selected))

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
