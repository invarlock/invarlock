from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MATRIX = Path("examples/integrations/source_matrix.json")


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
        lane_artifact = _read_json(lane_artifact_path)
        lane_label = lane_artifact.get("lane_artifact_label")
        if lane_label != expected.get("lane_artifact_label"):
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
        verify_payload = _read_json(verify_path)
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
