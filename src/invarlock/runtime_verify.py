from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jsonschema

from invarlock.public_contracts import load_runtime_manifest_schema
from invarlock.runtime_security import RUNTIME_VERIFIER_CONTRACT_VERSION


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invarlock-runtime-verify",
        description="Verify runtime.manifest.json against an evaluation report.",
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def verify_report_manifest(report_path: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []

    try:
        report_bytes = report_path.read_bytes()
    except OSError as exc:
        return [f"unable to read report: {exc}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [f"unable to read manifest: {exc}"]
    except json.JSONDecodeError as exc:
        return [f"unable to parse manifest: {exc}"]

    if not isinstance(manifest, dict):
        return ["manifest payload must be a JSON object"]

    schema = load_runtime_manifest_schema()
    if not schema:
        return ["runtime manifest schema is unavailable"]
    try:
        jsonschema.validate(instance=manifest, schema=schema)
    except jsonschema.ValidationError as exc:
        return [f"runtime manifest schema validation failed: {exc.message}"]

    contract_version = manifest.get("verifier_contract_version")
    if contract_version != RUNTIME_VERIFIER_CONTRACT_VERSION:
        errors.append(
            f"unexpected verifier contract version: {contract_version or '<missing>'}"
        )

    execution_mode = manifest.get("execution_mode")
    if execution_mode != "container":
        errors.append(
            f'execution_mode must be "container", got {execution_mode or "<missing>"}'
        )

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
    if runtime.get("container_execution") is not True:
        errors.append("runtime.container_execution must be true")
    if not str(runtime.get("image_digest") or "").strip():
        errors.append("runtime.image_digest must be present")

    report = manifest.get("report")
    if not isinstance(report, dict):
        report = {}
    expected_sha = report.get("sha256")
    actual_sha = hashlib.sha256(report_bytes).hexdigest()
    if not isinstance(expected_sha, str) or not expected_sha:
        errors.append("manifest is missing report.sha256")
    elif expected_sha != actual_sha:
        errors.append(
            f"report digest mismatch: manifest={expected_sha} actual={actual_sha}"
        )

    if not report_bytes:
        errors.append("report file is empty")

    return errors


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report_path = Path(args.report)
    manifest_path = Path(args.manifest)
    errors = verify_report_manifest(report_path, manifest_path)
    ok = not errors

    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "errors": errors,
                    "report": str(report_path),
                    "manifest": str(manifest_path),
                }
            )
        )
    elif ok:
        print(f"runtime verify ok report={report_path} manifest={manifest_path}")
    else:
        for error in errors:
            print(error)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
