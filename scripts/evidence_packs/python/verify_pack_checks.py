from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from scripts.evidence_workflows.workflow_state import (
        WorkflowVerificationSummary,
        write_verification_summary,
    )
except ImportError:  # pragma: no cover - direct script execution path
    _SCRIPTS_DIR = Path(__file__).resolve().parents[2]
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from evidence_workflows.workflow_state import (
        WorkflowVerificationSummary,
        write_verification_summary,
    )

CONTROL_FILES = {
    "checksums.sha256",
    "manifest.json",
    "manifest.signature.json",
    "metadata/manifest.json",
    "metadata/manifest.signature.json",
    "metadata/checksums.sha256",
}
CONTROL_FILE_MIRRORS = {
    "manifest.json": "metadata/manifest.json",
    "manifest.signature.json": "metadata/manifest.signature.json",
    "checksums.sha256": "metadata/checksums.sha256",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_pack_path(value: str) -> str:
    path = value.strip()
    if path.startswith("*"):
        path = path[1:]
    if path.startswith("./"):
        path = path[2:]
    return path


def _checksum_paths(path: Path) -> set[str]:
    paths: set[str] = set()
    if not path.is_file():
        return paths
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 1:
            paths.add(_normalize_pack_path(parts[0]))
            continue
        paths.add(_normalize_pack_path(parts[1]))
    return {path for path in paths if path}


def _is_transport_artifact(rel_path: str) -> bool:
    return (
        rel_path == ".DS_Store"
        or rel_path.endswith("/.DS_Store")
        or rel_path.startswith("._")
        or "/._" in rel_path
        or rel_path.startswith("__MACOSX/")
    )


def _actual_pack_files(pack_dir: Path) -> set[str]:
    files: set[str] = set()
    for path in pack_dir.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(pack_dir).as_posix()
        if _is_transport_artifact(rel_path):
            continue
        files.add(rel_path)
    return files


def cmd_manifest_field(args: argparse.Namespace) -> int:
    payload = _load_json(args.manifest)
    if not isinstance(payload, dict):
        return 1
    value = payload.get(args.field)
    if value is None:
        return 1
    if isinstance(value, str):
        print(value)
    else:
        print(str(value))
    return 0


def cmd_path_within(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    candidate = args.candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return 1
    return 0


def cmd_scenario_strictness(args: argparse.Namespace) -> int:
    strictness = _scenario_strictness(args.scenarios, args.scenario_id)
    if strictness:
        print(strictness)
        return 0
    return 1


def cmd_extra_files(args: argparse.Namespace) -> int:
    pack_dir = args.pack_dir.resolve()
    mirror_errors = _control_file_mirror_errors(pack_dir)
    if mirror_errors:
        for error in mirror_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    expected = _checksum_paths(pack_dir / "checksums.sha256") | CONTROL_FILES
    actual = _actual_pack_files(pack_dir)
    extras = sorted(actual - expected)
    if not extras:
        return 0

    prefix = "ERROR" if args.strict else "WARNING"
    print(
        f"{prefix}: Pack contains extra files not covered by checksums.sha256:",
        file=sys.stderr,
    )
    for rel_path in extras:
        print(f"  - {rel_path}", file=sys.stderr)
    return 1 if args.strict else 0


def _control_file_mirror_errors(pack_dir: Path) -> list[str]:
    errors: list[str] = []
    for canonical_rel, mirror_rel in CONTROL_FILE_MIRRORS.items():
        mirror_path = pack_dir / mirror_rel
        if not mirror_path.is_file():
            continue
        canonical_path = pack_dir / canonical_rel
        if not canonical_path.is_file():
            errors.append(
                f"{mirror_rel} exists but canonical {canonical_rel} is missing."
            )
            continue
        if mirror_path.read_bytes() != canonical_path.read_bytes():
            errors.append(
                f"{mirror_rel} must match canonical {canonical_rel} byte-for-byte."
            )
    return errors


def cmd_json_object(args: argparse.Namespace) -> int:
    payload = _load_json(args.path)
    if not isinstance(payload, dict):
        print(
            f"ERROR: {args.label} must be a JSON object: {args.path}",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_scenarios_manifest(args: argparse.Namespace) -> int:
    payload = _load_json(args.path)
    if not isinstance(payload, dict):
        print(
            f"ERROR: scenarios manifest must be a JSON object: {args.path}",
            file=sys.stderr,
        )
        return 1
    if payload.get("schema") != "evidence_pack_scenarios_v1":
        print(
            f"ERROR: scenarios manifest schema must be evidence_pack_scenarios_v1: {args.path}",
            file=sys.stderr,
        )
        return 1
    try:
        version = int(payload.get("schema_version", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        version = 0
    if version != 1:
        print(
            f"ERROR: scenarios manifest schema_version must be 1: {args.path}",
            file=sys.stderr,
        )
        return 1
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        print(
            f"ERROR: scenarios manifest must include a non-empty scenarios list: {args.path}",
            file=sys.stderr,
        )
        return 1
    return 0


def _run_manifest_check(path: Path, check: Callable[[Path], list[str]]) -> int:
    errors = check(path)
    if errors:
        for error in errors:
            print(error)
        return 1
    return 0


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from invarlock.evidence_pack import validate_manifest

    return _run_manifest_check(args.manifest, validate_manifest)


def cmd_verify_manifest_provenance(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from invarlock.evidence_pack import verify_manifest_provenance

    return _run_manifest_check(args.pack_dir, verify_manifest_provenance)


def cmd_verify_signature(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from invarlock.evidence_pack_integrity import (
        normalize_expected_fingerprint,
        verify_signature,
    )

    expected_fingerprints = None
    if args.expected_fingerprint:
        normalized = normalize_expected_fingerprint(args.expected_fingerprint)
        if normalized is None:
            print(
                "--expected-fingerprint must be a sha256:... signing key fingerprint",
                file=sys.stderr,
            )
            return 2
        expected_fingerprints = frozenset({normalized})
    errors, warnings, signer_fingerprint = verify_signature(
        args.pack_dir,
        strict=args.strict,
        expected_fingerprints=expected_fingerprints,
    )
    for warning in warnings:
        print(warning, file=sys.stderr)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    if signer_fingerprint:
        print(signer_fingerprint)
    return 0


def _report_scenario_id(pack_dir: Path, report: Path) -> str | None:
    try:
        rel = report.resolve().relative_to((pack_dir / "reports").resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 3:
        return None
    if parts[1] == "errors" and len(parts) >= 4:
        scenario = parts[2].strip()
        return scenario or None
    scenario = parts[1].strip()
    return scenario or None


def _scenario_strictness(scenarios_path: Path, scenario_id: str) -> str | None:
    payload = _load_json(scenarios_path)
    if not isinstance(payload, dict):
        return None
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        return None
    for scenario in scenarios:
        if not isinstance(scenario, dict) or scenario.get("id") != scenario_id:
            continue
        strictness = scenario.get("strictness")
        if isinstance(strictness, str) and strictness:
            return strictness
    return None


def _report_expects_verify_failure(pack_dir: Path, report: Path) -> bool:
    scenario_id = _report_scenario_id(pack_dir, report)
    is_error = _is_error_injection_report(report)
    scenarios_path = pack_dir / "metadata" / "scenarios.json"
    if scenario_id is not None and scenarios_path.is_file():
        strictness = _scenario_strictness(scenarios_path, scenario_id)
        return strictness == "must_fail"

    # Legacy packs did not always carry scenario metadata. Preserve the old
    # hard-fault behavior for unclassified reports under errors/.
    return is_error


def _is_error_injection_report(report: Path) -> bool:
    return "errors" in report.parts and report.name == "evaluation.report.json"


def _report_counts_as_clean_pass(pack_dir: Path, report: Path) -> bool:
    return not _is_error_injection_report(
        report
    ) and not _report_expects_verify_failure(pack_dir, report)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _summary_evaluate_assurance() -> str | None:
    return os.environ.get("PACK_EVALUATE_ASSURANCE_USED") or os.environ.get(
        "PACK_EVALUATE_ASSURANCE"
    )


def _summary_release_review() -> bool | None:
    raw = os.environ.get("PACK_RELEASE_REVIEW_USED") or os.environ.get(
        "PACK_RELEASE_REVIEW"
    )
    if raw is None:
        return None
    return raw == "1"


def _expected_failure_signal(report: Path) -> bool:
    try:
        payload = _load_json(report)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False

    validation = payload.get("validation")
    if isinstance(validation, dict):
        for flag in (
            "primary_metric_acceptable",
            "preview_final_drift_acceptable",
            "invariants_pass",
            "spectral_stable",
            "rmt_stable",
            "guard_overhead_acceptable",
        ):
            if flag in validation and _truthy(validation.get(flag)) is False:
                return True

    primary_metric = payload.get("primary_metric")
    if isinstance(primary_metric, dict) and (
        _truthy(primary_metric.get("invalid"))
        or _truthy(primary_metric.get("degraded"))
    ):
        return True

    invariants = payload.get("invariants")
    if isinstance(invariants, dict):
        status = str(invariants.get("status") or "").strip().lower()
        if status in {"fail", "error", "warn"}:
            return True

    spectral = payload.get("spectral")
    if isinstance(spectral, dict):
        try:
            if int(spectral.get("caps_applied") or 0) > 0:
                return True
        except (TypeError, ValueError, OverflowError):
            pass
        summary = spectral.get("summary")
        if isinstance(summary, dict):
            status = str(summary.get("status") or "").strip().lower()
            if status in {"capped", "fail", "failed", "warn"}:
                return True

    return False


def _report_paths(pack_dir: Path) -> list[Path]:
    reports_root = pack_dir / "reports"
    if not reports_root.is_dir():
        return []
    return sorted(reports_root.rglob("evaluation.report.json"))


def _verify_command(
    reports: list[Path],
    *,
    profile: str,
    report_assurance: str,
    stdout_path: Path | None = None,
    stdout_to_null: bool = False,
) -> int:
    cmd = [
        "invarlock",
        "verify",
        "--json",
        "--profile",
        profile,
        "--assurance",
        report_assurance,
        *[str(path) for path in reports],
    ]
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as handle:
            return subprocess.run(cmd, check=False, stdout=handle).returncode
    if stdout_to_null:
        return subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL).returncode
    return subprocess.run(cmd, check=False).returncode


def cmd_report_scenario_id(args: argparse.Namespace) -> int:
    scenario_id = _report_scenario_id(args.pack_dir, args.report)
    if scenario_id is None:
        return 1
    print(scenario_id)
    return 0


def cmd_report_expects_verify_failure(args: argparse.Namespace) -> int:
    return 0 if _report_expects_verify_failure(args.pack_dir, args.report) else 1


def _verify_reports_with_sidecars(
    reports: list[Path],
    *,
    pack_dir: Path,
    profile: str,
    report_assurance: str,
    summary_out: Path | None,
) -> int:
    count_clean = 0
    count_error = 0
    count_expected_failure = 0
    count_failed = 0

    for report in reports:
        verify_out = report.parent / "verify.json"
        if _report_expects_verify_failure(pack_dir, report):
            rc = _verify_command(
                [report],
                profile=profile,
                report_assurance=report_assurance,
                stdout_path=verify_out,
            )
            if rc == 0:
                print(
                    f"ERROR: Expected verify failure verified as passing: {report}",
                    file=sys.stderr,
                )
                count_failed += 1
            elif _is_error_injection_report(report):
                count_error += 1
            else:
                count_expected_failure += 1
            continue

        rc = _verify_command(
            [report],
            profile=profile,
            report_assurance=report_assurance,
            stdout_path=verify_out,
        )
        if rc == 0:
            if _is_error_injection_report(report):
                count_error += 1
            else:
                count_clean += 1
        else:
            print(f"ERROR: Unexpected verify failure: {report}", file=sys.stderr)
            count_failed += 1

    total = count_clean + count_error + count_expected_failure + count_failed
    if total == 0:
        print("ERROR: No reports found to verify.", file=sys.stderr)
        return 1

    if summary_out is not None:
        write_verification_summary(
            summary_out,
            summary=WorkflowVerificationSummary(
                clean_reports=count_clean,
                error_injection_reports=count_error,
                expected_failure_reports=count_expected_failure,
                failed_reports=count_failed,
                policy_profile=profile,
                report_assurance=report_assurance,
                evaluate_assurance=_summary_evaluate_assurance(),
                release_review=_summary_release_review(),
            ),
        )

    print(
        "Verified: "
        f"{count_clean} expected-pass, "
        f"{count_error} error/probe reports, "
        f"{count_expected_failure} scenario expected-fail, "
        f"{count_failed} unexpected failures; "
        f"report assurance={report_assurance}"
    )
    return 1 if count_failed else 0


def _verify_reports_aggregate(
    reports: list[Path],
    *,
    pack_dir: Path,
    profile: str,
    report_assurance: str,
    json_out: Path | None,
    require_clean: bool,
) -> int:
    expected_pass_reports = [
        report
        for report in reports
        if not _report_expects_verify_failure(pack_dir, report)
    ]
    clean_reports = [
        report
        for report in expected_pass_reports
        if _report_counts_as_clean_pass(pack_dir, report)
    ]
    expected_failure_reports = [
        report for report in reports if _report_expects_verify_failure(pack_dir, report)
    ]
    if not reports:
        print("ERROR: No reports found in pack.", file=sys.stderr)
        return 1
    if require_clean and not clean_reports:
        print(
            "ERROR: No reports expected to pass in pack "
            "(only expected-failure reports present).",
            file=sys.stderr,
        )
        return 1

    if expected_pass_reports:
        rc = _verify_command(
            expected_pass_reports,
            profile=profile,
            report_assurance=report_assurance,
            stdout_path=json_out,
        )
        if rc != 0:
            return 1

    for report in expected_failure_reports:
        rc = _verify_command(
            [report],
            profile=profile,
            report_assurance=report_assurance,
            stdout_to_null=True,
        )
        if rc == 0:
            print(
                f"ERROR: Expected verify failure verified as passing: {report}",
                file=sys.stderr,
            )
            return 1
    return 0


def cmd_verify_reports(args: argparse.Namespace) -> int:
    reports = _report_paths(args.pack_dir)
    if args.write_sidecars:
        return _verify_reports_with_sidecars(
            reports,
            pack_dir=args.pack_dir,
            profile=args.profile,
            report_assurance=args.report_assurance,
            summary_out=args.summary_out,
        )
    return _verify_reports_aggregate(
        reports,
        pack_dir=args.pack_dir,
        profile=args.profile,
        report_assurance=args.report_assurance,
        json_out=args.json_out,
        require_clean=args.require_clean,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structured JSON/path checks for evidence-pack shell entrypoints."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_field = subparsers.add_parser("manifest-field")
    manifest_field.add_argument("manifest", type=Path)
    manifest_field.add_argument("field")
    manifest_field.set_defaults(func=cmd_manifest_field)

    path_within = subparsers.add_parser("path-within")
    path_within.add_argument("root", type=Path)
    path_within.add_argument("candidate", type=Path)
    path_within.set_defaults(func=cmd_path_within)

    scenario_strictness = subparsers.add_parser("scenario-strictness")
    scenario_strictness.add_argument("scenarios", type=Path)
    scenario_strictness.add_argument("scenario_id")
    scenario_strictness.set_defaults(func=cmd_scenario_strictness)

    report_scenario_id = subparsers.add_parser("report-scenario-id")
    report_scenario_id.add_argument("pack_dir", type=Path)
    report_scenario_id.add_argument("report", type=Path)
    report_scenario_id.set_defaults(func=cmd_report_scenario_id)

    report_expects_verify_failure = subparsers.add_parser(
        "report-expects-verify-failure"
    )
    report_expects_verify_failure.add_argument("pack_dir", type=Path)
    report_expects_verify_failure.add_argument("report", type=Path)
    report_expects_verify_failure.set_defaults(func=cmd_report_expects_verify_failure)

    verify_reports = subparsers.add_parser("verify-reports")
    verify_reports.add_argument("pack_dir", type=Path)
    verify_reports.add_argument("--json-out", type=Path)
    verify_reports.add_argument("--profile", default="dev")
    verify_reports.add_argument(
        "--report-assurance",
        default="report",
        choices=("report", "strict", "off"),
    )
    verify_reports.add_argument("--require-clean", action="store_true")
    verify_reports.add_argument("--write-sidecars", action="store_true")
    verify_reports.add_argument("--summary-out", type=Path)
    verify_reports.set_defaults(func=cmd_verify_reports)

    json_object = subparsers.add_parser("json-object")
    json_object.add_argument("path", type=Path)
    json_object.add_argument("--label", default="metadata file")
    json_object.set_defaults(func=cmd_json_object)

    scenarios_manifest = subparsers.add_parser("scenarios-manifest")
    scenarios_manifest.add_argument("path", type=Path)
    scenarios_manifest.set_defaults(func=cmd_scenarios_manifest)

    extra_files = subparsers.add_parser("extra-files")
    extra_files.add_argument("pack_dir", type=Path)
    extra_files.add_argument("--strict", action="store_true")
    extra_files.set_defaults(func=cmd_extra_files)

    validate_manifest = subparsers.add_parser("validate-manifest")
    validate_manifest.add_argument("manifest", type=Path)
    validate_manifest.set_defaults(func=cmd_validate_manifest)

    manifest_provenance = subparsers.add_parser("manifest-provenance")
    manifest_provenance.add_argument("pack_dir", type=Path)
    manifest_provenance.set_defaults(func=cmd_verify_manifest_provenance)

    signature = subparsers.add_parser(
        "signature", help="Verify a package-native evidence-pack signature bundle."
    )
    signature.add_argument("pack_dir", type=Path)
    signature.add_argument(
        "--strict",
        action="store_true",
        help="Fail closed when manifest.signature.json is missing.",
    )
    signature.add_argument(
        "--expected-fingerprint",
        help="Require the signer to match this sha256:... key fingerprint.",
    )
    signature.set_defaults(func=cmd_verify_signature)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
