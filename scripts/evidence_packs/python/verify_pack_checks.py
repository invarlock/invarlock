from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from scripts.evidence_packs.python.verify_pack_cli import parse_args
except ImportError:  # pragma: no cover - direct script execution path
    from verify_pack_cli import parse_args

try:
    from invarlock.evidence_pack_json import (
        StrictJsonError,
        load_json_object,
        parse_json_bytes,
    )
except ImportError:  # pragma: no cover - direct script execution path
    _REPO_SRC = Path(__file__).resolve().parents[3] / "src"
    if str(_REPO_SRC) not in sys.path:
        sys.path.insert(0, str(_REPO_SRC))
    from invarlock.evidence_pack_json import (
        StrictJsonError,
        load_json_object,
        parse_json_bytes,
    )

try:
    from scripts.evidence_packs.python import (
        verify_pack_report_classification as _report_classification,
    )
except ImportError:  # pragma: no cover - direct script execution path
    import verify_pack_report_classification as _report_classification

try:
    from scripts.evidence_packs.python.artifact_io import (
        VerificationSummary,
        write_verification_summary,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from artifact_io import (
        VerificationSummary,
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


def _parse_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    """Apply the package's canonical strict JSON parser to an output stream."""

    payload = parse_json_bytes(raw, label=label)
    if not isinstance(payload, dict):
        raise StrictJsonError(f"{label} must be a JSON object")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    return load_json_object(
        path,
        label=f"JSON input {path}",
    )


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
    strictness = _report_classification.scenario_strictness(
        args.scenarios, args.scenario_id
    )
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
    try:
        _load_json(args.path)
    except StrictJsonError as exc:
        print(
            f"ERROR: {args.label} must be a valid JSON object: {args.path} ({exc})",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_scenarios_manifest(args: argparse.Namespace) -> int:
    try:
        payload = _load_json(args.path)
    except StrictJsonError as exc:
        print(
            f"ERROR: scenarios manifest must be a valid JSON object: {args.path} ({exc})",
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
    try:
        _load_json(args.manifest)
    except StrictJsonError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from invarlock.evidence_pack import validate_manifest

    return _run_manifest_check(args.manifest, validate_manifest)


def cmd_verify_manifest_provenance(args: argparse.Namespace) -> int:
    manifest_path = args.pack_dir / "manifest.json"
    try:
        _load_json(manifest_path)
    except StrictJsonError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from invarlock.evidence_pack import verify_manifest_provenance

    return _run_manifest_check(args.pack_dir, verify_manifest_provenance)


def cmd_final_verdict_binding(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from invarlock.evidence_pack import verify_final_verdict_report_binding

    errors = verify_final_verdict_report_binding(
        args.pack_dir,
        require_binding=args.require_binding,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


def _validated_baseline_mapping(
    pack_dir: Path,
    *,
    report_assurance: str,
) -> tuple[dict[Path, Path], list[str]]:
    if not (pack_dir / "manifest.json").is_file():
        if report_assurance == "strict":
            return {}, [
                "strict report verification requires a signed manifest baseline declaration"
            ]
        return {}, []
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from invarlock.evidence_pack_baselines import verify_baseline_materials

    result = verify_baseline_materials(
        pack_dir,
        report_assurance=report_assurance,
    )
    return result.baseline_by_report, list(result.errors)


def _staged_baseline_mapping(
    pack_dir: Path,
    *,
    report_assurance: str,
) -> tuple[dict[Path, Path], list[str]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from invarlock.evidence_pack_baselines import discover_staged_baseline_materials

    result = discover_staged_baseline_materials(
        pack_dir,
        report_assurance=report_assurance,
    )
    return result.baseline_by_report, list(result.errors)


def cmd_baseline_materials(args: argparse.Namespace) -> int:
    _mapping, errors = _validated_baseline_mapping(
        args.pack_dir,
        report_assurance=args.report_assurance,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


def cmd_verify_signature(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from invarlock.evidence_pack_integrity import (
        normalize_expected_fingerprint,
        verify_signature,
    )

    signature_path = args.pack_dir / "manifest.signature.json"
    if signature_path.exists() or signature_path.is_symlink():
        try:
            _load_json(args.pack_dir / "manifest.json")
            _load_json(signature_path)
        except StrictJsonError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

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


def _verify_command(
    reports: list[Path],
    *,
    profile: str,
    report_assurance: str,
    baseline: Path | None = None,
    policy_pack: Path | None = None,
    expected_runtime_image_digest: str | None = None,
    stdout_path: Path | None = None,
    stdout_to_null: bool = False,
) -> tuple[int, dict[str, Any] | None]:
    cmd = [
        "invarlock",
        "verify",
        "--json",
        "--profile",
        profile,
        "--assurance",
        report_assurance,
    ]
    if expected_runtime_image_digest is not None:
        cmd.extend(["--expected-runtime-image-digest", expected_runtime_image_digest])
    if baseline is not None:
        cmd.extend(["--baseline", str(baseline)])
    if policy_pack is not None:
        cmd.extend(["--policy-pack", str(policy_pack)])
    cmd.extend(str(path) for path in reports)
    completed = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
    )
    stdout_bytes = completed.stdout or b""
    try:
        stdout = stdout_bytes.decode("utf-8")
    except UnicodeDecodeError:
        if stdout_path is not None:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_bytes(stdout_bytes)
        print(
            "ERROR: invarlock verify did not emit UTF-8 JSON output.",
            file=sys.stderr,
        )
        return 1, None
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(stdout, encoding="utf-8")
    elif not stdout_to_null and stdout:
        print(stdout, end="")
    try:
        payload = _parse_json_object(stdout_bytes, label="invarlock verify output")
    except StrictJsonError:
        payload = None
    if payload is None and completed.returncode == 0:
        print(
            "ERROR: invarlock verify exited successfully without a strict JSON object.",
            file=sys.stderr,
        )
        return 1, None
    return completed.returncode, payload


def _write_portable_verify_payload(
    path: Path,
    payload: dict[str, Any] | None,
    *,
    pack_dir: Path,
) -> None:
    if payload is None:
        return
    results = payload.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            try:
                relative = Path(item["id"]).resolve().relative_to(pack_dir.resolve())
            except (OSError, ValueError):
                continue
            item["id"] = relative.as_posix()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _expected_failure_contract_errors(
    pack_dir: Path,
    report: Path,
    *,
    returncode: int,
    payload: dict[str, Any] | None,
    expected_runtime_image_digest: str | None,
) -> list[str]:
    if payload is None:
        return [
            f"expected-failure report verification did not emit a JSON object: {report}"
        ]
    summary = payload.get("summary")
    reason = summary.get("reason") if isinstance(summary, dict) else None
    if returncode == 0:
        outcome_name = "ok"
    elif reason == "policy_fail":
        outcome_name = "policy_fail"
    elif reason == "malformed" or returncode == 2:
        outcome_name = "malformed"
    else:
        return [
            "expected-failure report verification JSON lacks a recognized outcome: "
            f"{report}"
        ]

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from invarlock.evidence_pack import _expected_failure_result_errors
    from invarlock.reporting.verify_contract import (
        VerifyExecutionResult,
        VerifyOutcome,
    )

    result = VerifyExecutionResult(
        outcome=VerifyOutcome(outcome_name),
        payload=payload,
        diagnostics=(),
    )
    contract_errors = _expected_failure_result_errors(
        pack_dir,
        report,
        result,
        expected_runtime_image_digest=expected_runtime_image_digest,
    )
    if not isinstance(contract_errors, list):
        return [f"expected-failure contract returned invalid errors: {report}"]
    return [str(error) for error in contract_errors]


def cmd_report_scenario_id(args: argparse.Namespace) -> int:
    scenario_id = _report_classification.report_scenario_id(args.pack_dir, args.report)
    if scenario_id is None:
        return 1
    print(scenario_id)
    return 0


def cmd_report_expects_verify_failure(args: argparse.Namespace) -> int:
    return (
        0
        if _report_classification.report_expects_verify_failure(
            args.pack_dir, args.report
        )
        else 1
    )


def _verify_reports_with_sidecars(
    reports: list[Path],
    *,
    pack_dir: Path,
    profile: str,
    report_assurance: str,
    expected_runtime_image_digest: str | None,
    summary_out: Path | None,
    baseline_by_report: dict[Path, Path],
    policy_pack: Path | None,
) -> int:
    if (pack_dir / "metadata" / "scenarios.json").is_file():
        unclassified = _report_classification.unclassified_reports(pack_dir, reports)
        if unclassified:
            for report in unclassified:
                print(
                    "ERROR: Report is not classified by the current scenario "
                    f"manifest: {report}",
                    file=sys.stderr,
                )
            return 1

    count_clean = 0
    count_error = 0
    count_expected_failure = 0
    count_failed = 0

    for report in reports:
        verify_out = report.parent / "verify.json"
        baseline = baseline_by_report.get(report.resolve())
        if _report_classification.report_expects_verify_failure(pack_dir, report):
            rc, verify_payload = _verify_command(
                [report],
                profile=profile,
                report_assurance=report_assurance,
                baseline=baseline,
                policy_pack=policy_pack,
                expected_runtime_image_digest=expected_runtime_image_digest,
                stdout_path=verify_out,
            )
            _write_portable_verify_payload(
                verify_out, verify_payload, pack_dir=pack_dir
            )
            if rc == 0:
                print(
                    f"ERROR: Expected verify failure verified as passing: {report}",
                    file=sys.stderr,
                )
                count_failed += 1
            else:
                contract_errors = _expected_failure_contract_errors(
                    pack_dir,
                    report,
                    returncode=rc,
                    payload=verify_payload,
                    expected_runtime_image_digest=expected_runtime_image_digest,
                )
                if contract_errors:
                    for error in contract_errors:
                        print(f"ERROR: {error}", file=sys.stderr)
                    count_failed += 1
                elif _report_classification.is_error_injection_report(report):
                    count_error += 1
                else:
                    count_expected_failure += 1
            continue

        rc, verify_payload = _verify_command(
            [report],
            profile=profile,
            report_assurance=report_assurance,
            baseline=baseline,
            policy_pack=policy_pack,
            expected_runtime_image_digest=expected_runtime_image_digest,
            stdout_path=verify_out,
        )
        _write_portable_verify_payload(verify_out, verify_payload, pack_dir=pack_dir)
        if rc == 0:
            if _report_classification.is_error_injection_report(report):
                count_error += 1
            else:
                count_clean += 1
        elif _report_classification.report_is_informational(pack_dir, report):
            if _report_classification.is_error_injection_report(report):
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
            summary=VerificationSummary(
                clean_reports=count_clean,
                error_injection_reports=count_error,
                expected_failure_reports=count_expected_failure,
                failed_reports=count_failed,
                policy_profile=profile,
                report_assurance=report_assurance,
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
    expected_runtime_image_digest: str | None,
    json_out: Path | None,
    require_clean: bool,
    baseline_by_report: dict[Path, Path],
    policy_pack: Path | None,
) -> int:
    if (pack_dir / "metadata" / "scenarios.json").is_file():
        unclassified = _report_classification.unclassified_reports(pack_dir, reports)
        if unclassified:
            for report in unclassified:
                print(
                    "ERROR: Report is not classified by the current scenario "
                    f"manifest: {report}",
                    file=sys.stderr,
                )
            return 1

    expected_pass_reports = [
        report
        for report in reports
        if not _report_classification.report_expects_verify_failure(pack_dir, report)
    ]
    clean_reports = [
        report
        for report in expected_pass_reports
        if _report_classification.report_counts_as_clean_pass(pack_dir, report)
    ]
    expected_failure_reports = [
        report
        for report in reports
        if _report_classification.report_expects_verify_failure(pack_dir, report)
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
        groups: dict[Path | None, list[Path]] = {}
        for report in expected_pass_reports:
            baseline = baseline_by_report.get(report.resolve())
            groups.setdefault(baseline, []).append(report)
        group_payloads: list[dict[str, Any]] = []
        for baseline, grouped_reports in groups.items():
            rc, verify_payload = _verify_command(
                grouped_reports,
                profile=profile,
                report_assurance=report_assurance,
                baseline=baseline,
                policy_pack=policy_pack,
                expected_runtime_image_digest=expected_runtime_image_digest,
                stdout_to_null=len(groups) > 1,
                stdout_path=json_out if len(groups) == 1 else None,
            )
            if rc != 0:
                return 1
            if verify_payload is not None:
                group_payloads.append(verify_payload)
            if json_out is not None and len(groups) == 1:
                _write_portable_verify_payload(
                    json_out, verify_payload, pack_dir=pack_dir
                )
        if json_out is not None and len(groups) > 1:
            combined_results: list[Any] = []
            for payload in group_payloads:
                results = payload.get("results")
                if isinstance(results, list):
                    combined_results.extend(results)
            combined = dict(group_payloads[0]) if group_payloads else {}
            combined["summary"] = {"ok": True, "reason": "ok"}
            combined["evaluation_report"] = {"count": len(expected_pass_reports)}
            combined["results"] = combined_results
            json_out.parent.mkdir(parents=True, exist_ok=True)
            _write_portable_verify_payload(json_out, combined, pack_dir=pack_dir)

    for report in expected_failure_reports:
        rc, verify_payload = _verify_command(
            [report],
            profile=profile,
            report_assurance=report_assurance,
            baseline=baseline_by_report.get(report.resolve()),
            policy_pack=policy_pack,
            expected_runtime_image_digest=expected_runtime_image_digest,
            stdout_to_null=True,
        )
        if rc == 0:
            print(
                f"ERROR: Expected verify failure verified as passing: {report}",
                file=sys.stderr,
            )
            return 1
        contract_errors = _expected_failure_contract_errors(
            pack_dir,
            report,
            returncode=rc,
            payload=verify_payload,
            expected_runtime_image_digest=expected_runtime_image_digest,
        )
        if contract_errors:
            for error in contract_errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
    return 0


def cmd_verify_reports(args: argparse.Namespace) -> int:
    reports = _report_classification.report_paths(args.pack_dir)
    mapping_fn = (
        _staged_baseline_mapping
        if args.staged_baselines
        else _validated_baseline_mapping
    )
    baseline_by_report, baseline_errors = mapping_fn(
        args.pack_dir, report_assurance=args.report_assurance
    )
    if baseline_errors:
        for error in baseline_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.write_sidecars:
        return _verify_reports_with_sidecars(
            reports,
            pack_dir=args.pack_dir,
            profile=args.profile,
            report_assurance=args.report_assurance,
            expected_runtime_image_digest=args.expected_runtime_image_digest,
            summary_out=args.summary_out,
            baseline_by_report=baseline_by_report,
            policy_pack=args.policy_pack,
        )
    return _verify_reports_aggregate(
        reports,
        pack_dir=args.pack_dir,
        profile=args.profile,
        report_assurance=args.report_assurance,
        expected_runtime_image_digest=args.expected_runtime_image_digest,
        json_out=args.json_out,
        require_clean=args.require_clean,
        baseline_by_report=baseline_by_report,
        policy_pack=args.policy_pack,
    )


def cmd_policy_materials(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from invarlock.evidence_pack_policy import verify_policy_material

    result = verify_policy_material(
        args.pack_dir,
        report_assurance=args.report_assurance,
        acceptance_policy_path=args.policy_pack,
    )
    for error in result.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if result.errors else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(
        argv,
        handlers={
            "manifest-field": cmd_manifest_field,
            "path-within": cmd_path_within,
            "scenario-strictness": cmd_scenario_strictness,
            "report-scenario-id": cmd_report_scenario_id,
            "report-expects-verify-failure": cmd_report_expects_verify_failure,
            "verify-reports": cmd_verify_reports,
            "policy-materials": cmd_policy_materials,
            "json-object": cmd_json_object,
            "scenarios-manifest": cmd_scenarios_manifest,
            "extra-files": cmd_extra_files,
            "validate-manifest": cmd_validate_manifest,
            "manifest-provenance": cmd_verify_manifest_provenance,
            "final-verdict-binding": cmd_final_verdict_binding,
            "baseline-materials": cmd_baseline_materials,
            "signature": cmd_verify_signature,
        },
    )
    try:
        return int(args.func(args))
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
