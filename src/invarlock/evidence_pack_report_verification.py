"""Canonical-report verification for evidence packs.

All reports share scenario classification and failure-signal validation. The
injected runner keeps the boundary testable without hiding subprocess failures.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from invarlock import evidence_pack_integrity as evidence_pack_integrity_mod
from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome

_load_json = evidence_pack_integrity_mod._load_json
_json_load_error_types = evidence_pack_integrity_mod._json_load_error_types
RunVerifyCommand = Callable[..., VerifyExecutionResult]


def _scenario_strictness_by_id(pack_dir: Path) -> dict[str, str]:
    return {
        scenario_id: str(scenario.get("strictness"))
        for scenario_id, scenario in _scenario_by_id(pack_dir).items()
        if isinstance(scenario.get("strictness"), str)
    }


def _load_scenario_by_id(
    pack_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Load the scenario classifier without accepting ambiguous JSON."""

    scenarios_path = pack_dir / "metadata" / "scenarios.json"
    if not scenarios_path.is_file():
        return {}, []
    payload, errors = evidence_pack_integrity_mod._load_json_object(
        scenarios_path, label="scenario manifest"
    )
    if errors:
        return {}, errors
    assert payload is not None
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        return {}, ["scenario manifest scenarios must be a list."]
    scenario_by_id: dict[str, dict[str, Any]] = {}
    errors = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            continue
        scenario_id = scenario.get("id")
        if isinstance(scenario_id, str) and scenario_id.strip():
            if scenario_id in scenario_by_id:
                errors.append(
                    "scenario manifest contains duplicate scenario id: "
                    f"{scenario_id!r} at index {index}."
                )
                continue
            scenario_by_id[scenario_id] = scenario
    return scenario_by_id, errors


def _scenario_by_id(pack_dir: Path) -> dict[str, dict[str, Any]]:
    """Compatibility helper for callers that only need a best-effort index."""

    scenarios, errors = _load_scenario_by_id(pack_dir)
    return {} if errors else scenarios


def _report_scenario_id(pack_dir: Path, report: Path) -> str | None:
    try:
        parts = report.relative_to(pack_dir / "reports").parts
    except ValueError:
        return None
    if len(parts) == 2:
        scenario = parts[0].strip()
        return scenario or None
    if len(parts) < 3:
        return None
    if parts[1] == "errors" and len(parts) >= 4:
        scenario = parts[2].strip()
        return scenario or None
    return parts[1]


def _is_error_injection_report(report: Path) -> bool:
    return "errors" in report.parts and report.name == "evaluation.report.json"


def _report_expects_verify_failure(
    pack_dir: Path,
    report: Path,
    *,
    strictness_by_id: dict[str, str],
) -> bool:
    scenario_id = _report_scenario_id(pack_dir, report)
    if scenario_id and scenario_id in strictness_by_id:
        strictness = strictness_by_id[scenario_id]
        return strictness == "must_fail"
    return False


def _verify_command_succeeded(result: VerifyExecutionResult) -> bool:
    return result.outcome == VerifyOutcome.OK


def _detector_matches_report(
    report_payload: dict[str, Any], detector: dict[str, Any]
) -> bool:
    kind = str(detector.get("kind") or "").strip().lower()
    validation = report_payload.get("validation")
    primary_metric = report_payload.get("primary_metric")

    if kind == "validation_flag" and isinstance(validation, dict):
        flag = detector.get("flag")
        if not isinstance(flag, str) or flag not in validation:
            return False
        expected = detector.get("expected")
        actual = validation.get(flag)
        if isinstance(expected, bool):
            return actual is expected
        return actual == expected

    if kind == "primary_metric" and isinstance(primary_metric, dict):
        field = detector.get("field")
        if not isinstance(field, str) or field not in primary_metric:
            return False
        expected = detector.get("expected")
        actual = primary_metric.get(field)
        if isinstance(expected, bool):
            return actual is expected
        return actual == expected

    if kind == "spectral_caps_applied":
        spectral = report_payload.get("spectral")
        if not isinstance(spectral, dict):
            return False
        try:
            actual = int(spectral.get("caps_applied") or 0)
            minimum = int(detector.get("min") or 1)
        except (TypeError, ValueError, OverflowError):
            return False
        return actual >= minimum

    if kind == "invariants_status":
        invariants = report_payload.get("invariants")
        allowed = detector.get("allowed")
        if not isinstance(invariants, dict) or not isinstance(allowed, list):
            return False
        status = str(invariants.get("status") or "").strip().lower()
        return status in {
            str(value).strip().lower() for value in allowed if isinstance(value, str)
        }

    return False


def _primary_guard_failure_signal(
    report_payload: dict[str, Any], primary_guard: str
) -> bool:
    validation = report_payload.get("validation")
    validation = validation if isinstance(validation, dict) else {}
    guard = primary_guard.strip().lower()

    validation_key_by_guard = {
        "primary_metric": "primary_metric_acceptable",
        "invariants": "invariants_pass",
        "spectral": "spectral_stable",
        "rmt": "rmt_stable",
        "guard_metric_impact": "guard_metric_impact_acceptable",
    }
    validation_key = validation_key_by_guard.get(guard)
    if validation_key is not None and validation.get(validation_key) is False:
        return True

    if guard == "primary_metric":
        primary_metric = report_payload.get("primary_metric")
        return isinstance(primary_metric, dict) and (
            primary_metric.get("invalid") is True
            or primary_metric.get("degraded") is True
        )

    block = report_payload.get(guard)
    if not isinstance(block, dict):
        return False
    status = str(block.get("status") or "").strip().lower()
    if status in {"capped", "fail", "failed", "error", "warn"}:
        return True
    if guard == "spectral":
        try:
            return int(block.get("caps_applied") or 0) > 0
        except (TypeError, ValueError, OverflowError):
            return False
    return False


def _report_has_intended_failure_signal(pack_dir: Path, report: Path) -> bool:
    try:
        report_payload = _load_json(report)
    except _json_load_error_types():
        return False
    if not isinstance(report_payload, dict):
        return False

    scenario_id = _report_scenario_id(pack_dir, report)
    scenario = _scenario_by_id(pack_dir).get(scenario_id or "")
    if isinstance(scenario, dict) and scenario.get("strictness") == "must_fail":
        requirements = scenario.get("requirements")
        detectors = (
            requirements.get("detectors_any_of")
            if isinstance(requirements, dict)
            else None
        )
        if isinstance(detectors, list) and detectors:
            return any(
                _detector_matches_report(report_payload, detector)
                for detector in detectors
                if isinstance(detector, dict)
            )
        primary_guard = scenario.get("primary_guard")
        return isinstance(primary_guard, str) and _primary_guard_failure_signal(
            report_payload, primary_guard
        )

    return False


def _runtime_provenance_from_verify_payload(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        return None
    result = results[0]
    verification = result.get("verification") if isinstance(result, dict) else None
    runtime = (
        verification.get("runtime_provenance")
        if isinstance(verification, dict)
        else None
    )
    return runtime if isinstance(runtime, dict) else None


def _expected_failure_result_errors(
    pack_dir: Path,
    report: Path,
    result: VerifyExecutionResult,
    *,
    expected_runtime_image_digest: str | None,
) -> list[str]:
    rel_report = str(report.relative_to(pack_dir)).replace("\\", "/")
    if result.outcome != VerifyOutcome.POLICY_FAIL:
        return [
            "expected-failure report must produce POLICY_FAIL, not "
            f"{result.outcome.value}: {rel_report}"
        ]
    if not isinstance(result.payload, dict):
        return [
            f"expected-failure report verification payload is malformed: {rel_report}"
        ]

    runtime = _runtime_provenance_from_verify_payload(result.payload)
    if runtime is None or runtime.get("binding_verified") is not True:
        return [
            f"expected-failure report lacks valid report/runtime binding: {rel_report}"
        ]

    if expected_runtime_image_digest is not None:
        expected_digest_matched = runtime.get("expected_digest_matched") is True
        expected_image_digest_matched = (
            runtime.get("trust_status") == "expected_image_digest_matched"
        )
        if not expected_digest_matched or not expected_image_digest_matched:
            return [
                "expected-failure report did not match the expected runtime image "
                f"digest: {rel_report}"
            ]

    if not _report_has_intended_failure_signal(pack_dir, report):
        return [
            "expected-failure report lacks its intended report-local failure "
            f"signal: {rel_report}"
        ]
    return []


def verify_reports(
    pack_dir: Path,
    *,
    json_out_path: Path | None,
    profile: str,
    report_assurance: str,
    run_verify_command: RunVerifyCommand,
    expected_runtime_image_digest: str | None = None,
    baseline_by_report: dict[Path, Path] | None = None,
    policy_pack: Path | None = None,
) -> tuple[list[str], dict[str, Any] | None]:
    """Verify all canonical reports while honoring expected-failure scenarios."""
    reports = sorted(pack_dir.glob("reports/**/evaluation.report.json"))
    if not reports:
        return ["No reports found in pack."], None
    scenario_by_id, scenario_errors = _load_scenario_by_id(pack_dir)
    if scenario_errors:
        return scenario_errors, None
    strictness_by_id = {
        scenario_id: str(scenario.get("strictness"))
        for scenario_id, scenario in scenario_by_id.items()
        if isinstance(scenario.get("strictness"), str)
    }
    for report in reports:
        _report_payload, report_errors = evidence_pack_integrity_mod._load_json_object(
            report, label="canonical report"
        )
        if report_errors:
            return report_errors, None
    unclassified_reports = [
        path
        for path in reports
        if (_report_scenario_id(pack_dir, path) or "") not in strictness_by_id
    ]
    if unclassified_reports:
        rendered = ", ".join(
            str(path.relative_to(pack_dir)) for path in unclassified_reports
        )
        return [
            "Every report must reference a scenario declared by the current "
            f"scenario manifest; unclassified reports: {rendered}"
        ], None
    expected_failure_reports = [
        path
        for path in reports
        if _report_expects_verify_failure(
            pack_dir, path, strictness_by_id=strictness_by_id
        )
    ]
    expected_pass_reports = [
        path for path in reports if path not in expected_failure_reports
    ]
    if not expected_pass_reports:
        return [
            "No reports expected to pass in pack (only expected-failure reports present)."
        ], None

    common_verify_kwargs: dict[str, Any] = {
        "profile": profile,
        "report_assurance": report_assurance,
    }
    if expected_runtime_image_digest is not None:
        common_verify_kwargs["expected_runtime_image_digest"] = (
            expected_runtime_image_digest
        )
    if policy_pack is not None:
        common_verify_kwargs["policy_pack"] = policy_pack

    grouped_pass_reports: dict[Path | None, list[Path]] = {}
    for report in expected_pass_reports:
        baseline = (
            (baseline_by_report or {}).get(report.resolve())
            if baseline_by_report
            else None
        )
        grouped_pass_reports.setdefault(baseline, []).append(report)

    expected_pass_results: list[VerifyExecutionResult] = []
    for baseline, grouped_reports in grouped_pass_reports.items():
        verify_kwargs = dict(common_verify_kwargs)
        if baseline is not None:
            verify_kwargs["baseline"] = baseline
        result = run_verify_command(grouped_reports, **verify_kwargs)
        if not isinstance(result.payload, dict):
            return [
                "expected-pass report verification did not return a JSON object."
            ], None
        expected_pass_results.append(result)

    first_payload = expected_pass_results[0].payload
    assert isinstance(first_payload, dict)
    verify_payload = dict(first_payload)
    if len(expected_pass_results) > 1:
        combined_results: list[Any] = []
        for result in expected_pass_results:
            result_payload = result.payload
            assert isinstance(result_payload, dict)
            result_items = result_payload.get("results")
            if isinstance(result_items, list):
                combined_results.extend(result_items)
        verify_payload["results"] = combined_results
        verify_payload["evaluation_report"] = {"count": len(expected_pass_reports)}
        all_ok = all(_verify_command_succeeded(item) for item in expected_pass_results)
        verify_payload["summary"] = {
            "ok": all_ok,
            "reason": "ok" if all_ok else "policy_fail",
        }
    expected_failure_payloads: list[dict[str, Any]] = []
    for report in expected_failure_reports:
        verify_kwargs = dict(common_verify_kwargs)
        baseline = (baseline_by_report or {}).get(report.resolve())
        if baseline is not None:
            verify_kwargs["baseline"] = baseline
        try:
            expected_failure_result = run_verify_command([report], **verify_kwargs)
        except (
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return [
                f"expected-failure report verification failed unexpectedly: {exc}"
            ], verify_payload
        if not isinstance(expected_failure_result.payload, dict):
            return [
                "expected-failure report verification did not return a JSON object."
            ], verify_payload
        if _verify_command_succeeded(expected_failure_result):
            rel_report = str(report.relative_to(pack_dir)).replace("\\", "/")
            return [
                f"expected-failure report verified as passing: {rel_report}"
            ], verify_payload
        expected_failure_errors = _expected_failure_result_errors(
            pack_dir,
            report,
            expected_failure_result,
            expected_runtime_image_digest=expected_runtime_image_digest,
        )
        if expected_failure_errors:
            return expected_failure_errors, verify_payload
        expected_failure_payloads.append(expected_failure_result.payload)
    if expected_failure_reports:
        verify_payload["expected_failures"] = {
            "verify": expected_failure_payloads,
            "reports": [
                str(path.relative_to(pack_dir)).replace("\\", "/")
                for path in expected_failure_reports
            ],
        }
    if json_out_path is not None:
        json_out_path.write_text(
            json.dumps(verify_payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if not all(_verify_command_succeeded(result) for result in expected_pass_results):
        return [
            "invarlock verify reported report verification failures."
        ], verify_payload
    return [], verify_payload
