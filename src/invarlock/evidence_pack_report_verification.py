from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from invarlock.reporting.verify_contract import VerifyExecutionResult, VerifyOutcome

RunVerifyCommand = Callable[..., VerifyExecutionResult]


def _scenario_strictness_by_id(pack_dir: Path) -> dict[str, str]:
    scenarios_path = pack_dir / "metadata" / "scenarios.json"
    if not scenarios_path.is_file():
        return {}
    try:
        payload = json.loads(scenarios_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        return {}
    strictness_by_id: dict[str, str] = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        scenario_id = scenario.get("id")
        strictness = scenario.get("strictness")
        if isinstance(scenario_id, str) and isinstance(strictness, str):
            strictness_by_id[scenario_id] = strictness
    return strictness_by_id


def _report_scenario_id(pack_dir: Path, report: Path) -> str | None:
    try:
        parts = report.relative_to(pack_dir / "reports").parts
    except ValueError:
        return None
    if len(parts) < 4:
        return None
    return parts[1]


def _report_expects_verify_failure(
    pack_dir: Path,
    report: Path,
    *,
    strictness_by_id: dict[str, str],
) -> bool:
    if "/errors/" in report.as_posix():
        return True
    scenario_id = _report_scenario_id(pack_dir, report)
    return bool(scenario_id and strictness_by_id.get(scenario_id) == "must_fail")


def _verify_command_succeeded(result: VerifyExecutionResult) -> bool:
    return result.outcome == VerifyOutcome.OK


def verify_reports(
    pack_dir: Path,
    *,
    json_out_path: Path | None,
    profile: str,
    report_assurance: str,
    run_verify_command: RunVerifyCommand,
) -> tuple[list[str], dict[str, Any] | None]:
    reports = sorted(pack_dir.glob("reports/**/evaluation.report.json"))
    if not reports:
        return ["No reports found in pack."], None
    strictness_by_id = _scenario_strictness_by_id(pack_dir)
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

    expected_pass_result = run_verify_command(
        expected_pass_reports,
        profile=profile,
        report_assurance=report_assurance,
    )
    if not isinstance(expected_pass_result.payload, dict):
        return ["expected-pass report verification did not return a JSON object."], None
    verify_payload = dict(expected_pass_result.payload)
    expected_failure_payloads: list[dict[str, Any]] = []
    for report in expected_failure_reports:
        try:
            expected_failure_result = run_verify_command(
                [report],
                profile=profile,
                report_assurance=report_assurance,
            )
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
            json.dumps(verify_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    if not _verify_command_succeeded(expected_pass_result):
        return [
            "invarlock verify reported report verification failures."
        ], verify_payload
    return [], verify_payload
