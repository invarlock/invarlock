"""Shared, evaluator-neutral support for the qualification example runners."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    return parser.parse_args()


def load_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    profile = load_json(args.profile)
    schedule = load_json(args.schedule)
    case_document = load_json(args.cases)
    cases = case_document.get("records")
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("cases.records must be an array of objects")
    expected = schedule.get("records")
    if not isinstance(expected, list) or len(expected) != len(cases):
        raise ValueError("schedule and cases must contain the same record count")
    normalized: list[dict[str, str]] = []
    for position, (case, scheduled) in enumerate(zip(cases, expected, strict=True)):
        required = ("input", "output", "record_id", "reference")
        if any(not isinstance(case.get(field), str) for field in required):
            raise ValueError(f"case {position} has an invalid field")
        normalized_case = {field: case[field] for field in required}
        if (
            scheduled.get("record_id") != normalized_case["record_id"]
            or scheduled.get("input_sha256") != sha256_text(normalized_case["input"])
            or scheduled.get("reference_output_sha256")
            != sha256_text(normalized_case["reference"])
        ):
            raise ValueError(f"case {position} does not match the independent schedule")
        normalized.append(normalized_case)
    return profile, schedule, normalized


def installed_inventory() -> list[dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower().replace("_", "-")] = distribution.version
    return [
        {"name": name, "version": version} for name, version in sorted(packages.items())
    ]


def require_profile_package(profile: dict[str, Any]) -> dict[str, str]:
    package = profile["upstream"]["package"]
    if package["ecosystem"] == "pypi":
        observed = importlib.metadata.version(package["name"])
        if observed != package["version"]:
            raise ValueError(
                f"installed {package['name']} version {observed} does not match "
                f"profile version {package['version']}"
            )
    return package


def _write_raw_and_export(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    raw: dict[str, Any],
    records: list[dict[str, Any]],
    summary: dict[str, str] | None,
) -> None:
    raw_bytes = canonical_bytes(raw)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_bytes(raw_bytes)
    export = {
        "bindings": {
            "dependency_lock_sha256": profile["execution"]["dependency_lock_sha256"],
            "profile_sha256": sha256_bytes(args.profile.read_bytes()),
            "raw_output_sha256": sha256_bytes(raw_bytes),
            "runner_sha256": profile["execution"]["runner_sha256"],
            "schedule_sha256": sha256_bytes(args.schedule.read_bytes()),
        },
        "format": "invarlock/evaluator-qualification-export-v1",
        "profile_id": profile["profile_id"],
        "records": records,
        "summary": summary,
        "upstream": profile["upstream"]["package"],
    }
    args.export.write_bytes(canonical_bytes(export))


def finish_deterministic(
    *,
    args: argparse.Namespace,
    entrypoint: str,
    scores: list[float],
    details: list[dict[str, Any]],
    environment: list[dict[str, str]] | None = None,
) -> None:
    profile, _, cases = load_inputs(args)
    package = require_profile_package(profile)
    if len(scores) != len(cases) or len(details) != len(cases):
        raise ValueError("scores and details must cover every scheduled record")
    normalized_scores = [float(score) for score in scores]
    if any(score not in (0.0, 1.0) for score in normalized_scores):
        raise ValueError("exact-match runners must report only 0 or 1")
    raw_records = [
        {
            "detail": detail,
            "record_id": case["record_id"],
            "score": score,
        }
        for case, score, detail in zip(cases, normalized_scores, details, strict=True)
    ]
    raw = {
        "entrypoint": entrypoint,
        "environment": environment
        if environment is not None
        else installed_inventory(),
        "format": "invarlock/upstream-evaluator-execution-v1",
        "profile_id": profile["profile_id"],
        "records": raw_records,
        "upstream": package,
    }
    export_records = [
        {
            "input_sha256": sha256_text(case["input"]),
            "output_sha256": sha256_text(case["output"]),
            "output_text": case["output"],
            "record_id": case["record_id"],
            "reported_score": score,
            "status": "ok",
        }
        for case, score in zip(cases, normalized_scores, strict=True)
    ]
    _write_raw_and_export(
        args=args,
        profile=profile,
        raw=raw,
        records=export_records,
        summary=None,
    )


def finish_observation(
    *,
    args: argparse.Namespace,
    entrypoint: str,
    summary_kind: str,
    summary_data: dict[str, Any],
    environment: list[dict[str, str]] | None = None,
) -> None:
    profile, _, _ = load_inputs(args)
    package = require_profile_package(profile)
    raw = {
        "entrypoint": entrypoint,
        "environment": environment
        if environment is not None
        else installed_inventory(),
        "format": "invarlock/upstream-evaluator-execution-v1",
        "profile_id": profile["profile_id"],
        "summary": summary_data,
        "upstream": package,
    }
    raw_bytes = canonical_bytes(raw)
    _write_raw_and_export(
        args=args,
        profile=profile,
        raw=raw,
        records=[],
        summary={"kind": summary_kind, "sha256": sha256_bytes(raw_bytes)},
    )
