#!/usr/bin/env python3
"""Validate a non-synthetic guard-evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "invarlock/empirical-guard-evidence-v1"
REQUIRED_GUARDS = {"spectral", "rmt", "variance"}
ALLOWED_EVIDENCE_KINDS = {
    "calibration_null_sweep",
    "calibration_ve_sweep",
    "evidence_pack",
    "model_evidence_sweep",
    "checkpoint_probe",
}
REAL_PRODUCER_MARKERS = (
    "model-evidence-sweep",
    "model_evidence_sweep.py",
    "calibrate null-sweep",
    "calibrate ve-sweep",
    "run_pack.sh",
    "run_suite.sh",
    "run_model_evidence_remote.py",
    "run_qwen14_sentinels.sh",
)


def _load_json(path: Path, label: str, failures: list[str]) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(f"{label} missing: {path}")
    except json.JSONDecodeError as exc:
        failures.append(f"{label} is not valid JSON: {path}: {exc}")
    return None


def _resolve_artifact(
    root: Path, value: object, label: str, failures: list[str]
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        failures.append(f"{label} artifact must be a non-empty relative path.")
        return None
    raw = Path(value)
    if raw.is_absolute():
        failures.append(f"{label} artifact must be relative: {value}")
        return None
    root_resolved = root.resolve()
    candidate = (root_resolved / raw).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        failures.append(f"{label} artifact escapes evidence root: {value}")
        return None
    if not candidate.is_file():
        failures.append(f"{label} artifact missing: {candidate}")
        return None
    if candidate.stat().st_size == 0:
        failures.append(f"{label} artifact must not be empty: {candidate}")
    return candidate


def _validate_source_commands(payload: dict[str, object], failures: list[str]) -> None:
    commands = payload.get("source_commands")
    if not isinstance(commands, list) or not commands:
        failures.append("empirical evidence source_commands must be a non-empty list.")
        return
    valid_commands: list[str] = []
    for index, command in enumerate(commands):
        if not isinstance(command, str) or not command.strip():
            failures.append(
                f"empirical evidence source_commands[{index}] must be a string."
            )
            continue
        valid_commands.append(command)
    if valid_commands and not any(
        marker in command
        for command in valid_commands
        for marker in REAL_PRODUCER_MARKERS
    ):
        failures.append(
            "empirical evidence source_commands must include a real evidence producer."
        )


def _validate_guard_rows(
    root: Path, payload: dict[str, object], failures: list[str]
) -> None:
    rows = payload.get("guard_rows")
    if not isinstance(rows, list) or not rows:
        failures.append("empirical evidence guard_rows must be a non-empty list.")
        return
    observed: set[str] = set()
    for index, row in enumerate(rows):
        label = f"guard_rows[{index}]"
        if not isinstance(row, dict):
            failures.append(f"{label} must be an object.")
            continue
        guard = row.get("guard")
        if guard not in REQUIRED_GUARDS:
            failures.append(
                f"{label}.guard must be one of: {', '.join(sorted(REQUIRED_GUARDS))}."
            )
            continue
        kind = row.get("evidence_kind")
        if kind not in ALLOWED_EVIDENCE_KINDS:
            failures.append(
                f"{label}.evidence_kind must be one of: "
                + ", ".join(sorted(ALLOWED_EVIDENCE_KINDS))
                + "."
            )
        status = row.get("status")
        if status != "empirical":
            failures.append(f"{label}.status must be empirical.")
        scope = str(row.get("scope", "")).lower()
        if row.get("synthetic") is True or "synthetic" in scope:
            failures.append(f"{label} must not be synthetic evidence.")
        if not isinstance(row.get("model_family"), str) or not row.get("model_family"):
            failures.append(f"{label}.model_family must be a non-empty string.")
        _resolve_artifact(root, row.get("artifact"), label, failures)
        if (
            guard in REQUIRED_GUARDS
            and kind in ALLOWED_EVIDENCE_KINDS
            and status == "empirical"
            and row.get("synthetic") is not True
            and "synthetic" not in scope
        ):
            observed.add(str(guard))
    missing = sorted(REQUIRED_GUARDS - observed)
    if missing:
        failures.append("empirical evidence missing guard rows: " + ", ".join(missing))


def _validate_model_family_rows(
    root: Path, payload: dict[str, object], failures: list[str]
) -> None:
    rows = payload.get("model_family_rows")
    if not isinstance(rows, list) or not rows:
        failures.append(
            "empirical evidence model_family_rows must be a non-empty list."
        )
        return
    for index, row in enumerate(rows):
        label = f"model_family_rows[{index}]"
        if not isinstance(row, dict):
            failures.append(f"{label} must be an object.")
            continue
        if not isinstance(row.get("model_family"), str) or not row.get("model_family"):
            failures.append(f"{label}.model_family must be a non-empty string.")
        if row.get("status") not in {"observed", "empirical"}:
            failures.append(f"{label}.status must be observed or empirical.")
        _resolve_artifact(root, row.get("artifact"), label, failures)


def check_empirical_guard_evidence(*, root: Path) -> list[str]:
    failures: list[str] = []
    manifest_path = root / "manifest.json"
    payload = _load_json(manifest_path, "empirical guard evidence manifest", failures)
    if not isinstance(payload, dict):
        failures.append("empirical guard evidence manifest must be a JSON object.")
        return failures
    if payload.get("schema") != SCHEMA:
        failures.append(f"empirical guard evidence schema must be {SCHEMA}.")
    _validate_source_commands(payload, failures)
    _validate_guard_rows(root, payload, failures)
    _validate_model_family_rows(root, payload, failures)
    return failures


def _build_summary(*, root: Path, failures: list[str]) -> dict[str, object]:
    return {
        "schema": "invarlock/empirical-guard-evidence-check-v1",
        "root": str(root),
        "ok": not failures,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate non-synthetic empirical guard-evidence artifacts."
    )
    parser.add_argument(
        "--root",
        default="artifacts/guard-validation/empirical",
        help="Empirical guard-evidence bundle root.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable summary.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    failures = check_empirical_guard_evidence(root=root)
    summary = _build_summary(root=root, failures=failures)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
    else:
        print("Empirical guard evidence check passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
