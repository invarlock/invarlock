#!/usr/bin/env python3
"""Validate a non-synthetic guard-evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:  # pragma: no branch
    sys.path.insert(0, str(SCRIPT_DIR))

from evidence_contracts import (  # noqa: E402
    REQUIRED_GUARDS,
    EmpiricalGuardEvidenceManifest,
    GuardEvidenceRow,
    ModelFamilyEvidenceRow,
    load_json,
    resolve_artifact,
)


def _load_json(path: Path, label: str, failures: list[str]) -> object | None:
    return load_json(path, label, failures)


def _resolve_artifact(
    root: Path, value: object, label: str, failures: list[str]
) -> Path | None:
    return resolve_artifact(root, value, label, failures)


def _validate_source_commands(payload: dict[str, object], failures: list[str]) -> None:
    manifest = EmpiricalGuardEvidenceManifest(root=Path(), payload=payload)
    manifest._validate_source_commands(failures)


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
        guard = GuardEvidenceRow(index=index, payload=row).validate(
            root=root,
            failures=failures,
        )
        if guard is not None:
            observed.add(guard)
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
        ModelFamilyEvidenceRow(index=index, payload=row).validate(
            root=root,
            failures=failures,
        )


def check_empirical_guard_evidence(*, root: Path) -> list[str]:
    failures: list[str] = []
    manifest = EmpiricalGuardEvidenceManifest.load(root=root, failures=failures)
    if manifest.payload is None:
        failures.append("empirical guard evidence manifest must be a JSON object.")
        return failures
    failures.extend(manifest.validate())
    return failures


def _build_summary(*, root: Path, failures: list[str]) -> dict[str, object]:
    manifest = EmpiricalGuardEvidenceManifest(root=root, payload={})
    return manifest.summary(failures)


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
