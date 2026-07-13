"""Non-authoritative empirical guard-artifact inventory helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EMPIRICAL_GUARD_EVIDENCE_SCHEMA = "invarlock/empirical-guard-inventory-v1"
EMPIRICAL_GUARD_EVIDENCE_CHECK_SCHEMA = "invarlock/empirical-guard-inventory-check-v1"
EMPIRICAL_GUARD_EVIDENCE_AUTHORITY = "diagnostic_inventory"

REQUIRED_GUARDS = frozenset({"spectral", "rmt", "variance"})
ALLOWED_EVIDENCE_KINDS = frozenset(
    {
        "calibration_null_sweep",
        "calibration_ve_sweep",
        "catalog_evaluation",
        "evidence_pack",
        "checkpoint_probe",
    }
)
REAL_PRODUCER_MARKERS = (
    "invarlock evaluate",
    "evidence-pack seal",
    "calibrate null-sweep",
    "calibrate ve-sweep",
)
_MANIFEST_OBJECT_ERROR = "empirical guard evidence manifest must be a JSON object."


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def load_json(path: Path, label: str, failures: list[str]) -> object | None:
    try:
        payload: object = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
        return payload
    except FileNotFoundError:
        failures.append(f"{label} missing: {path}")
    except json.JSONDecodeError as exc:
        failures.append(f"{label} is not valid JSON: {path}: {exc}")
    except ValueError as exc:
        failures.append(f"{label} is ambiguous JSON: {path}: {exc}")
    return None


def resolve_artifact(
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
    unresolved_candidate = root_resolved / raw
    if unresolved_candidate.is_symlink():
        failures.append(f"{label} artifact must be a regular file, not a symlink.")
        return None
    candidate = unresolved_candidate.resolve()
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


@dataclass(frozen=True)
class GuardEvidenceRow:
    index: int
    payload: dict[str, Any]

    def validate(self, *, root: Path, failures: list[str]) -> str | None:
        label = f"guard_rows[{self.index}]"
        guard = self.payload.get("guard")
        if guard not in REQUIRED_GUARDS:
            failures.append(
                f"{label}.guard must be one of: {', '.join(sorted(REQUIRED_GUARDS))}."
            )
            return None
        kind = self.payload.get("evidence_kind")
        if kind not in ALLOWED_EVIDENCE_KINDS:
            failures.append(
                f"{label}.evidence_kind must be one of: "
                + ", ".join(sorted(ALLOWED_EVIDENCE_KINDS))
                + "."
            )
        status = self.payload.get("status")
        if status != "indexed":
            failures.append(f"{label}.status must be indexed.")
        scope = str(self.payload.get("scope", "")).lower()
        if self.payload.get("synthetic") is True or "synthetic" in scope:
            failures.append(f"{label} must not be synthetic evidence.")
        if not isinstance(
            self.payload.get("model_family"), str
        ) or not self.payload.get("model_family"):
            failures.append(f"{label}.model_family must be a non-empty string.")
        resolve_artifact(root, self.payload.get("artifact"), label, failures)
        if (
            guard in REQUIRED_GUARDS
            and kind in ALLOWED_EVIDENCE_KINDS
            and status == "indexed"
            and self.payload.get("synthetic") is not True
            and "synthetic" not in scope
        ):
            return str(guard)
        return None


@dataclass(frozen=True)
class ModelFamilyEvidenceRow:
    index: int
    payload: dict[str, Any]

    def validate(self, *, root: Path, failures: list[str]) -> None:
        label = f"model_family_rows[{self.index}]"
        if not isinstance(
            self.payload.get("model_family"), str
        ) or not self.payload.get("model_family"):
            failures.append(f"{label}.model_family must be a non-empty string.")
        if self.payload.get("status") != "indexed":
            failures.append(f"{label}.status must be indexed.")
        resolve_artifact(root, self.payload.get("artifact"), label, failures)


@dataclass(frozen=True)
class EmpiricalGuardEvidenceManifest:
    root: Path
    payload: dict[str, Any] | None

    @classmethod
    def load(cls, *, root: Path, failures: list[str]) -> EmpiricalGuardEvidenceManifest:
        payload = load_json(
            root / "manifest.json", "empirical guard evidence manifest", failures
        )
        if payload is not None and not isinstance(payload, dict):
            failures.append("empirical guard evidence manifest must be a JSON object.")
            payload = None
        return cls(root, payload)

    def validate(self) -> list[str]:
        failures: list[str] = []
        if self.payload is None:
            load_failures: list[str] = []
            load_json(
                self.root / "manifest.json",
                "empirical guard evidence manifest",
                load_failures,
            )
            return load_failures + [
                "empirical guard evidence manifest must be a JSON object."
            ]
        if self.payload.get("schema") != EMPIRICAL_GUARD_EVIDENCE_SCHEMA:
            failures.append(
                f"empirical guard evidence schema must be {EMPIRICAL_GUARD_EVIDENCE_SCHEMA}."
            )
        if self.payload.get("authority") != EMPIRICAL_GUARD_EVIDENCE_AUTHORITY:
            failures.append(
                "empirical guard evidence authority must be "
                f"{EMPIRICAL_GUARD_EVIDENCE_AUTHORITY}."
            )
        self._validate_source_commands(failures)
        self._validate_guard_rows(failures)
        self._validate_model_family_rows(failures)
        return failures

    def _validate_source_commands(self, failures: list[str]) -> None:
        if self.payload is None:
            return
        commands = self.payload.get("source_commands")
        if not isinstance(commands, list) or not commands:
            failures.append(
                "empirical evidence source_commands must be a non-empty list."
            )
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

    def _validate_guard_rows(self, failures: list[str]) -> None:
        if self.payload is None:
            return
        rows = self.payload.get("guard_rows")
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
                root=self.root, failures=failures
            )
            if guard is not None:
                observed.add(guard)
        missing = sorted(REQUIRED_GUARDS - observed)
        if missing:
            failures.append(
                "empirical evidence missing guard rows: " + ", ".join(missing)
            )

    def _validate_model_family_rows(self, failures: list[str]) -> None:
        if self.payload is None:
            return
        rows = self.payload.get("model_family_rows")
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
                root=self.root,
                failures=failures,
            )

    def summary(self, failures: list[str]) -> dict[str, object]:
        return {
            "schema": EMPIRICAL_GUARD_EVIDENCE_CHECK_SCHEMA,
            "root": str(self.root),
            "ok": not failures,
            "authoritative": False,
            "authority": EMPIRICAL_GUARD_EVIDENCE_AUTHORITY,
            "scope": "artifact_inventory_only",
            "failures": failures,
        }


__all__ = [
    "ALLOWED_EVIDENCE_KINDS",
    "EMPIRICAL_GUARD_EVIDENCE_CHECK_SCHEMA",
    "EMPIRICAL_GUARD_EVIDENCE_AUTHORITY",
    "EMPIRICAL_GUARD_EVIDENCE_SCHEMA",
    "EmpiricalGuardEvidenceManifest",
    "GuardEvidenceRow",
    "ModelFamilyEvidenceRow",
    "REAL_PRODUCER_MARKERS",
    "REQUIRED_GUARDS",
    "_MANIFEST_OBJECT_ERROR",
    "resolve_artifact",
]
