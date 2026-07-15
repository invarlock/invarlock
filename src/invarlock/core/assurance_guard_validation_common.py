"""Shared parsing and reconciliation helpers for assurance guard evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

_GUARD_NAMES = ("spectral", "rmt", "variance", "invariants")
_PASS_DECISIONS = frozenset({"allow", "allowed", "pass", "passed", "ok"})
_FAIL_DECISIONS = frozenset(
    {
        "block",
        "blocked",
        "deny",
        "denied",
        "fail",
        "failed",
        "reject",
        "rejected",
        "rollback",
    }
)
_PASS_STATUSES = frozenset({"ok", "pass", "passed", "stable", "success"})
_FAIL_STATUSES = frozenset(
    {
        "block",
        "blocked",
        "degraded",
        "deny",
        "denied",
        "error",
        "fail",
        "failed",
        "monitor-only",
        "rejected",
        "rollback",
        "unstable",
        "unsupported",
    }
)
_ERROR_SEVERITIES = frozenset({"critical", "error", "fatal"})
_DIAGNOSTIC_SEVERITIES = _ERROR_SEVERITIES | frozenset(
    {"budgeted", "info", "note", "warn", "warning"}
)


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _normalized_token(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _finite_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, tuple | list) or len(value) != 2:
        return None
    lower = _finite_number(value[0])
    upper = _finite_number(value[1])
    if lower is None or upper is None:
        return None
    return lower, upper


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _guard_base_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip().lower()
    if name == "invariants_post":
        return "invariants"
    return name if name in _GUARD_NAMES else None


def _guard_inventory(
    report: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any], str]]:
    raw = report.get("guards")
    if not isinstance(raw, list):
        return []
    entries: list[tuple[str, dict[str, Any], str]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            continue
        guard_name = _guard_base_name(value.get("name"))
        if guard_name is not None:
            entries.append((guard_name, value, f"guards[{index}]"))
    return entries


def _validate_diagnostics(source: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [f"{source}.diagnostics must be an array."]
    errors: list[str] = []
    for index, diagnostic in enumerate(value):
        if not isinstance(diagnostic, dict):
            errors.append(f"{source}.diagnostics[{index}] must be an object.")
            continue
        severity = diagnostic.get("severity")
        if severity is None:
            errors.append(f"{source}.diagnostics[{index}].severity is required.")
            continue
        if not isinstance(severity, str):
            errors.append(f"{source}.diagnostics[{index}].severity must be a string.")
            continue
        normalized_severity = _normalized_token(severity)
        if normalized_severity not in _DIAGNOSTIC_SEVERITIES:
            errors.append(
                f"{source}.diagnostics[{index}].severity is unsupported: {severity}."
            )
        elif normalized_severity in _ERROR_SEVERITIES:
            errors.append(
                f"{source}.diagnostics[{index}] records a blocking {severity} event."
            )
    return errors


def _validate_guard_outcome(
    guard_name: str,
    block: Any,
    *,
    source: str,
    require_complete: bool,
) -> list[str]:
    if not isinstance(block, dict) or not block:
        return (
            [f"strict assurance missing {source} guard evidence."]
            if require_complete
            else []
        )

    errors: list[str] = []
    spectral_intervention = guard_name == "spectral"
    supported = block.get("supported")
    if "supported" not in block:
        if require_complete:
            errors.append(f"{source}.supported is required for strict assurance.")
    elif not isinstance(supported, bool):
        errors.append(f"{source}.supported must be a boolean.")
    elif supported is not True:
        reason = block.get("reason")
        suffix = f": {reason}" if isinstance(reason, str) and reason else ""
        errors.append(f"{source} is unsupported for strict assurance{suffix}.")

    passed = block.get("passed")
    if "passed" not in block:
        if require_complete:
            errors.append(f"{source}.passed is required for strict assurance.")
    elif not isinstance(passed, bool):
        errors.append(f"{source}.passed must be a boolean.")
    elif passed is not True:
        errors.append(f"{source}.passed is false; {guard_name} did not pass.")

    decision = block.get("decision")
    if "decision" not in block:
        if require_complete:
            errors.append(f"{source}.decision is required for strict assurance.")
    elif not isinstance(decision, str) or not decision.strip():
        errors.append(f"{source}.decision must be a non-empty string.")
    else:
        normalized_decision = _normalized_token(decision)
        if normalized_decision in _FAIL_DECISIONS:
            errors.append(f"{source}.decision={decision!r} is blocking.")
        elif (
            require_complete
            and normalized_decision not in _PASS_DECISIONS
            and not (spectral_intervention and normalized_decision == "monitor")
        ):
            errors.append(
                f"{source}.decision must be an allow/pass decision for strict assurance."
            )

    status = block.get("status")
    if "status" in block:
        if not isinstance(status, str) or not status.strip():
            errors.append(f"{source}.status must be a non-empty string.")
        else:
            normalized_status = _normalized_token(status)
            if normalized_status in _FAIL_STATUSES:
                errors.append(f"{source}.status={status!r} is not passing.")
            elif (
                require_complete
                and normalized_status not in _PASS_STATUSES
                and not (spectral_intervention and normalized_status == "capped")
            ):
                errors.append(
                    f"{source}.status must be a canonical passing status when present."
                )

    violations = block.get("violations")
    if "violations" not in block:
        if require_complete:
            errors.append(f"{source}.violations is required for strict assurance.")
    elif not isinstance(violations, list):
        errors.append(f"{source}.violations must be an array.")
    elif violations and not spectral_intervention:
        errors.append(f"{source}.violations must be empty for strict assurance.")

    failures = block.get("failures")
    if "failures" in block:
        if not isinstance(failures, list):
            errors.append(f"{source}.failures must be an array.")
        elif failures:
            errors.append(f"{source}.failures must be empty for strict assurance.")

    for field in ("errors", "warnings"):
        values = block.get(field)
        if field not in block:
            continue
        if not isinstance(values, list):
            errors.append(f"{source}.{field} must be an array.")
        elif values and not (spectral_intervention and field == "warnings"):
            errors.append(f"{source}.{field} must be empty for strict assurance.")

    assurance_blocking = block.get("assurance_blocking")
    if "assurance_blocking" in block and not isinstance(assurance_blocking, bool):
        errors.append(f"{source}.assurance_blocking must be a boolean.")
    elif assurance_blocking is True:
        errors.append(f"{source}.assurance_blocking is true.")

    errors.extend(_validate_diagnostics(source, block.get("diagnostics")))
    return errors


def _field_values(
    sources: Sequence[tuple[str, Mapping[str, Any]]], field: str
) -> list[tuple[str, Any]]:
    return [
        (f"{path}.{field}", block[field]) for path, block in sources if field in block
    ]


def _consistent_bool(
    errors: list[str],
    values: Sequence[tuple[str, Any]],
    *,
    required_path: str | None = None,
) -> bool | None:
    if not values:
        if required_path is not None:
            errors.append(f"{required_path} is required for strict assurance.")
        return None
    parsed: list[tuple[str, bool]] = []
    for path, value in values:
        if not isinstance(value, bool):
            errors.append(f"{path} must be a boolean.")
        else:
            parsed.append((path, value))
    if not parsed:
        return None
    first = parsed[0][1]
    if any(value is not first for _, value in parsed[1:]):
        errors.append(
            f"{', '.join(path for path, _ in parsed)} disagree across guard evidence."
        )
        return None
    return first


def _consistent_nonnegative_int(
    errors: list[str],
    values: Sequence[tuple[str, Any]],
    *,
    required_path: str | None = None,
) -> int | None:
    if not values:
        if required_path is not None:
            errors.append(f"{required_path} is required for strict assurance.")
        return None
    parsed: list[tuple[str, int]] = []
    for path, value in values:
        numeric = _nonnegative_int(value)
        if numeric is None:
            errors.append(f"{path} must be a non-negative integer.")
        else:
            parsed.append((path, numeric))
    if not parsed:
        return None
    first = parsed[0][1]
    if any(value != first for _, value in parsed[1:]):
        errors.append(
            f"{', '.join(path for path, _ in parsed)} disagree across guard evidence."
        )
        return None
    return first
