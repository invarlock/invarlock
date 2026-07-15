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


def _validate_diagnostics(
    source: str,
    value: Any,
    *,
    allowed_error_kinds: frozenset[str] = frozenset(),
) -> list[str]:
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
        diagnostic_kind = _normalized_token(str(diagnostic.get("kind") or ""))
        if (
            normalized_severity in _ERROR_SEVERITIES
            and diagnostic_kind not in allowed_error_kinds
        ):
            errors.append(
                f"{source}.diagnostics[{index}] records a blocking {severity} event."
            )
    return errors


def _guard_support_errors(
    block: Mapping[str, Any], *, source: str, require_complete: bool
) -> list[str]:
    if "supported" not in block:
        return (
            [f"{source}.supported is required for strict assurance."]
            if require_complete
            else []
        )
    supported = block.get("supported")
    if not isinstance(supported, bool):
        return [f"{source}.supported must be a boolean."]
    if supported:
        return []
    reason = block.get("reason")
    suffix = f": {reason}" if isinstance(reason, str) and reason else ""
    return [f"{source} is unsupported for strict assurance{suffix}."]


def _guard_passed_errors(
    guard_name: str,
    block: Mapping[str, Any],
    *,
    source: str,
    require_complete: bool,
    enforce_outcome: bool,
) -> list[str]:
    if "passed" not in block:
        return (
            [f"{source}.passed is required for strict assurance."]
            if require_complete
            else []
        )
    passed = block.get("passed")
    if not isinstance(passed, bool):
        return [f"{source}.passed must be a boolean."]
    if not passed and enforce_outcome:
        return [f"{source}.passed is false; {guard_name} did not pass."]
    return []


def _guard_decision_errors(
    block: Mapping[str, Any],
    *,
    source: str,
    require_complete: bool,
    enforce_outcome: bool,
    spectral_intervention: bool,
) -> list[str]:
    if "decision" not in block:
        return (
            [f"{source}.decision is required for strict assurance."]
            if require_complete
            else []
        )
    decision = block.get("decision")
    if not isinstance(decision, str) or not decision.strip():
        return [f"{source}.decision must be a non-empty string."]
    normalized = _normalized_token(decision)
    if normalized in _FAIL_DECISIONS and enforce_outcome:
        return [f"{source}.decision={decision!r} is blocking."]
    known = normalized in _PASS_DECISIONS or normalized in _FAIL_DECISIONS
    if (
        require_complete
        and not known
        and not (spectral_intervention and normalized == "monitor")
    ):
        return [
            f"{source}.decision must be an allow/pass decision for strict assurance."
        ]
    return []


def _guard_status_errors(
    block: Mapping[str, Any],
    *,
    source: str,
    require_complete: bool,
    enforce_outcome: bool,
    spectral_intervention: bool,
) -> list[str]:
    if "status" not in block:
        return []
    status = block.get("status")
    if not isinstance(status, str) or not status.strip():
        return [f"{source}.status must be a non-empty string."]
    normalized = _normalized_token(status)
    evidence_failures = {"degraded", "error", "monitor-only", "unsupported"}
    if normalized in evidence_failures:
        return [f"{source}.status={status!r} is not passing."]
    if normalized in _FAIL_STATUSES and enforce_outcome:
        return [f"{source}.status={status!r} is not passing."]
    known = normalized in _PASS_STATUSES or normalized in _FAIL_STATUSES
    if (
        require_complete
        and not known
        and not (spectral_intervention and normalized == "capped")
    ):
        return [f"{source}.status must be a canonical passing status when present."]
    return []


def _guard_collection_errors(
    block: Mapping[str, Any],
    *,
    source: str,
    require_complete: bool,
    enforce_outcome: bool,
    spectral_intervention: bool,
) -> tuple[list[str], Any]:
    errors: list[str] = []
    violations = block.get("violations")
    if "violations" not in block:
        if require_complete:
            errors.append(f"{source}.violations is required for strict assurance.")
    elif not isinstance(violations, list):
        errors.append(f"{source}.violations must be an array.")
    elif violations and not spectral_intervention and enforce_outcome:
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
    return errors, violations


def _validate_guard_outcome(
    guard_name: str,
    block: Any,
    *,
    source: str,
    require_complete: bool,
    enforce_outcome: bool = True,
) -> list[str]:
    if not isinstance(block, dict) or not block:
        return (
            [f"strict assurance missing {source} guard evidence."]
            if require_complete
            else []
        )

    spectral_intervention = guard_name == "spectral"
    errors = _guard_support_errors(
        block, source=source, require_complete=require_complete
    )
    errors.extend(
        _guard_passed_errors(
            guard_name,
            block,
            source=source,
            require_complete=require_complete,
            enforce_outcome=enforce_outcome,
        )
    )
    errors.extend(
        _guard_decision_errors(
            block,
            source=source,
            require_complete=require_complete,
            enforce_outcome=enforce_outcome,
            spectral_intervention=spectral_intervention,
        )
    )
    errors.extend(
        _guard_status_errors(
            block,
            source=source,
            require_complete=require_complete,
            enforce_outcome=enforce_outcome,
            spectral_intervention=spectral_intervention,
        )
    )
    collection_errors, violations = _guard_collection_errors(
        block,
        source=source,
        require_complete=require_complete,
        enforce_outcome=enforce_outcome,
        spectral_intervention=spectral_intervention,
    )
    errors.extend(collection_errors)

    assurance_blocking = block.get("assurance_blocking")
    if "assurance_blocking" in block and not isinstance(assurance_blocking, bool):
        errors.append(f"{source}.assurance_blocking must be a boolean.")
    elif assurance_blocking is True:
        errors.append(f"{source}.assurance_blocking is true.")

    allowed_error_kinds: frozenset[str] = frozenset()
    if not enforce_outcome and isinstance(violations, list):
        allowed_error_kinds = frozenset(
            _normalized_token(str(item.get("type") or ""))
            for item in violations
            if isinstance(item, dict) and item.get("type")
        )
    errors.extend(
        _validate_diagnostics(
            source,
            block.get("diagnostics"),
            allowed_error_kinds=allowed_error_kinds,
        )
    )
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
