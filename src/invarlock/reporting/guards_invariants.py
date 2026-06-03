from __future__ import annotations

from collections import Counter
from typing import Any

from .report_types import RunReport

_VOCAB_COERCION_ERRORS = (OverflowError, TypeError, ValueError)


def _coerce_vocab_counts(vocab_sizes: Any) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not isinstance(vocab_sizes, dict):
        return counts
    for value in vocab_sizes.values():
        try:
            counts[int(value)] += 1
        except _VOCAB_COERCION_ERRORS:
            continue
    return counts


def _embedding_vocab_size_matches(
    baseline_vocab_sizes: Any,
    current_vocab_sizes: Any,
    module_name: str,
    baseline_size: Any,
) -> tuple[bool, int | None]:
    try:
        expected = int(baseline_size)
    except _VOCAB_COERCION_ERRORS:
        return False, None
    current_size = None
    if isinstance(current_vocab_sizes, dict):
        current_size = current_vocab_sizes.get(module_name)
    if current_size is not None:
        try:
            current_int = int(current_size)
        except _VOCAB_COERCION_ERRORS:
            return False, None
        return current_int == expected, current_int

    baseline_counts = _coerce_vocab_counts(baseline_vocab_sizes)
    current_counts = _coerce_vocab_counts(current_vocab_sizes)
    if baseline_counts and current_counts.get(expected, 0) >= baseline_counts[expected]:
        return True, expected
    return False, None


def _extract_invariants(
    report: RunReport, baseline: RunReport | None = None
) -> dict[str, Any]:
    """Extract invariant check results (matches the shape used in tests)."""
    invariants_data = (report.get("metrics", {}) or {}).get("invariants", {})
    failures: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    # Collect failures from metrics.invariants
    if isinstance(invariants_data, dict) and invariants_data:
        for check_name, check_result in invariants_data.items():
            if isinstance(check_result, dict):
                if bool(check_result.get("passed", True)):
                    continue
                recorded_violation = False
                violations = check_result.get("violations")
                if isinstance(violations, list) and violations:
                    for violation in violations:
                        if not isinstance(violation, dict):
                            continue
                        entry: dict[str, Any] = {
                            "check": check_name,
                            "type": str(violation.get("type", "violation")),
                            "severity": violation.get("severity", "warning"),
                        }
                        detail = {k: v for k, v in violation.items() if k != "type"}
                        if detail:
                            entry["detail"] = detail
                        failures.append(entry)
                        recorded_violation = True
                if recorded_violation:
                    continue
                # No explicit violations list – treat as error
                failure_entry: dict[str, Any] = {"check": check_name}
                failure_entry["type"] = str(check_result.get("type") or "failure")
                failure_entry["severity"] = "error"
                detail = {
                    k: v
                    for k, v in check_result.items()
                    if k not in {"passed", "violations", "type"}
                }
                if check_result.get("message"):
                    detail.setdefault("message", check_result["message"])
                if detail:
                    failure_entry["detail"] = detail
                failures.append(failure_entry)
            else:
                if not bool(check_result):
                    failures.append(
                        {"check": check_name, "type": "failure", "severity": "error"}
                    )

    guard_entry: Any = None
    for guard in report.get("guards", []) or []:
        if str(guard.get("name", "")).lower() == "invariants":
            guard_entry = guard
            break

    baseline_guard_entry: Any = None
    if baseline is not None:
        for guard in baseline.get("guards", []) or []:
            if str(guard.get("name", "")).lower() == "invariants":
                baseline_guard_entry = guard
                break

    def _coerce_checks(value: Any) -> dict[str, Any] | None:
        return value if isinstance(value, dict) else None

    def _extract_guard_checks(
        entry: Any,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not isinstance(entry, dict):
            return None, None
        details = entry.get("details")
        if not isinstance(details, dict):
            return None, None
        return _coerce_checks(details.get("baseline_checks")), _coerce_checks(
            details.get("current_checks")
        )

    def _compare_invariants(
        baseline_checks: dict[str, Any],
        current_checks: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int, int]:
        violations: list[dict[str, Any]] = []

        baseline_layer_norms = set(baseline_checks.get("layer_norm_paths", ()))
        current_layer_norms = set(current_checks.get("layer_norm_paths", ()))
        missing_layer_norms = sorted(baseline_layer_norms - current_layer_norms)
        if missing_layer_norms:
            violations.append(
                {
                    "type": "layer_norm_missing",
                    "missing": missing_layer_norms,
                    "message": "Expected LayerNorm modules are missing vs baseline",
                }
            )

        baseline_vocab_sizes = baseline_checks.get("embedding_vocab_sizes")
        current_vocab_sizes = current_checks.get("embedding_vocab_sizes")
        if isinstance(baseline_vocab_sizes, dict):
            for module_name, baseline_size in baseline_vocab_sizes.items():
                size_matches, current_size = _embedding_vocab_size_matches(
                    baseline_vocab_sizes,
                    current_vocab_sizes,
                    str(module_name),
                    baseline_size,
                )
                if not size_matches:
                    mismatch = {
                        "module": module_name,
                        "baseline": int(baseline_size),
                        "current": current_size,
                    }
                    violations.append(
                        {
                            "type": "tokenizer_mismatch",
                            "message": "Embedding vocabulary size changed vs baseline",
                            **mismatch,
                        }
                    )

        handled_keys = {
            "layer_norm_paths",
            "embedding_vocab_sizes",
            "config_vocab_size",
        }
        for check_name, baseline_value in baseline_checks.items():
            if check_name in handled_keys:
                continue
            current_value = current_checks.get(check_name)
            if current_value != baseline_value:
                violations.append(
                    {
                        "type": "invariant_violation",
                        "check": check_name,
                        "baseline": baseline_value,
                        "current": current_value,
                        "message": (
                            f"Invariant {check_name} changed from {baseline_value} to {current_value}"
                        ),
                    }
                )

        fatal_violation_types = {"tokenizer_mismatch"}
        fatal_count = 0
        warning_count = 0
        annotated: list[dict[str, Any]] = []
        for violation in violations:
            violation_type = str(violation.get("type") or "")
            severity = "fatal" if violation_type in fatal_violation_types else "warning"
            annotated_violation = dict(violation)
            annotated_violation.setdefault("severity", severity)
            annotated.append(annotated_violation)
            if severity == "fatal":
                fatal_count += 1
            else:
                warning_count += 1

        return annotated, fatal_count, warning_count

    severity_status = "pass"
    if guard_entry:
        gm = guard_entry.get("metrics", {}) or {}
        summary = {
            "checks_performed": gm.get("checks_performed"),
            "violations_found": gm.get("violations_found"),
            "fatal_violations": gm.get("fatal_violations"),
            "warning_violations": gm.get("warning_violations"),
        }
        violations = guard_entry.get("violations", [])
        fatal_count = int(gm.get("fatal_violations", 0) or 0)
        warning_count = int(gm.get("warning_violations", 0) or 0)
        if violations:
            for violation in violations:
                if not isinstance(violation, dict):
                    continue
                row: dict[str, Any] = {
                    "check": str(
                        violation.get("check") or violation.get("name") or "invariant"
                    ),
                    "type": str(violation.get("type") or "violation"),
                    "severity": str(violation.get("severity") or "warning"),
                }
                failure_detail: dict[str, Any] = {
                    k: v for k, v in violation.items() if k not in row
                }
                if failure_detail:
                    row["detail"] = failure_detail
                failures.append(row)
        base_fatal = 0
        base_warn = 0
        baseline_failures: list[dict[str, Any]] = []
        if baseline_guard_entry is not None:
            baseline_pre, baseline_post = _extract_guard_checks(baseline_guard_entry)
            current_pre, current_post = _extract_guard_checks(guard_entry)
            baseline_snapshot = baseline_pre or baseline_post
            current_snapshot = current_post or current_pre
            if isinstance(baseline_snapshot, dict) and isinstance(
                current_snapshot, dict
            ):
                baseline_failures, base_fatal, base_warn = _compare_invariants(
                    baseline_snapshot, current_snapshot
                )
                for violation in baseline_failures:
                    baseline_check_name: Any = violation.get("check")
                    if not baseline_check_name:
                        baseline_check_name = (
                            violation.get("module")
                            or violation.get("type")
                            or "invariant"
                        )
                    baseline_row: dict[str, Any] = {
                        "check": str(baseline_check_name),
                        "type": str(violation.get("type") or "violation"),
                        "severity": str(violation.get("severity") or "warning"),
                    }
                    baseline_detail: dict[str, Any] = {
                        k: v for k, v in violation.items() if k not in baseline_row
                    }
                    if baseline_detail:
                        baseline_detail.setdefault("source", "baseline_compare")
                        baseline_row["detail"] = baseline_detail
                    failures.append(baseline_row)

        fatal_total = fatal_count + base_fatal
        warn_total = warning_count + base_warn
        try:
            summary["fatal_violations"] = fatal_total
            summary["warning_violations"] = warn_total
            summary["violations_found"] = fatal_total + warn_total
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
            pass

        if fatal_total > 0:
            severity_status = "fail"
        elif warn_total > 0 or violations:
            severity_status = "warn"

    if failures:
        has_error = any(str(f.get("severity", "warning")) == "error" for f in failures)
        if has_error:
            severity_status = "fail"
        elif severity_status == "pass":
            severity_status = "warn"

    status = severity_status
    if not summary:
        summary = {
            "checks_performed": 0,
            "violations_found": len(failures),
            "fatal_violations": 0,
            "warning_violations": len(failures),
        }

    details_out: dict[str, Any] = invariants_data
    if not details_out and guard_entry and isinstance(guard_entry.get("details"), dict):
        details_out = guard_entry.get("details", {})

    return {
        "pre": "pass",
        "post": status,
        "status": status,
        "summary": summary,
        "details": details_out,
        "failures": failures,
    }
