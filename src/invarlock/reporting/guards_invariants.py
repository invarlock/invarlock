from __future__ import annotations

import copy
from typing import Any

from invarlock.guards.invariant_embeddings import embedding_vocab_size_matches

from .report_types import RunReport

_VOCAB_COERCION_ERRORS = (OverflowError, TypeError, ValueError)
_BNB_RUNTIME_LINEAR_TYPES = {
    "bitsandbytes.nn.modules.Linear4bit",
    "bitsandbytes.nn.modules.Linear8bitLt",
}
_DENSE_LINEAR_TYPE = "torch.nn.modules.linear.Linear"


def _string_map(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            return None
        result[key] = item
    return result


def _linear_dimension_map(value: Any) -> dict[str, tuple[int, int]] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, tuple[int, int]] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(item, list | tuple)
            or len(item) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) for part in item)
            or any(part <= 0 for part in item)
        ):
            return None
        result[key] = (item[0], item[1])
    return result


def _parameter_shape_map(value: Any) -> dict[str, tuple[int, ...]] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, tuple[int, ...]] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(item, list | tuple)
            or any(isinstance(part, bool) or not isinstance(part, int) for part in item)
            or any(part <= 0 for part in item)
        ):
            return None
        result[key] = tuple(item)
    return result


def _shape_numel(shape: tuple[int, ...]) -> int:
    result = 1
    for part in shape:
        result *= part
    return result


def _bnb_parameter_transition(
    *,
    baseline_checks: dict[str, Any],
    current_checks: dict[str, Any],
    changed_modules: dict[str, str],
    baseline_dimensions: dict[str, tuple[int, int]],
) -> tuple[bool, str]:
    """Check that the reported parameter-count delta is fully explained.

    This validates consistency between persisted guard fields. It is not a
    live-runtime proof; release authority must cross-bind the separate runtime
    quantization proof produced from the loaded model.
    """

    baseline_shapes = _parameter_shape_map(baseline_checks.get("parameter_shapes"))
    current_shapes = _parameter_shape_map(current_checks.get("parameter_shapes"))
    if baseline_shapes is None or current_shapes is None:
        return False, "parameter_shape_map_missing"
    if set(baseline_shapes) != set(current_shapes):
        return False, "parameter_paths_changed"

    baseline_count = baseline_checks.get("parameter_count")
    current_count = current_checks.get("parameter_count")
    if (
        isinstance(baseline_count, bool)
        or not isinstance(baseline_count, int)
        or baseline_count <= 0
        or isinstance(current_count, bool)
        or not isinstance(current_count, int)
        or current_count <= 0
    ):
        return False, "parameter_count_missing"
    if sum(_shape_numel(shape) for shape in baseline_shapes.values()) != baseline_count:
        return False, "baseline_parameter_inventory_count_mismatch"
    if sum(_shape_numel(shape) for shape in current_shapes.values()) != current_count:
        return False, "current_parameter_inventory_count_mismatch"

    scoped_paths: set[str] = set()
    expected_count = baseline_count
    for module_path, current_type in changed_modules.items():
        prefix = f"{module_path}." if module_path else ""
        weight_path = f"{prefix}weight"
        bias_path = f"{prefix}bias"
        expected_paths = {weight_path}
        if bias_path in baseline_shapes:
            expected_paths.add(bias_path)
        actual_scoped_paths = {
            path for path in baseline_shapes if path.startswith(prefix)
        }
        if actual_scoped_paths != expected_paths:
            return False, "unexpected_baseline_quantized_module_parameters"
        if {
            path for path in current_shapes if path.startswith(prefix)
        } != expected_paths:
            return False, "unexpected_current_quantized_module_parameters"

        in_features, out_features = baseline_dimensions[module_path]
        baseline_weight_shape = baseline_shapes[weight_path]
        current_weight_shape = current_shapes[weight_path]
        if baseline_weight_shape != (out_features, in_features):
            return False, "baseline_linear_weight_shape_mismatch"
        if bias_path in expected_paths and (
            baseline_shapes[bias_path] != (out_features,)
            or current_shapes[bias_path] != (out_features,)
        ):
            return False, "linear_bias_shape_changed"

        logical_weight_numel = in_features * out_features
        current_weight_numel = _shape_numel(current_weight_shape)
        if current_type == "bitsandbytes.nn.modules.Linear8bitLt":
            if current_weight_shape != (out_features, in_features):
                return False, "bnb8_weight_shape_invalid"
        elif current_type == "bitsandbytes.nn.modules.Linear4bit":
            packed_numel = (logical_weight_numel + 1) // 2
            if current_weight_shape != (out_features, in_features) and (
                current_weight_numel != packed_numel
            ):
                return False, "bnb4_weight_shape_invalid"
        else:  # Defensive: callers already restrict the accepted type set.
            return False, "unrecognized_module_substitution"

        expected_count += current_weight_numel - logical_weight_numel
        scoped_paths.update(expected_paths)

    if any(
        baseline_shapes[path] != current_shapes[path]
        for path in baseline_shapes
        if path not in scoped_paths
    ):
        return False, "out_of_scope_parameter_shape_changed"
    if current_count != expected_count:
        return False, "parameter_count_delta_unexplained"
    return True, "parameter_transition_consistent"


def _bnb_structure_transition(
    baseline_checks: dict[str, Any],
    current_checks: dict[str, Any],
) -> tuple[bool, str]:
    """Validate exact persisted dense-Linear to BNB report consistency.

    This does not establish that the reported modules were observed live. A
    release claim must additionally cross-bind an independently validated live
    runtime proof.
    """

    observation = current_checks.get("quantized_runtime_observation")
    if not isinstance(observation, dict):
        return False, "runtime_observation_missing"
    if set(observation) != {"schema", "adapter", "count", "types", "kinds", "modules"}:
        return False, "runtime_observation_fields_invalid"
    if observation.get("schema") != "invarlock/quantized-structure-observation-v1":
        return False, "runtime_observation_schema_mismatch"
    if observation.get("adapter") != "hf_bnb":
        return False, "runtime_observation_adapter_mismatch"
    count = observation.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        return False, "recognized_runtime_module_count_missing"
    kinds = observation.get("kinds")
    if kinds != ["module"]:
        return False, "recognized_runtime_observation_kind_mismatch"
    types = observation.get("types")
    if (
        not isinstance(types, list)
        or not types
        or types != sorted(set(types))
        or not all(isinstance(item, str) for item in types)
        or not set(types).issubset(_BNB_RUNTIME_LINEAR_TYPES)
    ):
        return False, "recognized_runtime_types_invalid"
    observed_modules = _string_map(observation.get("modules"))
    baseline_modules = _string_map(baseline_checks.get("module_type_paths"))
    current_modules = _string_map(current_checks.get("module_type_paths"))
    if observed_modules is None or baseline_modules is None or current_modules is None:
        return False, "module_type_map_missing"
    if set(baseline_modules) != set(current_modules):
        return False, "module_paths_changed"
    baseline_dimensions = _linear_dimension_map(
        baseline_checks.get("linear_dimensions")
    )
    current_dimensions = _linear_dimension_map(current_checks.get("linear_dimensions"))
    if baseline_dimensions is None or current_dimensions is None:
        return False, "linear_dimensions_missing"
    if baseline_dimensions != current_dimensions:
        return False, "logical_linear_dimensions_changed"
    changed_modules = {
        path: current_type
        for path, current_type in current_modules.items()
        if baseline_modules[path] != current_type
    }
    if not changed_modules or changed_modules != observed_modules:
        return False, "runtime_observation_does_not_bind_all_structure_changes"
    if len(observed_modules) != count:
        return False, "runtime_observation_count_mismatch"
    if set(observed_modules.values()) != set(types):
        return False, "runtime_observation_type_mismatch"
    if any(
        baseline_modules[path] != _DENSE_LINEAR_TYPE
        or current_type not in _BNB_RUNTIME_LINEAR_TYPES
        for path, current_type in changed_modules.items()
    ):
        return False, "unrecognized_module_substitution"
    parameters_ok, parameter_reason = _bnb_parameter_transition(
        baseline_checks=baseline_checks,
        current_checks=current_checks,
        changed_modules=changed_modules,
        baseline_dimensions=baseline_dimensions,
    )
    if not parameters_ok:
        return False, parameter_reason
    return True, "reported_bnb_linear_substitutions_consistent"


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
            size_matches, current_size = embedding_vocab_size_matches(
                baseline_vocab_sizes,
                current_vocab_sizes,
                str(module_name),
                baseline_size,
            )
            if not size_matches:
                violations.append(
                    {
                        "type": "tokenizer_mismatch",
                        "message": "Embedding vocabulary size changed vs baseline",
                        "module": module_name,
                        "baseline": int(baseline_size),
                        "current": current_size,
                    }
                )

    handled_keys = {
        "layer_norm_paths",
        "embedding_vocab_sizes",
        "config_vocab_size",
        "quantized_runtime_observation",
    }
    bnb_allowed, bnb_reason = _bnb_structure_transition(baseline_checks, current_checks)
    runtime_observation = current_checks.get("quantized_runtime_observation")
    if isinstance(runtime_observation, dict) and not bnb_allowed:
        violations.append(
            {
                "type": "quantized_structure_unproven",
                "check": "structure_hash",
                "reason": bnb_reason,
                "message": (
                    "Quantized structure report fields do not consistently bind "
                    "the dense-to-quantized module substitutions"
                ),
            }
        )
    transition_keys = {
        "parameter_count",
        "structure_hash",
        "module_type_paths",
        "parameter_shapes",
    }
    for check_name, baseline_value in baseline_checks.items():
        if check_name in handled_keys or (
            check_name in transition_keys and bnb_allowed
        ):
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
                        f"Invariant {check_name} changed from {baseline_value} "
                        f"to {current_value}"
                    ),
                }
            )

    fatal_types = {"tokenizer_mismatch", "quantized_structure_unproven"}
    annotated: list[dict[str, Any]] = []
    fatal_count = 0
    for violation in violations:
        violation_type = str(violation.get("type") or "")
        annotated_violation = dict(violation)
        annotated_violation.setdefault(
            "severity", "error" if violation_type in fatal_types else "warning"
        )
        annotated.append(annotated_violation)
        fatal_count += violation_type in fatal_types
    return annotated, fatal_count, len(annotated) - fatal_count


def _metric_invariant_failures(invariants_data: Any) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not isinstance(invariants_data, dict):
        return failures
    for check_name, check_result in invariants_data.items():
        if not isinstance(check_result, dict):
            if check_result is not True:
                failures.append(
                    {"check": check_name, "type": "failure", "severity": "error"}
                )
            continue
        if check_result.get("passed") is True:
            continue
        violations = check_result.get("violations")
        recorded_violation = False
        if isinstance(violations, list):
            for violation in violations:
                if not isinstance(violation, dict):
                    continue
                entry: dict[str, Any] = {
                    "check": check_name,
                    "type": str(violation.get("type", "violation")),
                    "severity": violation.get("severity", "warning"),
                }
                detail = {
                    key: value for key, value in violation.items() if key != "type"
                }
                if detail:
                    entry["detail"] = detail
                failures.append(entry)
                recorded_violation = True
        if recorded_violation:
            continue
        detail = {
            key: value
            for key, value in check_result.items()
            if key not in {"passed", "violations", "type"}
        }
        if check_result.get("message"):
            detail.setdefault("message", check_result["message"])
        failure: dict[str, Any] = {
            "check": check_name,
            "type": str(check_result.get("type") or "failure"),
            "severity": "error",
        }
        if detail:
            failure["detail"] = detail
        failures.append(failure)
    return failures


def _select_guard_entries(
    report: RunReport, baseline: RunReport | None
) -> tuple[Any, Any, Any, bool]:
    pre_entry: Any = None
    post_entry: Any = None
    for guard in report.get("guards", []) or []:
        guard_name = str(guard.get("name", "")).lower()
        stage = str(guard.get("stage", "")).lower()
        if guard_name == "invariants_post" or stage == "post":
            post_entry = guard
        elif guard_name == "invariants":
            pre_entry = guard
    staged = post_entry is not None or (
        isinstance(pre_entry, dict) and str(pre_entry.get("stage", "")).lower() == "pre"
    )
    baseline_entry: Any = None
    if baseline is not None:
        for guard in baseline.get("guards", []) or []:
            guard_name = str(guard.get("name", "")).lower()
            stage = str(guard.get("stage", "")).lower()
            if guard_name == "invariants_post" or stage == "post":
                baseline_entry = guard
                break
            if guard_name == "invariants" and baseline_entry is None:
                baseline_entry = guard
    return pre_entry, post_entry, baseline_entry, staged


def _append_guard_verdict_failure(
    failures: list[dict[str, Any]], guard_entry: Any
) -> None:
    if not isinstance(guard_entry, dict):
        return
    passed = guard_entry.get("passed")
    decision = guard_entry.get("decision")
    if not isinstance(passed, bool):
        failures.append(
            {
                "check": "invariants",
                "type": "missing_explicit_verdict",
                "severity": "error",
                "detail": {
                    "message": "Invariant guard evidence lacks an explicit pass verdict"
                },
            }
        )
        return
    if passed is not False and decision not in {"block", "rollback"}:
        return
    metrics = guard_entry.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    violations = guard_entry.get("violations")
    violations = violations if isinstance(violations, list) else []
    fatal = bool(metrics.get("fatal_violations")) or any(
        isinstance(item, dict)
        and str(item.get("severity") or "").lower() in {"error", "fatal"}
        for item in violations
    )
    if not fatal:
        failures.append(
            {
                "check": "invariants",
                "type": "guard_verdict_failed",
                "severity": "error",
                "detail": {"passed": passed, "decision": decision},
            }
        )


def _failure_row(
    violation: dict[str, Any], *, baseline: bool = False
) -> dict[str, Any]:
    check_name = violation.get("check") or violation.get("name")
    if baseline and not check_name:
        check_name = violation.get("module") or violation.get("type")
    row: dict[str, Any] = {
        "check": str(check_name or "invariant"),
        "type": str(violation.get("type") or "violation"),
        "severity": str(violation.get("severity") or "warning"),
    }
    detail = {key: value for key, value in violation.items() if key not in row}
    if detail:
        if baseline:
            detail.setdefault("source", "baseline_compare")
        row["detail"] = detail
    return row


def _guard_summary(
    guard_entry: Any,
    baseline_guard_entry: Any,
    failures: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    if not isinstance(guard_entry, dict):
        return {}, "pass"
    metrics = guard_entry.get("metrics", {}) or {}
    summary = {
        "checks_performed": metrics.get("checks_performed"),
        "violations_found": metrics.get("violations_found"),
        "fatal_violations": metrics.get("fatal_violations"),
        "warning_violations": metrics.get("warning_violations"),
    }
    violations = guard_entry.get("violations", [])
    violations = violations if isinstance(violations, list) else []
    failures.extend(
        _failure_row(violation)
        for violation in violations
        if isinstance(violation, dict)
    )
    fatal_count = int(metrics.get("fatal_violations", 0) or 0)
    warning_count = int(metrics.get("warning_violations", 0) or 0)
    baseline_pre, baseline_post = _extract_guard_checks(baseline_guard_entry)
    current_pre, current_post = _extract_guard_checks(guard_entry)
    baseline_snapshot = baseline_pre or baseline_post
    current_snapshot = current_post or current_pre
    if isinstance(baseline_snapshot, dict) and isinstance(current_snapshot, dict):
        compared, baseline_fatal, baseline_warning = _compare_invariants(
            baseline_snapshot, current_snapshot
        )
        failures.extend(_failure_row(item, baseline=True) for item in compared)
        fatal_count += baseline_fatal
        warning_count += baseline_warning
    summary["fatal_violations"] = fatal_count
    summary["warning_violations"] = warning_count
    summary["violations_found"] = fatal_count + warning_count
    if fatal_count > 0:
        return summary, "fail"
    if warning_count > 0 or violations:
        return summary, "warn"
    return summary, "pass"


def _extract_invariants(
    report: RunReport, baseline: RunReport | None = None
) -> dict[str, Any]:
    """Extract invariant check results (matches the shape used in tests)."""
    invariants_data = (report.get("metrics", {}) or {}).get("invariants", {})
    failures = _metric_invariant_failures(invariants_data)
    pre_guard_entry, post_guard_entry, baseline_guard_entry, staged_guard_evidence = (
        _select_guard_entries(report, baseline)
    )
    guard_entry: Any = post_guard_entry or pre_guard_entry

    evidence_present = bool(invariants_data) or guard_entry is not None
    severity_status = "pass" if evidence_present else "unknown"
    _append_guard_verdict_failure(failures, guard_entry)
    summary: dict[str, Any] = {}
    if guard_entry:
        summary, severity_status = _guard_summary(
            guard_entry, baseline_guard_entry, failures
        )

    post_status = severity_status
    if failures:
        post_has_error = any(
            str(failure.get("severity", "warning")) == "error" for failure in failures
        )
        if post_has_error:
            post_status = "fail"
        elif post_status == "pass":
            post_status = "warn"

    pre_status = "pass" if evidence_present else "unknown"
    if staged_guard_evidence:
        pre_passed = bool(
            isinstance(pre_guard_entry, dict)
            and pre_guard_entry.get("passed") is True
            and pre_guard_entry.get("decision") not in {"block", "rollback"}
        )
        pre_status = "pass" if pre_passed else "fail"
        if isinstance(pre_guard_entry, dict) and pre_guard_entry is not guard_entry:
            pre_metrics = pre_guard_entry.get("metrics")
            pre_metrics = pre_metrics if isinstance(pre_metrics, dict) else {}
            for field in (
                "checks_performed",
                "violations_found",
                "fatal_violations",
                "warning_violations",
            ):
                try:
                    summary[field] = int(summary.get(field, 0) or 0) + int(
                        pre_metrics.get(field, 0) or 0
                    )
                except _VOCAB_COERCION_ERRORS:
                    pass
            pre_violations = pre_guard_entry.get("violations")
            if isinstance(pre_violations, list):
                for violation in pre_violations:
                    if not isinstance(violation, dict):
                        continue
                    failures.append(
                        {
                            "check": str(
                                violation.get("check")
                                or violation.get("name")
                                or "invariants_pre"
                            ),
                            "type": str(violation.get("type") or "violation"),
                            "severity": str(violation.get("severity") or "error"),
                            "detail": {
                                "source": "pre_edit",
                                **{
                                    key: value
                                    for key, value in violation.items()
                                    if key not in {"check", "name", "type", "severity"}
                                },
                            },
                        }
                    )
        if not pre_passed:
            severity_status = "fail"
            if not any(
                isinstance(item.get("detail"), dict)
                and item["detail"].get("source") == "pre_edit"
                for item in failures
            ):
                failures.append(
                    {
                        "check": "invariants_pre",
                        "type": "stage_failed",
                        "severity": "error",
                        "detail": {"source": "pre_edit"},
                    }
                )

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
    if staged_guard_evidence:
        details_out = {
            "pre": (
                pre_guard_entry.get("details", {})
                if isinstance(pre_guard_entry, dict)
                else {}
            ),
            "post": (
                post_guard_entry.get("details", {})
                if isinstance(post_guard_entry, dict)
                else {}
            ),
        }
    elif (
        not details_out and guard_entry and isinstance(guard_entry.get("details"), dict)
    ):
        details_out = guard_entry.get("details", {})

    post_passed = post_status == "pass" or post_status == "warn"
    overall_passed = pre_status == "pass" and post_passed

    return {
        "pre": pre_status,
        "post": post_status,
        "status": "fail" if not overall_passed else status,
        "passed": overall_passed,
        "decision": "allow" if overall_passed else "block",
        "summary": summary,
        "details": copy.deepcopy(details_out),
        "failures": failures,
    }
