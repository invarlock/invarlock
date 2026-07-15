"""Cross-surface variance result consistency checks."""

from __future__ import annotations

from typing import Any

from .verify_check_helpers_metrics import _coerce_float, _coerce_int


def _collect_provenance_window_ids(node: Any) -> list[Any]:
    if isinstance(node, dict):
        window_ids = node.get("window_ids")
        if isinstance(window_ids, list):
            return list(window_ids)
        collected: list[Any] = []
        for value in node.values():
            collected.extend(_collect_provenance_window_ids(value))
        return collected
    if isinstance(node, list):
        collected = []
        for value in node:
            collected.extend(_collect_provenance_window_ids(value))
        return collected
    return []


def _validate_variance_enablement(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    variance = report.get("variance") or {}
    if not isinstance(variance, dict):
        return errors
    predictive_gate = variance.get("predictive_gate")
    predictive_success = bool(
        isinstance(predictive_gate, dict)
        and predictive_gate.get("passed") is True
        and str(predictive_gate.get("reason") or "").strip().lower().replace("_", "-")
        != "no-adjustment-required"
    )
    enabled = bool(variance.get("enabled", False))
    if not enabled and not predictive_success:
        return errors

    if predictive_success and not enabled:
        if variance.get("ve_enabled_during_validation") is not True:
            errors.append(
                "variance mitigation validation requires "
                "ve_enabled_during_validation=true."
            )
        if variance.get("subject_restored_after_ab") is not True:
            errors.append(
                "variance mitigation validation requires "
                "subject_restored_after_ab=true."
            )
        if variance.get("met_threshold") is not True:
            errors.append("variance mitigation validation requires met_threshold=true.")

    resolved_policy = report.get("resolved_policy") or {}
    variance_policy = (
        resolved_policy.get("variance") if isinstance(resolved_policy, dict) else {}
    )
    min_effect = 0.0
    if isinstance(variance_policy, dict):
        parsed_min_effect = _coerce_float(variance_policy.get("min_effect_lognll"))
        if parsed_min_effect is not None:
            min_effect = max(0.0, parsed_min_effect)
    improvement_threshold = -min_effect

    if not isinstance(predictive_gate, dict) or not predictive_gate:
        errors.append(
            "variance mitigation validation requires predictive_gate evidence."
        )
    else:
        if predictive_gate.get("passed") is not True:
            errors.append(
                "variance mitigation validation requires predictive_gate.passed == true."
            )

        mean_delta = _coerce_float(predictive_gate.get("mean_delta"))
        if mean_delta is None:
            errors.append(
                "variance mitigation validation requires finite "
                "predictive_gate.mean_delta."
            )
        elif mean_delta >= 0.0:
            errors.append(
                "variance.predictive_gate.mean_delta must be negative for a "
                "successful mitigation."
            )
        elif mean_delta > improvement_threshold:
            errors.append(
                "variance.predictive_gate.mean_delta does not meet "
                f"-min_effect_lognll ({improvement_threshold:.6g})."
            )

        delta_ci = predictive_gate.get("delta_ci")
        if delta_ci is None:
            delta_ci = predictive_gate.get("ci")
        lower = upper = None
        if isinstance(delta_ci, tuple | list) and len(delta_ci) == 2:
            lower = _coerce_float(delta_ci[0])
            upper = _coerce_float(delta_ci[1])
        if lower is None or upper is None:
            errors.append(
                "variance mitigation validation requires finite "
                "predictive_gate.delta_ci."
            )
        elif lower > upper:
            errors.append(
                "variance.predictive_gate.delta_ci lower bound exceeds upper bound."
            )
        elif upper >= 0.0:
            errors.append(
                "variance.predictive_gate.delta_ci must exclude zero for a "
                "successful mitigation."
            )
        elif upper > improvement_threshold:
            errors.append(
                "variance.predictive_gate.delta_ci upper bound does not meet "
                f"-min_effect_lognll ({improvement_threshold:.6g})."
            )

    ab_test = variance.get("ab_test")
    if not isinstance(ab_test, dict) or not ab_test:
        errors.append(
            "variance mitigation validation requires variance.ab_test evidence."
        )
        return errors

    provenance = ab_test.get("provenance")
    seed = ab_test.get("seed")
    if seed in (None, "") and isinstance(provenance, dict):
        seed = provenance.get("seed")
    if seed in (None, ""):
        errors.append("variance mitigation validation requires variance.ab_test.seed.")

    windows_used = _coerce_int(ab_test.get("windows_used"))
    if windows_used is None or windows_used <= 0:
        errors.append(
            "variance mitigation validation requires positive "
            "variance.ab_test.windows_used."
        )

    if not isinstance(provenance, dict) or not provenance:
        errors.append(
            "variance mitigation validation requires variance.ab_test.provenance."
        )
    elif not _collect_provenance_window_ids(provenance):
        errors.append(
            "variance mitigation validation requires "
            "variance.ab_test.provenance.window_ids."
        )

    return errors
