"""Strict replay of finding-bound spectral correction evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .assurance_spectral_replay_common import (
    _close,
    _compare_tree,
    _compare_violations,
    _degeneracy_map,
    _finite,
    _mapping,
    _measurement_close,
    _numeric_map,
)
from .assurance_spectral_replay_decision import replay_selected_findings

_LEDGER_FIELDS = {
    "schema_version",
    "phase",
    "correction_enabled",
    "correction_cap_ratio",
    "pre_correction_metrics",
    "pre_correction_z_scores",
    "pre_correction_degeneracy",
    "multiple_testing_selection",
    "selected_findings",
    "corrections",
    "policy_result",
    "post_correction_metrics",
}
_CORRECTION_FIELDS = {
    "correction_id",
    "finding_ids",
    "module",
    "operation",
    "attempted",
    "mutation_applied",
    "outcome",
    "pre_sigma",
    "baseline_sigma",
    "target_sigma",
    "post_sigma",
    "scale_factor",
    "failure",
    "pre_weight_digest",
    "post_weight_digest",
}


def _finding_bindings(
    errors: list[str], observed_findings: list[Any], source: str
) -> tuple[set[str], dict[str, set[str]]]:
    finding_ids: set[str] = set()
    finding_ids_by_module: dict[str, set[str]] = defaultdict(set)
    for index, raw in enumerate(observed_findings):
        finding = _mapping(raw)
        if finding is None:
            continue
        finding_id = finding.get("finding_id")
        finding_module = finding.get("module")
        if not isinstance(finding_id, str) or not finding_id:
            errors.append(
                f"{source}.correction_ledger.selected_findings[{index}].finding_id "
                "must be a non-empty string."
            )
            continue
        if finding_id in finding_ids:
            errors.append(f"{source}.correction_ledger finding IDs must be unique.")
        finding_ids.add(finding_id)
        if isinstance(finding_module, str) and finding_module:
            finding_ids_by_module[finding_module].add(finding_id)
    return finding_ids, finding_ids_by_module


def _correction_bindings(
    errors: list[str],
    corrections_raw: list[Any],
    source: str,
    finding_ids: set[str],
    finding_ids_by_module: Mapping[str, set[str]],
) -> dict[str, Mapping[str, Any]]:
    corrections_by_module: dict[str, Mapping[str, Any]] = {}
    referenced_findings: set[str] = set()
    expected_correction_ids = {
        module: f"correction-{index:04d}:{module}"
        for index, module in enumerate(sorted(finding_ids_by_module), start=1)
    }
    for index, raw in enumerate(corrections_raw):
        correction = _mapping(raw)
        path = f"{source}.correction_ledger.corrections[{index}]"
        if correction is None:
            errors.append(f"{path} must be an object.")
            continue
        unknown_correction_fields = sorted(set(correction) - _CORRECTION_FIELDS)
        if unknown_correction_fields:
            errors.append(
                f"{path} contains unsupported fields: {unknown_correction_fields}."
            )
        correction_module = correction.get("module")
        if (
            not isinstance(correction_module, str)
            or correction_module not in finding_ids_by_module
        ):
            errors.append(f"{path}.module is not bound to a selected finding.")
            continue
        if correction_module in corrections_by_module:
            errors.append(
                f"{source}.correction_ledger has duplicate module corrections."
            )
        corrections_by_module[correction_module] = correction
        if (
            correction.get("correction_id")
            != expected_correction_ids[correction_module]
        ):
            errors.append(f"{path}.correction_id is not canonical for its module.")
        ids = correction.get("finding_ids")
        if (
            not isinstance(ids, list)
            or set(ids) != finding_ids_by_module[correction_module]
        ):
            errors.append(f"{path}.finding_ids do not bind every module finding.")
        else:
            referenced_findings.update(str(item) for item in ids)
    if set(corrections_by_module) != set(finding_ids_by_module):
        errors.append(
            f"{source}.correction_ledger corrections must cover every selected module."
        )
    if referenced_findings != finding_ids:
        errors.append(
            f"{source}.correction_ledger corrections must reference every selected finding."
        )
    return corrections_by_module


def _digest_is_valid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _replay_correction_entries(
    errors: list[str],
    *,
    source: str,
    corrections_by_module: Mapping[str, Mapping[str, Any]],
    selected: list[dict[str, Any]],
    baseline: Mapping[str, float],
    pre_metrics: Mapping[str, float],
    post_metrics: Mapping[str, float],
    correction_enabled: bool,
    correction_cap_ratio: float,
) -> tuple[int, int, str]:
    applied_count = 0
    attempted_count = 0
    expected_policy_result = "no_selected_findings"
    if selected and not correction_enabled:
        expected_policy_result = "correction_disabled"
    elif selected and correction_enabled:
        expected_policy_result = "corrections_not_required"
    for module, correction in corrections_by_module.items():
        pre_sigma = pre_metrics[module]
        post_sigma = post_metrics[module]
        target = baseline[module] * correction_cap_ratio
        should_mutate = correction_enabled and pre_sigma > target
        expected_attempted = correction_enabled
        expected_operation = "relative_spectral_cap" if correction_enabled else "none"
        expected_outcome = (
            "applied_and_remeasured"
            if should_mutate
            else (
                "no_mutation_required"
                if correction_enabled
                else "not_attempted_policy_disabled"
            )
        )
        if correction.get("attempted") is not expected_attempted:
            errors.append(
                f"{source}.correction_ledger correction for {module!r} has invalid attempted state."
            )
        if correction.get("operation") != expected_operation:
            errors.append(
                f"{source}.correction_ledger correction for {module!r} has invalid operation."
            )
        if correction.get("mutation_applied") is not should_mutate:
            errors.append(
                f"{source}.correction_ledger correction for {module!r} has forged mutation state."
            )
        if correction.get("outcome") != expected_outcome:
            errors.append(
                f"{source}.correction_ledger correction for {module!r} has invalid outcome."
            )
        if expected_attempted:
            attempted_count += 1
        observed_pre = _finite(correction.get("pre_sigma"))
        observed_base = _finite(correction.get("baseline_sigma"))
        observed_post = _finite(correction.get("post_sigma"))
        scale = _finite(correction.get("scale_factor"))
        pre_digest = correction.get("pre_weight_digest")
        post_digest = correction.get("post_weight_digest")
        if not _digest_is_valid(pre_digest):
            errors.append(
                f"{source}.correction_ledger correction pre_weight_digest is invalid."
            )
        if not _digest_is_valid(post_digest):
            errors.append(
                f"{source}.correction_ledger correction post_weight_digest is invalid."
            )
        if observed_pre is None or not _close(observed_pre, pre_sigma):
            errors.append(
                f"{source}.correction_ledger correction pre_sigma is inconsistent."
            )
        if observed_base is None or not _close(observed_base, baseline[module]):
            errors.append(
                f"{source}.correction_ledger correction baseline_sigma is inconsistent."
            )
        if observed_post is None or not _close(observed_post, post_sigma):
            errors.append(
                f"{source}.correction_ledger correction post_sigma is inconsistent."
            )
        if should_mutate:
            applied_count += 1
            expected_policy_result = "corrections_applied"
            expected_scale = target / pre_sigma
            if scale is None or not _close(scale, expected_scale):
                errors.append(
                    f"{source}.correction_ledger correction scale_factor is inconsistent."
                )
            if not _measurement_close(post_sigma, pre_sigma * expected_scale):
                errors.append(
                    f"{source}.correction_ledger post-correction remeasurement "
                    f"for {module!r} does not prove the recorded mutation."
                )
            if pre_digest == post_digest:
                errors.append(
                    f"{source}.correction_ledger mutation for {module!r} "
                    "did not change the weight digest."
                )
        else:
            if (
                scale is None
                or not _close(scale, 1.0)
                or not _close(post_sigma, pre_sigma)
            ):
                errors.append(
                    f"{source}.correction_ledger no-mutation correction for {module!r} "
                    "changed the retained measurement."
                )
            if pre_digest != post_digest:
                errors.append(
                    f"{source}.correction_ledger no-mutation correction for {module!r} "
                    "changed the weight digest."
                )
    return applied_count, attempted_count, expected_policy_result


def replay_correction_ledger(
    errors: list[str],
    *,
    entry: Mapping[str, Any],
    source: str,
    metrics: Mapping[str, Any],
    baseline: Mapping[str, float],
    final: Mapping[str, float],
    families: Mapping[str, str],
    family_stats: Mapping[str, Mapping[str, float | int]],
    family_caps: Mapping[str, float],
    deadband: float,
    max_norm: float | None,
    method: str,
    alpha: float,
    configured_m: int,
    degeneracy_enabled: bool,
    baseline_degeneracy: Mapping[str, Mapping[str, float]],
    thresholds: Mapping[str, tuple[float, float]],
    correction_enabled: bool,
    correction_cap_ratio: float,
    final_caps_applied: int,
    final_caps_exceeded: bool,
) -> None:
    ledger = _mapping(entry.get("correction_ledger"))
    if ledger is None:
        errors.append(f"{source}.correction_ledger must be an object.")
        return
    unknown_ledger_fields = sorted(set(ledger) - _LEDGER_FIELDS)
    if unknown_ledger_fields:
        errors.append(
            f"{source}.correction_ledger contains unsupported fields: "
            f"{unknown_ledger_fields}."
        )
    if ledger.get("schema_version") != 1:
        errors.append(f"{source}.correction_ledger.schema_version must be 1.")
    phase = ledger.get("phase")
    if not isinstance(phase, str) or not phase:
        errors.append(f"{source}.correction_ledger.phase must be a non-empty string.")
    if ledger.get("correction_enabled") is not correction_enabled:
        errors.append(
            f"{source}.correction_ledger.correction_enabled disagrees with policy."
        )
    ledger_ratio = _finite(ledger.get("correction_cap_ratio"))
    if ledger_ratio is None or not _close(ledger_ratio, correction_cap_ratio):
        errors.append(
            f"{source}.correction_ledger.correction_cap_ratio disagrees with policy."
        )
    pre_metrics = _numeric_map(
        errors,
        ledger.get("pre_correction_metrics"),
        f"{source}.correction_ledger.pre_correction_metrics",
        nonnegative=True,
    )
    post_metrics = _numeric_map(
        errors,
        ledger.get("post_correction_metrics"),
        f"{source}.correction_ledger.post_correction_metrics",
        nonnegative=True,
    )
    pre_z_scores = _numeric_map(
        errors,
        ledger.get("pre_correction_z_scores"),
        f"{source}.correction_ledger.pre_correction_z_scores",
    )
    if pre_metrics is None or post_metrics is None or pre_z_scores is None:
        return
    modules = set(baseline)
    if (
        set(pre_metrics) != modules
        or set(post_metrics) != modules
        or set(pre_z_scores) != modules
    ):
        errors.append(
            f"{source}.correction_ledger pre/post/z module inventories must match baseline."
        )
        return
    for module in sorted(modules):
        if not _close(post_metrics[module], final[module]):
            errors.append(
                f"{source}.correction_ledger.post_correction_metrics.{module} "
                "disagrees with final_metrics."
            )
    if degeneracy_enabled:
        pre_degeneracy = _degeneracy_map(
            errors,
            ledger.get("pre_correction_degeneracy"),
            f"{source}.correction_ledger.pre_correction_degeneracy",
            modules,
        )
        if pre_degeneracy is None:
            return
    else:
        raw_pre_degeneracy = ledger.get("pre_correction_degeneracy")
        if not isinstance(raw_pre_degeneracy, Mapping):
            errors.append(
                f"{source}.correction_ledger.pre_correction_degeneracy must be an object."
            )
            return
        pre_degeneracy = {}
    expected_z, _budgeted, _fatal, selection, selected = replay_selected_findings(
        baseline=baseline,
        current=pre_metrics,
        families=families,
        family_stats=family_stats,
        family_caps=family_caps,
        deadband=deadband,
        max_norm=max_norm,
        method=method,
        alpha=alpha,
        configured_m=configured_m,
        degeneracy_enabled=degeneracy_enabled,
        baseline_degeneracy=baseline_degeneracy,
        current_degeneracy=pre_degeneracy,
        thresholds=thresholds,
    )
    for module, expected in expected_z.items():
        if not _close(pre_z_scores[module], expected):
            errors.append(
                f"{source}.correction_ledger.pre_correction_z_scores.{module} "
                "disagrees with replayed pre-correction measurements."
            )
    _compare_tree(
        errors,
        ledger.get("multiple_testing_selection"),
        selection,
        f"{source}.correction_ledger.multiple_testing_selection",
    )
    observed_findings = ledger.get("selected_findings")
    _compare_violations(
        errors,
        observed_findings,
        selected,
        f"{source}.correction_ledger.selected_findings",
    )
    if not isinstance(observed_findings, list):
        return
    finding_ids, finding_ids_by_module = _finding_bindings(
        errors, observed_findings, source
    )

    corrections_raw = ledger.get("corrections")
    if not isinstance(corrections_raw, list):
        errors.append(f"{source}.correction_ledger.corrections must be an array.")
        return
    corrections_by_module = _correction_bindings(
        errors,
        corrections_raw,
        source,
        finding_ids,
        finding_ids_by_module,
    )
    applied_count, attempted_count, expected_policy_result = _replay_correction_entries(
        errors,
        source=source,
        corrections_by_module=corrections_by_module,
        selected=selected,
        baseline=baseline,
        pre_metrics=pre_metrics,
        post_metrics=post_metrics,
        correction_enabled=correction_enabled,
        correction_cap_ratio=correction_cap_ratio,
    )
    if ledger.get("policy_result") != expected_policy_result:
        errors.append(
            f"{source}.correction_ledger.policy_result disagrees with replayed corrections."
        )
    counter_expectations: dict[str, object] = {
        "selected_budgeted_findings": final_caps_applied,
        "cap_budget_exceeded": final_caps_exceeded,
        "corrections_attempted": attempted_count,
        "corrections_applied": applied_count,
        "correction_policy_result": expected_policy_result,
    }
    for field, counter_expected in counter_expectations.items():
        if metrics.get(field) != counter_expected:
            errors.append(f"{source}.metrics.{field} disagrees with correction replay.")


__all__ = ["replay_correction_ledger"]
