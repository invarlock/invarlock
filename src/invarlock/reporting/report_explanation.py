from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from invarlock.core.auto_tuning import get_tier_policies as default_tier_policies

from .report_outline import build_evaluation_report_outline
from .report_summary import build_quality_gates_summary

TierPoliciesGetter = Callable[[], Mapping[str, object]]


@dataclass(frozen=True)
class EvaluationReportExplanation:
    """Shared text view for `report explain` and report-adjacent summaries."""

    lines: tuple[str, ...]


def _mapping_dict(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_optional_float(value: object) -> float | None:
    try:
        coerced = float(cast(str | bytes | bytearray | int | float, value))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(coerced):
        return None
    return coerced


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        coerced = int(cast(str | bytes | bytearray | int | float, value))
    except (TypeError, ValueError, OverflowError):
        return default
    return coerced


def _dataset_split_line(report_payload: object) -> str | None:
    provenance = _mapping_dict(_mapping_dict(report_payload).get("provenance"))
    split = provenance.get("dataset_split")
    if not split:
        return None
    line = f"Dataset split: {split}"
    if provenance.get("split_fallback"):
        line += " (fallback)"
    return line


def _drift_ratio(preview: object, final: object) -> float | None:
    preview_value = _coerce_optional_float(preview)
    final_value = _coerce_optional_float(final)
    if preview_value is None or final_value is None or preview_value == 0.0:
        return None
    return final_value / preview_value


def _append_report_outline_summary(
    lines: list[str], evaluation_report: dict[str, object]
) -> None:
    outline = build_evaluation_report_outline(evaluation_report)
    lines.append("Report Outline")
    for section in outline.sections:
        if section.priority not in {"summary", "review", "audit"}:
            continue
        lines.append(f"  {section.title}: {section.summary}")
        for fact in section.facts:
            status = f" [{fact.status}]" if fact.status else ""
            source = fact.source or "-"
            lines.append(f"    - {fact.label}: {fact.value}{status}; source={source}")
    lines.append("")


def _append_primary_metric_tail_gate(
    lines: list[str], evaluation_report: dict[str, object]
) -> None:
    pm_tail = _mapping_dict(evaluation_report.get("primary_metric_tail"))
    if not pm_tail:
        return
    mode = str(pm_tail.get("mode", "warn") or "warn").strip().lower()
    evaluated = bool(pm_tail.get("evaluated", False))
    passed = pm_tail.get("passed") is True
    policy = _mapping_dict(pm_tail.get("policy"))
    stats = _mapping_dict(pm_tail.get("stats"))
    quantile = _coerce_optional_float(policy.get("quantile", 0.95))
    quantile = max(0.0, min(1.0, 0.95 if quantile is None else quantile))
    quantile_name = f"P{int(round(100.0 * quantile))}"
    quantile_value = stats.get(f"q{int(round(100.0 * quantile))}")
    mass = stats.get("tail_mass")
    status = (
        "INFO"
        if not evaluated
        else "PASS"
        if passed
        else "FAIL"
        if mode == "fail"
        else "WARN"
    )
    lines.extend(
        (
            "",
            "Gate: Primary Metric Tail (ΔlogNLL)",
            f"  mode: {mode}",
            f"  status: {status}",
        )
    )
    if isinstance(quantile_value, int | float):
        lines.append(f"  observed: {quantile_name}={float(quantile_value):.4f}")
    if isinstance(mass, int | float):
        lines.append(f"  tail_mass: Pr[ΔlogNLL > ε]={float(mass):.4f}")
    thresholds: list[str] = []
    for label, value in (
        (f"{quantile_name}≤", policy.get("quantile_max")),
        ("mass≤", policy.get("mass_max")),
        ("ε=", policy.get("epsilon", stats.get("epsilon"))),
    ):
        if isinstance(value, int | float):
            thresholds.append(
                f"{label}{float(value):.1e}"
                if label == "ε="
                else f"{label}{float(value):.4f}"
            )
    if thresholds:
        lines.append("  threshold: " + "; ".join(thresholds))


def _append_guard_warning_explanations(
    lines: list[str], evaluation_report: dict[str, object]
) -> None:
    warnings = _mapping_dict(evaluation_report.get("guard_warnings")).get("warnings")
    if not isinstance(warnings, list) or not warnings:
        return
    lines.extend(
        (
            "",
            "Guard Warnings",
            "  note: guard warnings are baseline-relative signal changes, not hard policy failures unless strict warning mode is enabled.",
        )
    )
    for entry in warnings[:5]:
        if not isinstance(entry, dict):
            continue
        module = entry.get("module")
        location = f" module={module}" if isinstance(module, str) and module else ""
        lines.append(
            f"  - {entry.get('guard', 'guard')}.{entry.get('kind', 'warning')}"
            f"{location}; policy: {entry.get('policy_gate', 'unknown')}"
        )


def _append_drift_gate(
    lines: list[str],
    *,
    validation: dict[str, object],
    quality_by_label: Mapping[str, Any],
    primary_metric: dict[str, object],
) -> None:
    kind = str(primary_metric.get("kind", "") or "").lower()
    if not (kind.startswith("ppl") or kind == "accuracy"):
        return
    lines.extend(("", "Gate: Drift (final/preview)"))
    drift_row = quality_by_label.get("Preview Final Drift Acceptable")
    if drift_row is not None:
        lines.append(f"  observed: {drift_row.measured}")
        lines.append(f"  threshold: {drift_row.threshold}")
        lines.append(f"  basis: {drift_row.basis}")
    elif kind.startswith("ppl"):
        drift = _drift_ratio(primary_metric.get("preview"), primary_metric.get("final"))
        if isinstance(drift, int | float):
            lines.append(f"  observed: {drift:.3f}")
        lines.append("  threshold: unavailable")
    status = (
        "PASS" if bool(validation.get("preview_final_drift_acceptable")) else "FAIL"
    )
    lines.append(f"  status: {status}")


def _append_primary_metric_gate(
    lines: list[str],
    *,
    evaluation_report: dict[str, object],
    validation: dict[str, object],
    quality_by_label: Mapping[str, Any],
    tier_policies_getter: TierPoliciesGetter,
) -> dict[str, object]:
    auto = _mapping_dict(evaluation_report.get("auto"))
    tiny_relax = bool(auto.get("tiny_relax"))
    tier = str(auto.get("tier", "balanced")).lower()
    effective_tier = "aggressive" if tiny_relax else tier
    tier_policies = tier_policies_getter()
    tier_defaults = _mapping_dict(
        tier_policies.get(effective_tier, tier_policies.get("balanced", {}))
    )
    resolved_policy = _mapping_dict(evaluation_report.get("resolved_policy"))
    metrics_policy: dict[str, object] = {}
    if not tiny_relax:
        metrics_policy = _mapping_dict(resolved_policy.get("metrics"))
    if not metrics_policy:
        metrics_policy = _mapping_dict(tier_defaults.get("metrics"))
    pm_policy = _mapping_dict(metrics_policy.get("pm_ratio"))
    hysteresis_ratio = _coerce_optional_float(pm_policy.get("hysteresis_ratio")) or 0.0
    min_tokens = _coerce_int(pm_policy.get("min_tokens"))
    limit_base = _coerce_optional_float(pm_policy.get("ratio_limit_base"))
    if limit_base is None:
        fallback = _mapping_dict(tier_defaults.get("metrics"))
        fallback_pm = _mapping_dict(fallback.get("pm_ratio"))
        limit_base = _coerce_optional_float(fallback_pm.get("ratio_limit_base"))
    limit_with_hyst = (
        float(limit_base) + max(0.0, hysteresis_ratio)
        if isinstance(limit_base, int | float)
        else None
    )
    telemetry = _mapping_dict(evaluation_report.get("telemetry"))
    total_tokens = _coerce_int(telemetry.get("preview_total_tokens")) + _coerce_int(
        telemetry.get("final_total_tokens")
    )
    tokens_ok = min_tokens == 0 or total_tokens >= min_tokens or tiny_relax

    primary_metric = _mapping_dict(evaluation_report.get("primary_metric"))
    ratio_ci = primary_metric.get("display_ci")
    hysteresis_applied = bool(validation.get("hysteresis_applied"))
    status = "PASS" if bool(validation.get("primary_metric_acceptable")) else "FAIL"
    primary_row = quality_by_label.get("Primary Metric Acceptable")
    lines.append("Gate: Primary Metric vs Baseline")
    lines.append(f"  status: {status}")
    if primary_row is not None:
        lines.append(f"  observed: {primary_row.measured}")
    else:
        kind = str(primary_metric.get("kind") or "").strip().lower()
        ratio = (
            primary_metric.get("ratio_vs_baseline") if kind.startswith("ppl") else None
        )
        delta_pp = (
            primary_metric.get("delta_vs_baseline_pp") if kind == "accuracy" else None
        )
        if (
            isinstance(ratio, int | float)
            and isinstance(ratio_ci, tuple | list)
            and len(ratio_ci) == 2
        ):
            lines.append(
                f"  observed: {ratio:.3f}x (CI {ratio_ci[0]:.3f}-{ratio_ci[1]:.3f})"
            )
        elif isinstance(ratio, int | float):
            lines.append(f"  observed: {ratio:.3f}x")
        elif isinstance(delta_pp, int | float):
            lines.append(f"  observed: {delta_pp:+.2f} pp")
    if primary_row is not None:
        lines.append(f"  threshold: {primary_row.threshold}")
    elif isinstance(limit_base, int | float):
        hyst_suffix = (
            f" (+hysteresis {hysteresis_ratio:.3f})" if hysteresis_ratio else ""
        )
        lines.append(f"  threshold: ≤ {float(limit_base):.2f}x{hyst_suffix}")
    else:
        lines.append("  threshold: unavailable")
    if tiny_relax:
        lines.append(
            "  note: tiny relax enabled; aggressive-tier gates and token floors are informational"
        )
    token_state = "ok" if tokens_ok else "below floor"
    lines.append(
        f"  tokens: {token_state} (token floors: min_tokens={min_tokens or 0}, total={total_tokens})"
    )
    if hysteresis_applied and isinstance(limit_with_hyst, int | float):
        lines.append(
            f"  note: hysteresis applied → effective threshold = {float(limit_with_hyst):.3f}x"
        )
    return primary_metric


def _append_spectral_gate(
    lines: list[str],
    *,
    evaluation_report: dict[str, object],
    validation: dict[str, object],
) -> None:
    spectral = _mapping_dict(evaluation_report.get("spectral"))
    if not spectral:
        return
    spectral_status = "PASS" if bool(validation.get("spectral_stable")) else "FAIL"
    caps_applied = spectral.get("caps_applied")
    max_caps = spectral.get("max_caps")
    lines.extend(("", "Gate: Spectral Guard"))
    if isinstance(caps_applied, int | float):
        lines.append(f"  observed: {int(caps_applied)} caps applied")
    else:
        lines.append("  observed: caps not recorded")
    if isinstance(max_caps, int | float):
        lines.append(f"  threshold: <= {int(max_caps)} caps")
    else:
        lines.append("  threshold: resolved tier max_caps")
    lines.append(f"  status: {spectral_status}")
    lines.append(
        "  note: budgeted caps are guard observations; they are hard failures only when the policy budget is exceeded."
    )


def _append_rmt_gate(
    lines: list[str],
    *,
    evaluation_report: dict[str, object],
    validation: dict[str, object],
) -> None:
    rmt = _mapping_dict(evaluation_report.get("rmt"))
    if not rmt:
        return
    rmt_status = "PASS" if bool(validation.get("rmt_stable")) else "FAIL"
    epsilon_violations = rmt.get("epsilon_violations")
    lines.extend(("", "Gate: RMT Guard"))
    if isinstance(epsilon_violations, list):
        lines.append(f"  observed: {len(epsilon_violations)} epsilon violations")
    elif rmt.get("status"):
        lines.append(f"  observed: {rmt.get('status')}")
    else:
        lines.append("  observed: N/A")
    lines.append("  threshold: ε-rule")
    lines.append(f"  status: {rmt_status}")


def _append_guard_metric_impact_gate(
    lines: list[str],
    *,
    evaluation_report: dict[str, object],
    validation: dict[str, object],
) -> None:
    metric_impact = _mapping_dict(evaluation_report.get("guard_metric_impact"))
    if not metric_impact:
        return
    evaluated = metric_impact.get("evaluated") is True
    passed = evaluated and validation.get("guard_metric_impact_acceptable") is True
    display_value = _coerce_optional_float(metric_impact.get("display_value"))
    degradation_limit = _coerce_optional_float(metric_impact.get("degradation_limit"))
    display_unit = metric_impact.get("display_unit")
    lines.extend(("", "Gate: Guard Metric Impact"))
    if not evaluated:
        lines.append("  observed: not evaluated")
    elif display_value is not None and display_unit == "percent":
        lines.append(f"  observed: {display_value:+.2f}%")
    elif display_value is not None and display_unit == "percentage_points":
        lines.append(f"  observed: {display_value:+.2f} pp")
    else:
        lines.append("  observed: N/A")
    if degradation_limit is not None and display_unit == "percent":
        lines.append(f"  threshold: ≤ +{degradation_limit * 100.0:.1f}%")
    elif degradation_limit is not None and display_unit == "percentage_points":
        lines.append(f"  threshold: ≤ +{degradation_limit * 100.0:.1f} pp")
    else:
        lines.append("  threshold: N/A")
    status = "PASS" if passed else "FAIL" if evaluated else "NOT EVALUATED"
    lines.append(f"  status: {status}")


def build_evaluation_report_explanation(
    evaluation_report: dict[str, object],
    *,
    report_payload: object | None = None,
    tier_policies_getter: TierPoliciesGetter = default_tier_policies,
) -> EvaluationReportExplanation:
    """Build the shared human explanation for an evaluation report.

    The outline owns the top-level report reading order, while this view expands
    individual gates into the concise threshold/observation wording used by the
    CLI and future report summaries.
    """

    lines: list[str] = []
    _append_report_outline_summary(lines, evaluation_report)

    validation = _mapping_dict(evaluation_report.get("validation"))
    quality_gates = build_quality_gates_summary(evaluation_report)
    quality_by_label = {row.label: row for row in quality_gates.rows}
    pm = _append_primary_metric_gate(
        lines,
        evaluation_report=evaluation_report,
        validation=validation,
        quality_by_label=quality_by_label,
        tier_policies_getter=tier_policies_getter,
    )

    _append_primary_metric_tail_gate(lines, evaluation_report)

    split_source = report_payload if report_payload is not None else evaluation_report
    split_line = _dataset_split_line(split_source)
    if split_line:
        lines.append(split_line)

    _append_drift_gate(
        lines,
        validation=validation,
        quality_by_label=quality_by_label,
        primary_metric=pm,
    )

    _append_spectral_gate(
        lines, evaluation_report=evaluation_report, validation=validation
    )
    _append_rmt_gate(lines, evaluation_report=evaluation_report, validation=validation)

    _append_guard_warning_explanations(lines, evaluation_report)

    _append_guard_metric_impact_gate(
        lines, evaluation_report=evaluation_report, validation=validation
    )

    return EvaluationReportExplanation(lines=tuple(lines))


def render_evaluation_report_explanation_lines(
    evaluation_report: dict[str, object],
    *,
    report_payload: object | None = None,
    tier_policies_getter: TierPoliciesGetter = default_tier_policies,
) -> list[str]:
    return list(
        build_evaluation_report_explanation(
            evaluation_report,
            report_payload=report_payload,
            tier_policies_getter=tier_policies_getter,
        ).lines
    )
