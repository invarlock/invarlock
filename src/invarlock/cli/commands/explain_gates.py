from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import typer
from rich.console import Console

from invarlock.core.auto_tuning import get_tier_policies
from invarlock.reporting.report_builder_support import (
    telemetry_output_enabled,
    telemetry_summary_line,
)
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_outline import build_evaluation_report_outline
from invarlock.reporting.report_summary import build_quality_gates_summary

console = Console()


def _load_json_payload(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping_dict(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_optional_float(value: object) -> float | None:
    try:
        coerced = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(coerced):
        return None
    return coerced


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


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


def _print_report_outline_summary(evaluation_report: dict[str, object]) -> None:
    outline = build_evaluation_report_outline(evaluation_report)
    console.print("[bold]Report Outline[/bold]")
    for section in outline.sections:
        if section.priority not in {"summary", "review", "audit"}:
            continue
        console.print(
            f"  {section.title}: {section.summary}",
            markup=False,
        )
        for fact in section.facts:
            status = f" [{fact.status}]" if fact.status else ""
            source = fact.source or "-"
            console.print(
                f"    - {fact.label}: {fact.value}{status}; source={source}",
                markup=False,
            )
    console.print("")


def explain_evaluation_report(
    evaluation_report: dict[str, object],
    *,
    report_payload: object | None = None,
) -> None:
    """Explain gate decisions from an already-built evaluation report."""
    if telemetry_output_enabled():
        summary_line = telemetry_summary_line(evaluation_report)
        if summary_line:
            console.print(summary_line, markup=False)
    _print_report_outline_summary(evaluation_report)
    validation = (
        evaluation_report.get("validation", {})
        if isinstance(evaluation_report.get("validation"), dict)
        else {}
    )
    quality_gates = build_quality_gates_summary(evaluation_report)
    quality_by_label = {row.label: row for row in quality_gates.rows}
    auto = (
        evaluation_report.get("auto", {})
        if isinstance(evaluation_report.get("auto"), dict)
        else {}
    )
    tiny_relax = bool(auto.get("tiny_relax"))

    # Extract tier + metric policy (floors/hysteresis)
    tier = str(auto.get("tier", "balanced")).lower()
    effective_tier = "aggressive" if tiny_relax else tier
    tier_policies = get_tier_policies()
    tier_defaults = tier_policies.get(effective_tier, tier_policies.get("balanced", {}))
    resolved_policy = (
        evaluation_report.get("resolved_policy", {})
        if isinstance(evaluation_report.get("resolved_policy"), dict)
        else {}
    )
    metrics_policy: dict[str, object] = {}
    if not tiny_relax:
        metrics_policy = (
            resolved_policy.get("metrics", {})
            if isinstance(resolved_policy.get("metrics"), dict)
            else {}
        )
    if not metrics_policy:
        metrics_policy = (
            tier_defaults.get("metrics", {}) if isinstance(tier_defaults, dict) else {}
        )
        if not isinstance(metrics_policy, dict):
            metrics_policy = {}
    pm_policy = (
        metrics_policy.get("pm_ratio", {})
        if isinstance(metrics_policy.get("pm_ratio"), dict)
        else {}
    )
    hysteresis_ratio = float(pm_policy.get("hysteresis_ratio", 0.0))
    min_tokens = int(pm_policy.get("min_tokens", 0))
    limit_base = _coerce_optional_float(pm_policy.get("ratio_limit_base"))
    if limit_base is None:
        fallback = (
            tier_defaults.get("metrics", {}) if isinstance(tier_defaults, dict) else {}
        )
        fallback_pm = fallback.get("pm_ratio", {}) if isinstance(fallback, dict) else {}
        limit_base = _coerce_optional_float(fallback_pm.get("ratio_limit_base"))
    limit_with_hyst = (
        float(limit_base) + max(0.0, hysteresis_ratio)
        if isinstance(limit_base, int | float)
        else None
    )
    telem = _mapping_dict(evaluation_report.get("telemetry"))
    total_tokens = _coerce_int(telem.get("preview_total_tokens")) + _coerce_int(
        telem.get("final_total_tokens")
    )
    tokens_ok = (min_tokens == 0) or (total_tokens >= min_tokens) or tiny_relax

    # Primary-metric gate explanation. PPL-like metrics use ratios; accuracy uses
    # baseline-relative percentage-point deltas.
    ratio_ci = None
    pm: dict[str, object] = {}
    if isinstance(evaluation_report.get("primary_metric"), dict):
        pm = evaluation_report.get("primary_metric", {})
        ratio_ci = pm.get("display_ci")
    hysteresis_applied = bool(validation.get("hysteresis_applied"))
    status = "PASS" if bool(validation.get("primary_metric_acceptable")) else "FAIL"
    primary_row = quality_by_label.get("Primary Metric Acceptable")
    console.print("[bold]Gate: Primary Metric vs Baseline[/bold]")
    console.print(f"  status: {status}")
    if primary_row is not None:
        console.print(f"  observed: {primary_row.measured}")
    else:
        ratio = pm.get("ratio_vs_baseline")
        if (
            isinstance(ratio, int | float)
            and isinstance(ratio_ci, tuple | list)
            and len(ratio_ci) == 2
        ):
            console.print(
                f"  observed: {ratio:.3f}x (CI {ratio_ci[0]:.3f}-{ratio_ci[1]:.3f})"
            )
        elif isinstance(ratio, int | float):
            console.print(f"  observed: {ratio:.3f}x")
    if primary_row is not None:
        console.print(f"  threshold: {primary_row.threshold}")
    elif isinstance(limit_base, int | float):
        hyst_suffix = (
            f" (+hysteresis {hysteresis_ratio:.3f})" if hysteresis_ratio else ""
        )
        console.print(f"  threshold: ≤ {float(limit_base):.2f}x{hyst_suffix}")
    else:
        console.print("  threshold: unavailable")
    if tiny_relax:
        console.print(
            "  note: tiny relax enabled; aggressive-tier gates and token floors are informational"
        )
    token_state = "ok" if tokens_ok else "below floor"
    console.print(
        f"  tokens: {token_state} (token floors: min_tokens={min_tokens or 0}, total={total_tokens})"
    )
    if hysteresis_applied:
        if isinstance(limit_with_hyst, int | float):
            console.print(
                f"  note: hysteresis applied → effective threshold = {float(limit_with_hyst):.3f}x"
            )

    # Tail gate explanation (warn/fail; based on per-window Δlog-loss vs baseline)
    pm_tail = (
        evaluation_report.get("primary_metric_tail", {})
        if isinstance(evaluation_report.get("primary_metric_tail"), dict)
        else {}
    )
    if pm_tail:
        mode = str(pm_tail.get("mode", "warn") or "warn").strip().lower()
        evaluated = bool(pm_tail.get("evaluated", False))
        passed = bool(pm_tail.get("passed", True))
        policy = (
            pm_tail.get("policy", {}) if isinstance(pm_tail.get("policy"), dict) else {}
        )
        stats = (
            pm_tail.get("stats", {}) if isinstance(pm_tail.get("stats"), dict) else {}
        )

        q = policy.get("quantile", 0.95)
        qf = _coerce_optional_float(q)
        if qf is None:
            qf = 0.95
        qf = max(0.0, min(1.0, qf))
        q_key = f"q{int(round(100.0 * qf))}"
        q_name = f"P{int(round(100.0 * qf))}"
        q_val = stats.get(q_key)
        qmax = policy.get("quantile_max")
        eps = policy.get("epsilon", stats.get("epsilon"))
        mass = stats.get("tail_mass")
        mmax = policy.get("mass_max")

        if not evaluated:
            status_tail = "INFO"
        elif passed:
            status_tail = "PASS"
        elif mode == "fail":
            status_tail = "FAIL"
        else:
            status_tail = "WARN"

        console.print("\n[bold]Gate: Primary Metric Tail (ΔlogNLL)[/bold]")
        console.print(f"  mode: {mode}")
        console.print(f"  status: {status_tail}")
        if isinstance(q_val, int | float):
            console.print(f"  observed: {q_name}={float(q_val):.4f}")
        if isinstance(mass, int | float):
            console.print(f"  tail_mass: Pr[ΔlogNLL > ε]={float(mass):.4f}")
        thr_parts: list[str] = []
        if isinstance(qmax, int | float):
            thr_parts.append(f"{q_name}≤{float(qmax):.4f}")
        if isinstance(mmax, int | float):
            thr_parts.append(f"mass≤{float(mmax):.4f}")
        if isinstance(eps, int | float):
            thr_parts.append(f"ε={float(eps):.1e}")
        if thr_parts:
            console.print("  threshold: " + "; ".join(thr_parts))

    # Dataset split visibility from report provenance
    split_source = report_payload if report_payload is not None else evaluation_report
    split_line = _dataset_split_line(split_source)
    if split_line:
        console.print(split_line)

    # Drift gate explanation.
    drift_status = (
        "PASS" if bool(validation.get("preview_final_drift_acceptable")) else "FAIL"
    )
    kind = str(pm.get("kind", "") or "").lower()
    drift_row = quality_by_label.get("Preview Final Drift Acceptable")
    if kind.startswith("ppl") or kind == "accuracy":
        console.print("\n[bold]Gate: Drift (final/preview)[/bold]")
        if drift_row is not None:
            console.print(f"  observed: {drift_row.measured}")
            console.print(f"  threshold: {drift_row.threshold}")
            console.print(f"  basis: {drift_row.basis}")
        elif kind.startswith("ppl"):
            preview = pm.get("preview")
            final = pm.get("final")
            drift = _drift_ratio(preview, final)
            if isinstance(drift, int | float):
                console.print(f"  observed: {drift:.3f}")
            console.print("  threshold: unavailable")
        console.print(f"  status: {drift_status}")

    spectral = (
        evaluation_report.get("spectral", {})
        if isinstance(evaluation_report.get("spectral"), dict)
        else {}
    )
    if spectral:
        spectral_status = "PASS" if bool(validation.get("spectral_stable")) else "FAIL"
        caps_applied = spectral.get("caps_applied")
        max_caps = spectral.get("max_caps")
        console.print("\n[bold]Gate: Spectral Guard[/bold]")
        if isinstance(caps_applied, int | float):
            console.print(f"  observed: {int(caps_applied)} caps applied")
        else:
            console.print("  observed: caps not recorded")
        if isinstance(max_caps, int | float):
            console.print(f"  threshold: <= {int(max_caps)} caps")
        else:
            console.print("  threshold: resolved tier max_caps")
        console.print(f"  status: {spectral_status}")
        console.print(
            "  note: budgeted caps are guard observations; they are hard failures only when the policy budget is exceeded."
        )

    rmt = (
        evaluation_report.get("rmt", {})
        if isinstance(evaluation_report.get("rmt"), dict)
        else {}
    )
    if rmt:
        rmt_status = "PASS" if bool(validation.get("rmt_stable")) else "FAIL"
        epsilon_violations = rmt.get("epsilon_violations")
        console.print("\n[bold]Gate: RMT Guard[/bold]")
        if isinstance(epsilon_violations, list):
            console.print(f"  observed: {len(epsilon_violations)} epsilon violations")
        elif rmt.get("status"):
            console.print(f"  observed: {rmt.get('status')}")
        else:
            console.print("  observed: N/A")
        console.print("  threshold: ε-rule")
        console.print(f"  status: {rmt_status}")

    guard_warnings = (
        evaluation_report.get("guard_warnings", {})
        if isinstance(evaluation_report.get("guard_warnings"), dict)
        else {}
    )
    warnings = guard_warnings.get("warnings")
    if isinstance(warnings, list) and warnings:
        console.print("\n[bold]Guard Warnings[/bold]")
        console.print(
            "  note: guard warnings are baseline-relative signal changes, not hard policy failures unless strict warning mode is enabled."
        )
        for entry in warnings[:5]:
            if not isinstance(entry, dict):
                continue
            guard_name = entry.get("guard", "guard")
            kind_name = entry.get("kind", "warning")
            module = entry.get("module")
            location = f" module={module}" if isinstance(module, str) and module else ""
            console.print(
                f"  - {guard_name}.{kind_name}{location}; policy: {entry.get('policy_gate', 'unknown')}"
            )

    # Guard Overhead explanation (if present)
    overhead = (
        evaluation_report.get("guard_overhead", {})
        if isinstance(evaluation_report.get("guard_overhead"), dict)
        else {}
    )
    if overhead:
        passed = bool(validation.get("guard_overhead_acceptable", True))
        threshold = overhead.get("threshold_percent")
        if not isinstance(threshold, int | float):
            threshold = float(overhead.get("overhead_threshold", 0.01)) * 100.0
        pct = overhead.get("overhead_percent")
        ratio = overhead.get("overhead_ratio")
        console.print("\n[bold]Gate: Guard Overhead[/bold]")
        if isinstance(pct, int | float):
            console.print(
                f"  observed: {pct:+.2f}%{f' ({ratio:.3f}x)' if isinstance(ratio, int | float) else ''}"
            )
        elif isinstance(ratio, int | float):
            console.print(f"  observed: {ratio:.3f}x")
        else:
            console.print("  observed: N/A")
        console.print(f"  threshold: ≤ +{float(threshold):.1f}%")
        console.print(f"  status: {'PASS' if passed else 'FAIL'}")


def explain_gates_command(
    subject_report: str = typer.Option(
        ...,
        "--subject-report",
        help="Path to the subject run report.json",
    ),
    baseline_report: str = typer.Option(
        ...,
        "--baseline-report",
        help="Path to the baseline run report.json",
    ),
) -> None:
    """Explain evaluation report gates for a report vs baseline.

    Loads the reports, builds an evaluation report, and prints gate thresholds,
    observed statistics, and pass/fail reasons in a compact, readable form.
    """
    report_path = Path(subject_report)
    baseline_path = Path(baseline_report)
    if not report_path.exists() or not baseline_path.exists():
        console.print("[red]Missing --subject-report or --baseline-report file[/red]")
        raise typer.Exit(1)

    try:
        report_data = _load_json_payload(report_path)
        baseline_data = _load_json_payload(baseline_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        console.print(f"[red]Failed to load inputs: {exc}[/red]")
        raise typer.Exit(1) from exc

    evaluation_report = make_report(report_data, baseline_data)
    explain_evaluation_report(evaluation_report, report_payload=report_data)
