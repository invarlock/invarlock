from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from invarlock.core.auto_tuning import get_tier_policies
from invarlock.reporting.certificate import make_certificate

console = Console()


def explain_gates_command(
    report: str = typer.Option(..., "--report", help="Path to primary report.json"),
    baseline: str = typer.Option(
        ..., "--baseline", help="Path to baseline report.json"
    ),
) -> None:
    """Explain certificate gates for a report vs baseline.

    Loads the reports, builds a certificate, and prints gate thresholds,
    observed statistics, and pass/fail reasons in a compact, readable form.
    """
    report_path = Path(report)
    baseline_path = Path(baseline)
    if not report_path.exists() or not baseline_path.exists():
        console.print("[red]Missing --report or --baseline file[/red]")
        raise typer.Exit(1)

    try:
        report_data = json.loads(report_path.read_text())
        baseline_data = json.loads(baseline_path.read_text())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to load inputs: {exc}[/red]")
        raise typer.Exit(1) from exc

    cert = make_certificate(report_data, baseline_data)
    validation = (
        cert.get("validation", {}) if isinstance(cert.get("validation"), dict) else {}
    )

    # Extract tier + metric policy (floors/hysteresis)
    tier = str((cert.get("auto", {}) or {}).get("tier", "balanced")).lower()
    tier_thresholds = {
        "conservative": 1.05,
        "balanced": 1.10,
        "aggressive": 1.20,
        "none": 1.10,
    }
    resolved_policy = (
        cert.get("resolved_policy", {})
        if isinstance(cert.get("resolved_policy"), dict)
        else {}
    )
    metrics_policy = (
        resolved_policy.get("metrics", {})
        if isinstance(resolved_policy.get("metrics"), dict)
        else {}
    )
    if not metrics_policy:
        tier_policies = get_tier_policies()
        tier_defaults = tier_policies.get(tier, tier_policies.get("balanced", {}))
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
    try:
        limit_base = float(
            pm_policy.get("ratio_limit_base", tier_thresholds.get(tier, 1.10))
            or tier_thresholds.get(tier, 1.10)
        )
    except Exception:
        limit_base = tier_thresholds.get(tier, 1.10)
    limit_with_hyst = limit_base + max(0.0, hysteresis_ratio)
    tokens_ok = True
    telem = cert.get("telemetry", {}) if isinstance(cert.get("telemetry"), dict) else {}
    try:
        total_tokens = int(telem.get("preview_total_tokens", 0)) + int(
            telem.get("final_total_tokens", 0)
        )
        tokens_ok = (min_tokens == 0) or (total_tokens >= min_tokens)
    except Exception:
        tokens_ok = True

    # Primary-metric ratio gate explanation (ppl-like kinds shown as ratios)
    ratio = None
    ratio_ci = None
    if isinstance(cert.get("primary_metric"), dict):
        pm = cert.get("primary_metric", {})
        ratio = pm.get("ratio_vs_baseline")
        ratio_ci = pm.get("display_ci")
    elif isinstance(cert.get("ppl"), dict):  # legacy
        ppl = cert.get("ppl", {})
        ratio = ppl.get("ratio_vs_baseline")
        ratio_ci = ppl.get("ratio_ci")
    hysteresis_applied = bool(validation.get("hysteresis_applied"))
    status = "PASS" if bool(validation.get("primary_metric_acceptable")) else "FAIL"
    console.print("[bold]Gate: Primary Metric vs Baseline[/bold]")
    console.print(f"  status: {status}")
    if isinstance(ratio, int | float):
        if isinstance(ratio_ci, tuple | list) and len(ratio_ci) == 2:
            console.print(
                f"  observed: {ratio:.3f}x (CI {ratio_ci[0]:.3f}-{ratio_ci[1]:.3f})"
            )
        else:
            console.print(f"  observed: {ratio:.3f}x")
    console.print(
        f"  threshold: ≤ {limit_base:.2f}x{(f' (+hysteresis {hysteresis_ratio:.3f})' if hysteresis_ratio else '')}"
    )
    console.print(
        f"  tokens: {'ok' if tokens_ok else 'below floor'} (token floors: min_tokens={min_tokens or 0}, total={int(telem.get('preview_total_tokens', 0)) + int(telem.get('final_total_tokens', 0)) if telem else 0})"
    )
    if hysteresis_applied:
        console.print(
            f"  note: hysteresis applied → effective threshold = {limit_with_hyst:.3f}x"
        )

    # Dataset split visibility from report provenance
    try:
        split = (report_data.get("provenance", {}) or {}).get("dataset_split")
        sf = (report_data.get("provenance", {}) or {}).get("split_fallback")
        if split:
            line = f"Dataset split: {split}"
            if sf:
                line += " (fallback)"
            # Click echo would be ideal, but we keep consistent console printing
            console.print(line)
    except Exception:
        pass

    # Drift gate explanation
    drift = None
    drift_ci = None
    if isinstance(cert.get("primary_metric"), dict):
        pm = cert.get("primary_metric", {})
        preview = pm.get("preview")
        final = pm.get("final")
        if isinstance(preview, int | float) and isinstance(final, int | float):
            try:
                if float(preview) != 0.0:
                    drift = float(final) / float(preview)
            except Exception:
                drift = None
    if isinstance(cert.get("ppl"), dict):  # legacy
        ppl = cert.get("ppl", {})
        drift = ppl.get("preview_final_ratio", drift)
        drift_ci = ppl.get("drift_ci")
    drift_status = (
        "PASS" if bool(validation.get("preview_final_drift_acceptable")) else "FAIL"
    )
    console.print("\n[bold]Gate: Drift (final/preview)[/bold]")
    if isinstance(drift, int | float):
        if isinstance(drift_ci, tuple | list) and len(drift_ci) == 2:
            console.print(
                f"  observed: {drift:.3f} (CI {drift_ci[0]:.3f}-{drift_ci[1]:.3f})"
            )
        else:
            console.print(f"  observed: {drift:.3f}")
    console.print("  threshold: 0.95-1.05")
    console.print(f"  status: {drift_status}")

    # Guard Overhead explanation (if present)
    overhead = (
        cert.get("guard_overhead", {})
        if isinstance(cert.get("guard_overhead"), dict)
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
