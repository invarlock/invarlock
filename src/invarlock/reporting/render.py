from __future__ import annotations

import json
import math
from pathlib import Path

# mypy: ignore-errors
from typing import Any

import yaml

# Import certificate module for helper access without creating hard cycles
from . import certificate as C

# Console Validation Block helpers (allow-list driven)
_CONSOLE_LABELS_DEFAULT = [
    "Primary Metric Acceptable",
    "Preview Final Drift Acceptable",
    "Guard Overhead Acceptable",
    "Invariants Pass",
    "Spectral Stable",
    "Rmt Stable",
]


def _load_console_labels() -> list[str]:
    """Load console labels allow-list from contracts with a safe fallback."""
    try:
        root = Path(__file__).resolve().parents[3]
        path = root / "contracts" / "console_labels.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return list(data)
    except Exception:
        pass
    return list(_CONSOLE_LABELS_DEFAULT)


def compute_console_validation_block(certificate: dict[str, Any]) -> dict[str, Any]:
    """Produce a normalized console validation block from a certificate.

    Returns a dict with keys:
    - labels: the canonical label list
    - rows: list of {label, status, evaluated, ok}
    - overall_pass: boolean computed from canonical rows only. Guard Overhead is
      counted only when evaluated.
    """
    labels = _load_console_labels()
    validation = certificate.get("validation", {}) or {}
    guard_ctx = certificate.get("guard_overhead", {}) or {}
    guard_evaluated = (
        bool(guard_ctx.get("evaluated")) if isinstance(guard_ctx, dict) else False
    )

    # Map label → validation key
    def _to_key(label: str) -> str:
        return label.strip().lower().replace(" ", "_")

    rows: list[dict[str, Any]] = []
    ok_map: dict[str, bool] = {}
    effective_labels: list[str] = []
    for label in labels:
        key = _to_key(label)
        ok = bool(validation.get(key, False))
        status: str
        evaluated = True
        if key == "guard_overhead_acceptable":
            evaluated = guard_evaluated
            if not evaluated:
                # Omit row entirely when not evaluated (policy/profile skipped)
                continue
            status = "✅ PASS" if ok else "❌ FAIL"
        else:
            status = "✅ PASS" if ok else "❌ FAIL"
        rows.append(
            {"label": label, "status": status, "evaluated": evaluated, "ok": ok}
        )
        effective_labels.append(label)
        ok_map[key] = ok

    # Overall policy from canonical rows only; exclude guard when not evaluated
    keys_for_overall = [
        "primary_metric_acceptable",
        "preview_final_drift_acceptable",
        "invariants_pass",
        "spectral_stable",
        "rmt_stable",
    ]
    # Include guard overhead only if evaluated
    if guard_evaluated:
        keys_for_overall.append("guard_overhead_acceptable")

    overall_pass = all(ok_map.get(k, False) for k in keys_for_overall)

    return {"labels": effective_labels, "rows": rows, "overall_pass": overall_pass}


def _format_plugin(plugin: dict[str, Any]) -> str:
    """Format a plugin entry for markdown list rendering."""
    name = plugin.get("name", "unknown")
    version = plugin.get("version") or "-"
    module = plugin.get("module") or "unknown"
    entry = plugin.get("entry_point")
    pieces = [f"**{name}** v{version}", f"`{module}`"]
    if entry:
        pieces.append(f"[{entry}]")
    return " ".join(pieces)


def _short_digest(v: str) -> str:
    v = str(v)
    return v if len(v) <= 16 else (v[:8] + "…" + v[-8:])


def _fmt_by_kind(x: Any, k: str) -> str:
    try:
        xv = float(x)
    except Exception:
        return "N/A"
    k = str(k).lower()
    if k in {"accuracy", "vqa_accuracy"}:
        return f"{xv * 100.0:.1f}"
    if k.startswith("ppl"):
        return f"{xv:.3g}"
    return f"{xv:.3f}"


def _fmtv(key: str, v: Any) -> str:
    if not (isinstance(v, int | float) and math.isfinite(float(v))):
        return "-"
    if key.startswith("latency_ms_"):
        return f"{float(v):.0f}"
    if key.startswith("throughput_"):
        return f"{float(v):.1f}"
    return f"{float(v):.3f}"


def _p(x: Any) -> str:
    try:
        return f"{float(x) * 100.0:.1f}%"
    except Exception:
        return "N/A"


def _append_system_overhead_section(lines: list[str], sys_over: dict[str, Any]) -> None:
    """Append the System Overhead markdown section to lines given a payload."""
    if not (isinstance(sys_over, dict) and sys_over):
        return
    lines.append("## System Overhead")
    lines.append("")
    lines.append("| Metric | Baseline | Edited | Δ | Ratio |")
    lines.append("|--------|----------|--------|---|-------|")

    mapping = {
        "latency_ms_p50": "Latency p50 (ms)",
        "latency_ms_p95": "Latency p95 (ms)",
        "throughput_sps": "Throughput (samples/s)",
    }
    for key, label in mapping.items():
        ent = sys_over.get(key)
        if not isinstance(ent, dict):
            continue
        b_raw = ent.get("baseline")
        e_raw = ent.get("edited")
        # If both baseline and edited are missing or zero, present N/A to avoid implying measured zeros
        try:
            b_val = float(b_raw)
        except Exception:
            b_val = float("nan")
        try:
            e_val = float(e_raw)
        except Exception:
            e_val = float("nan")
        if (not math.isfinite(b_val) or b_val == 0.0) and (
            not math.isfinite(e_val) or e_val == 0.0
        ):
            b_str = e_str = d_str = r_str = "N/A"
        else:
            b_str = _fmtv(key, b_val)
            e_str = _fmtv(key, e_val)
            d = ent.get("delta")
            r = ent.get("ratio")
            d_str = _fmtv(key, d) if isinstance(d, int | float) else "-"
            r_str = _fmtv(key, r) if isinstance(r, int | float) else "-"
        lines.append(f"| {label} | {b_str} | {e_str} | {d_str} | {r_str} |")
    lines.append("")


def _append_accuracy_subgroups(lines: list[str], subgroups: dict[str, Any]) -> None:
    """Append the Accuracy Subgroups markdown table given a subgroups payload."""
    if not (isinstance(subgroups, dict) and subgroups):
        return
    lines.append("## Accuracy Subgroups (informational)")
    lines.append("")
    lines.append("| Group | n(prev) | n(final) | Acc(prev) | Acc(final) | Δpp |")
    lines.append("|-------|---------|----------|-----------|------------|-----|")
    for g, rec in subgroups.items():
        try:
            npv = int(rec.get("n_preview", 0))
        except Exception:
            npv = 0
        try:
            nfi = int(rec.get("n_final", 0))
        except Exception:
            nfi = 0
        dp = rec.get("delta_pp")
        try:
            dp_str = f"{float(dp):+.1f} pp"
        except Exception:
            dp_str = "N/A"
        lines.append(
            f"| {g} | {npv} | {nfi} | {_p(rec.get('preview'))} | {_p(rec.get('final'))} | {dp_str} |"
        )
    lines.append("")


def _compute_certificate_hash(certificate: dict[str, Any]) -> str:
    """Compute integrity hash for the certificate.

    Hash ignores the `artifacts` section for stability across saves.
    """
    # Create a copy without the artifacts section for stable hashing
    cert_copy = dict(certificate or {})
    cert_copy.pop("artifacts", None)

    # Sort keys for deterministic hashing
    cert_str = json.dumps(cert_copy, sort_keys=True)
    import hashlib as _hash

    return _hash.sha256(cert_str.encode()).hexdigest()[:16]


def build_console_summary_pack(certificate: dict[str, Any]) -> dict[str, Any]:
    """Build a small, reusable console summary pack from a certificate.

    Returns a dict with:
    - overall_pass: bool
    - overall_line: human-friendly overall status line
    - gate_lines: list of "<Label>: <Status>" strings for each evaluated gate
    - labels: the canonical label list used
    """
    block = compute_console_validation_block(certificate)
    overall_pass = bool(block.get("overall_pass"))
    emoji = "✅" if overall_pass else "❌"
    overall_line = f"Overall Status: {emoji} {'PASS' if overall_pass else 'FAIL'}"

    gate_lines: list[str] = []
    for row in block.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        label = row.get("label", "Gate")
        status = row.get("status", "")
        gate_lines.append(f"{label}: {status}")

    return {
        "overall_pass": overall_pass,
        "overall_line": overall_line,
        "gate_lines": gate_lines,
        "labels": block.get("labels", []),
    }


def render_certificate_markdown(certificate: dict[str, Any]) -> str:
    """
    Render a certificate as a formatted Markdown report with pretty tables.

    This implementation is moved from certificate.py to keep that module lean.
    To avoid circular import issues, we alias helpers from the certificate
    module inside the function body.
    """
    # Alias frequently used helpers locally to avoid editing the large body
    validate_certificate = C.validate_certificate

    if not validate_certificate(certificate):
        raise ValueError("Invalid certificate structure")

    lines = []
    edit_name = str(certificate.get("edit_name") or "").lower()

    # Header
    lines.append("# InvarLock Safety Certificate")
    lines.append("")
    lines.append(
        "> *Basis: “point” gates check the point estimate; “upper” gates check the CI "
        "upper bound; “point & upper” requires both to pass.*"
    )
    lines.append("")
    lines.append(f"**Schema Version:** {certificate['schema_version']}")
    lines.append(f"**Run ID:** `{certificate['run_id']}`")
    lines.append(f"**Generated:** {certificate['artifacts']['generated_at']}")
    lines.append(f"**Edit Type:** {certificate.get('edit_name', 'Unknown')}")
    lines.append("")

    plugins = certificate.get("plugins", {})
    if isinstance(plugins, dict) and plugins:
        lines.append("## Plugin Provenance")
        lines.append("")

        adapter_plugin = plugins.get("adapter")
        if isinstance(adapter_plugin, dict):
            lines.append(f"- Adapter: {_format_plugin(adapter_plugin)}")

        edit_plugin = plugins.get("edit")
        if isinstance(edit_plugin, dict):
            lines.append(f"- Edit: {_format_plugin(edit_plugin)}")

        guard_plugins = plugins.get("guards")
        if isinstance(guard_plugins, list) and guard_plugins:
            guard_entries = [
                _format_plugin(plugin)
                for plugin in guard_plugins
                if isinstance(plugin, dict)
            ]
            if guard_entries:
                lines.append("- Guards:\n  - " + "\n  - ".join(guard_entries))
        lines.append("")

    # Executive Summary with validation status (canonical, from console block)
    lines.append("## Executive Summary")
    lines.append("")
    _block = compute_console_validation_block(certificate)
    overall_pass = bool(_block.get("overall_pass"))
    status_emoji = "✅" if overall_pass else "❌"
    lines.append(
        f"**Overall Status:** {status_emoji} {'PASS' if overall_pass else 'FAIL'}"
    )
    # Window Plan one-liner for quick audit
    try:
        plan_ctx = (
            certificate.get("window_plan")
            or certificate.get("dataset", {}).get("windows", {})
            or certificate.get("ppl", {}).get("window_plan")
        )
        seq_len = certificate.get("dataset", {}).get("seq_len") or certificate.get(
            "dataset", {}
        ).get("sequence_length")
        if isinstance(plan_ctx, dict):
            profile = plan_ctx.get("profile")
            preview_n = (
                plan_ctx.get("preview_n")
                if plan_ctx.get("preview_n") is not None
                else plan_ctx.get("actual_preview")
            )
            final_n = (
                plan_ctx.get("final_n")
                if plan_ctx.get("final_n") is not None
                else plan_ctx.get("actual_final")
            )
            lines.append(
                f"- Window Plan: {profile}, {preview_n}/{final_n}{', seq_len=' + str(seq_len) if seq_len else ''}"
            )
    except Exception:
        pass
    lines.append("")

    # Validation table with canonical gates (mirrors console allow-list)
    lines.append("## Quality Gates")
    lines.append("")
    lines.append("| Gate | Status | Measured | Threshold | Basis | Description |")
    lines.append("|------|--------|----------|-----------|-------|-------------|")

    pm_block = certificate.get("primary_metric", {}) or {}
    has_pm = isinstance(pm_block, dict) and bool(pm_block)
    auto_info = certificate.get("auto", {})
    tier = (auto_info.get("tier") or "balanced").lower()

    # Helper to emit Primary Metric Acceptable row
    def _emit_pm_gate_row() -> None:
        pm_kind = str(pm_block.get("kind", "")).lower()
        value = pm_block.get("ratio_vs_baseline")
        gating_basis = pm_block.get("gating_basis") or "point"
        ok = bool(
            certificate.get("validation", {}).get("primary_metric_acceptable", True)
        )
        status = "✅ PASS" if ok else "❌ FAIL"
        if pm_kind in {"accuracy", "vqa_accuracy"}:
            measured = f"{value:+.2f} pp" if isinstance(value, int | float) else "N/A"
            th_map = {
                "conservative": -0.5,
                "balanced": -1.0,
                "aggressive": -2.0,
                "none": -1.0,
            }
            th = th_map.get(tier, -1.0)
            lines.append(
                f"| Primary Metric Acceptable | {status} | {measured} | ≥ {th:+.2f} pp | {gating_basis} | Δ accuracy vs baseline |"
            )
        else:
            tier_thresholds = {
                "conservative": 1.05,
                "balanced": 1.10,
                "aggressive": 1.20,
                "none": 1.10,
            }
            ratio_limit = tier_thresholds.get(tier, 1.10)
            target_ratio = auto_info.get("target_pm_ratio")
            if isinstance(target_ratio, int | float) and target_ratio > 0:
                ratio_limit = min(ratio_limit, float(target_ratio))
            measured = f"{value:.3f}x" if isinstance(value, int | float) else "N/A"
            lines.append(
                f"| Primary Metric Acceptable | {status} | {measured} | ≤ {ratio_limit:.2f}x | {gating_basis} | Ratio vs baseline |"
            )

    # Helper to emit Preview Final Drift Acceptable row
    def _emit_drift_gate_row() -> None:
        ok = bool(
            certificate.get("validation", {}).get(
                "preview_final_drift_acceptable", True
            )
        )
        status = "✅ PASS" if ok else "❌ FAIL"
        # Compute drift from PM preview/final when available
        try:
            pv = (
                float(pm_block.get("preview"))
                if isinstance(pm_block.get("preview"), int | float)
                else float("nan")
            )
            fv = (
                float(pm_block.get("final"))
                if isinstance(pm_block.get("final"), int | float)
                else float("nan")
            )
            drift = (
                fv / pv
                if (math.isfinite(pv) and pv > 0 and math.isfinite(fv))
                else float("nan")
            )
        except Exception:
            drift = float("nan")
        measured = f"{drift:.3f}x" if math.isfinite(drift) else "N/A"
        lines.append(
            f"| Preview Final Drift Acceptable | {status} | {measured} | 0.95–1.05x | point | Final/Preview ratio stability |"
        )

    # Helper to emit Guard Overhead Acceptable row (only when evaluated)
    def _emit_overhead_gate_row() -> None:
        guard_overhead = certificate.get("guard_overhead", {}) or {}
        evaluated = bool(guard_overhead.get("evaluated"))
        if not evaluated:
            return
        ok = bool(
            certificate.get("validation", {}).get("guard_overhead_acceptable", True)
        )
        status = "✅ PASS" if ok else "❌ FAIL"
        overhead_pct = guard_overhead.get("overhead_percent")
        overhead_ratio = guard_overhead.get("overhead_ratio")
        if isinstance(overhead_pct, int | float) and math.isfinite(float(overhead_pct)):
            measured = f"{float(overhead_pct):+.2f}%"
        elif isinstance(overhead_ratio, int | float) and math.isfinite(
            float(overhead_ratio)
        ):
            measured = f"{float(overhead_ratio):.3f}x"
        else:
            measured = "N/A"
        threshold_pct = guard_overhead.get("threshold_percent")
        if not (
            isinstance(threshold_pct, int | float)
            and math.isfinite(float(threshold_pct))
        ):
            threshold_val = guard_overhead.get("overhead_threshold", 0.01)
            try:
                threshold_pct = float(threshold_val) * 100.0
            except Exception:
                threshold_pct = 1.0
        lines.append(
            f"| Guard Overhead Acceptable | {status} | {measured} | ≤ +{threshold_pct:.1f}% | point | Guarded vs bare PM overhead |"
        )

    # Emit canonical gate rows
    if has_pm:
        _emit_pm_gate_row()
        _emit_drift_gate_row()
        _emit_overhead_gate_row()

    # Annotate hysteresis usage if applied
    if certificate.get("validation", {}).get("hysteresis_applied"):
        lines.append("- Note: hysteresis applied to gate boundary")

    lines.append("")
    lines.append("## Safety Check Details")
    lines.append("")
    lines.append("| Safety Check | Status | Measured | Threshold | Description |")
    lines.append("|--------------|--------|----------|-----------|-------------|")

    inv_summary = certificate["invariants"]
    validation = certificate.get("validation", {})
    inv_status = "✅ PASS" if validation.get("invariants_pass", False) else "❌ FAIL"
    inv_counts = inv_summary.get("summary", {}) or {}
    inv_measure = inv_summary.get("status", "pass").upper()
    fatal_violations = inv_counts.get("fatal_violations") or 0
    warning_violations = (
        inv_counts.get("warning_violations") or inv_counts.get("violations_found") or 0
    )
    if fatal_violations:
        suffix = f"{fatal_violations} fatal"
        if warning_violations:
            suffix += f", {warning_violations} warning"
        inv_measure = f"{inv_measure} ({suffix})"
    elif warning_violations:
        inv_measure = f"{inv_measure} ({warning_violations} warning)"
    lines.append(
        f"| Invariants | {inv_status} | {inv_measure} | pass | Model integrity checks |"
    )
    invariants_failures = inv_summary.get("failures") or []
    if warning_violations and not fatal_violations:
        non_fatal_message = None
        for failure in invariants_failures:
            if isinstance(failure, dict):
                msg = failure.get("message") or failure.get("type")
                if msg:
                    non_fatal_message = msg
                    break
        if not non_fatal_message:
            non_fatal_message = "Non-fatal invariant warnings present."
        lines.append(f"- Non-fatal: {non_fatal_message}")

    spec_status = "✅ PASS" if validation.get("spectral_stable", False) else "❌ FAIL"
    caps_applied = certificate["spectral"]["caps_applied"]
    lines.append(
        f"| Spectral Stability | {spec_status} | {caps_applied} violations | < 5 | Weight matrix spectral norms |"
    )

    # Catastrophic spike safety stop row is now driven by primary metric flags
    if isinstance(certificate.get("primary_metric"), dict):
        pm_ok = bool(validation.get("primary_metric_acceptable", True))
        pm_ratio = certificate.get("primary_metric", {}).get("ratio_vs_baseline")
        if isinstance(pm_ratio, int | float):
            lines.append(
                f"| Catastrophic Spike Gate (safety stop) | {'✅ PASS' if pm_ok else '❌ FAIL'} | {pm_ratio:.3f}x | ≤ 2.0x | Hard stop @ 2.0× |"
            )

    # Include RMT Health row for compatibility and clarity
    rmt_status = "✅ PASS" if validation.get("rmt_stable", False) else "❌ FAIL"
    rmt_state = certificate.get("rmt", {}).get("status", "unknown").title()
    lines.append(
        f"| RMT Health | {rmt_status} | {rmt_state} | ε-rule | Random Matrix Theory guard status |"
    )

    # Pairing + Bootstrap snapshot (quick audit surface)
    try:
        stats = (
            certificate.get("dataset", {}).get("windows", {}).get("stats", {})
            or certificate.get("ppl", {}).get("stats", {})
            or {}
        )
        paired_windows = stats.get("paired_windows")
        match_frac = stats.get("window_match_fraction")
        overlap_frac = stats.get("window_overlap_fraction")
        bootstrap = stats.get("bootstrap") or {}
        if (
            paired_windows is not None
            or match_frac is not None
            or overlap_frac is not None
        ):
            lines.append("")
            lines.append(
                f"- Pairing: paired={paired_windows}, match={match_frac:.3f}, overlap={overlap_frac:.3f}"
            )
        if isinstance(bootstrap, dict):
            reps = bootstrap.get("replicates")
            bseed = bootstrap.get("seed")
            if reps is not None or bseed is not None:
                lines.append(f"- Bootstrap: replicates={reps}, seed={bseed}")
        # Optional: show log-space paired Δ CI next to ratio CI for clarity
        delta_ci = certificate.get("primary_metric", {}).get("ci") or certificate.get(
            "ppl", {}
        ).get("logloss_delta_ci")
        if (
            isinstance(delta_ci, tuple | list)
            and len(delta_ci) == 2
            and all(isinstance(x, int | float) for x in delta_ci)
        ):
            lines.append(f"- Log Δ (paired) CI: [{delta_ci[0]:.6f}, {delta_ci[1]:.6f}]")
    except Exception:
        pass

    if invariants_failures:
        lines.append("")
        lines.append("**Invariant Notes**")
        lines.append("")
        for failure in invariants_failures:
            severity = failure.get("severity", "warning")
            detail = failure.get("detail", {})
            detail_str = ""
            if isinstance(detail, dict) and detail:
                detail_str = ", ".join(f"{k}={v}" for k, v in detail.items())
                detail_str = f" ({detail_str})"
            lines.append(
                f"- {failure.get('check', 'unknown')} [{severity}]: {failure.get('type', 'violation')}{detail_str}"
            )

    lines.append("")

    # Guard observability snapshots
    lines.append("## Guard Observability")
    lines.append("")

    spectral_info = certificate.get("spectral", {}) or {}
    if spectral_info:
        lines.append("### Spectral Guard")
        lines.append("")
        mt_info = spectral_info.get("multiple_testing", {}) or {}
        if mt_info:
            lines.append("- **Multiple Testing:**")
            lines.append("  ```yaml")
            mt_yaml = (
                yaml.safe_dump(mt_info, sort_keys=True, width=70).strip().splitlines()
            )
            for line in mt_yaml:
                lines.append(f"  {line}")
            lines.append("  ```")
        # Spectral summary (place key knobs together for quick scan)
        spec_sigma = spectral_info.get("sigma_quantile")
        spec_deadband = spectral_info.get("deadband")
        spec_max_caps = spectral_info.get("max_caps")
        summary_yaml = {
            "sigma_quantile": float(spec_sigma)
            if isinstance(spec_sigma, int | float)
            else None,
            "deadband": float(spec_deadband)
            if isinstance(spec_deadband, int | float)
            else None,
            "max_caps": int(spec_max_caps)
            if isinstance(spec_max_caps, int | float)
            else None,
        }
        # Drop Nones from summary
        summary_yaml = {k: v for k, v in summary_yaml.items() if v is not None}
        if summary_yaml:
            lines.append("- **Spectral Summary:**")
            lines.append("  ```yaml")
            for line in (
                yaml.safe_dump(summary_yaml, sort_keys=True, width=70)
                .strip()
                .splitlines()
            ):
                lines.append(f"  {line}")
            lines.append("  ```")
        lines.append(
            f"- Caps Applied: {spectral_info.get('caps_applied')} / {spectral_info.get('max_caps')}"
        )
        summary = spectral_info.get("summary", {}) or {}
        lines.append(f"- Caps Exceeded: {summary.get('caps_exceeded', False)}")
        caps_by_family = spectral_info.get("caps_applied_by_family") or {}
        family_caps = spectral_info.get("family_caps") or {}
        if caps_by_family:
            lines.append("")
            lines.append("| Family | κ | Violations |")
            lines.append("|--------|---|------------|")
            for family, count in caps_by_family.items():
                kappa = family_caps.get(family, {}).get("kappa")
                if isinstance(kappa, int | float) and math.isfinite(float(kappa)):
                    kappa_str = f"{kappa:.3f}"
                else:
                    kappa_str = "-"
                lines.append(f"| {family} | {kappa_str} | {count} |")
            lines.append("")
        quantiles = spectral_info.get("family_z_quantiles") or {}
        if quantiles:
            lines.append("| Family | q95 | q99 | Max | Samples |")
            lines.append("|--------|-----|-----|-----|---------|")
            for family, stats in quantiles.items():
                q95 = stats.get("q95")
                q99 = stats.get("q99")
                max_z = stats.get("max")
                count = stats.get("count")
                q95_str = f"{q95:.3f}" if isinstance(q95, int | float) else "-"
                q99_str = f"{q99:.3f}" if isinstance(q99, int | float) else "-"
                max_str = f"{max_z:.3f}" if isinstance(max_z, int | float) else "-"
                count_str = str(count) if isinstance(count, int | float) else "-"
                lines.append(
                    f"| {family} | {q95_str} | {q99_str} | {max_str} | {count_str} |"
                )
            lines.append("")
        policy_caps = spectral_info.get("policy", {}).get("family_caps")
        if policy_caps:
            lines.append("- **Family κ (policy):**")
            lines.append("  ```yaml")
            caps_yaml = (
                yaml.safe_dump(policy_caps, sort_keys=True, width=70)
                .strip()
                .splitlines()
            )
            for line in caps_yaml:
                lines.append(f"  {line}")
            lines.append("  ```")
        top_scores = spectral_info.get("top_z_scores") or {}
        if top_scores:
            lines.append("Top |z| per family:")
            for family in sorted(top_scores.keys()):
                entries = top_scores[family]
                if not entries:
                    continue
                formatted_entries = []
                for entry in entries:
                    module_name = entry.get("module", "unknown")
                    z_val = entry.get("z")
                    if isinstance(z_val, int | float) and math.isfinite(float(z_val)):
                        z_str = f"{z_val:.3f}"
                    else:
                        z_str = "n/a"
                    formatted_entries.append(f"{module_name} (|z|={z_str})")
                lines.append(f"- {family}: {', '.join(formatted_entries)}")
            lines.append("")

    rmt_info = certificate.get("rmt", {}) or {}
    if rmt_info:
        lines.append("### RMT Guard")
        lines.append("")
        families = rmt_info.get("families") or {}
        if families:
            lines.append("| Family | ε_f | Bare | Guarded | Δ |")
            lines.append("|--------|-----|------|---------|---|")
            for family, data in families.items():
                epsilon_val = data.get("epsilon")
                epsilon_str = (
                    f"{epsilon_val:.3f}"
                    if isinstance(epsilon_val, int | float)
                    else "-"
                )
                bare_count = data.get("bare", 0)
                guarded_count = data.get("guarded", 0)
                delta_val = None
                try:
                    bare_str = str(int(bare_count))
                except (TypeError, ValueError):
                    bare_str = "-"
                try:
                    guarded_str = str(int(guarded_count))
                except (TypeError, ValueError):
                    guarded_str = "-"
                try:
                    delta_val = int(guarded_count) - int(bare_count)  # type: ignore[arg-type]
                except Exception:
                    delta_val = None
                delta_str = f"{delta_val:+d}" if isinstance(delta_val, int) else "-"
                lines.append(
                    f"| {family} | {epsilon_str} | {bare_str} | {guarded_str} | {delta_str} |"
                )
            lines.append("")
        # Delta total and stability flags
        delta_total = rmt_info.get("delta_total")
        if isinstance(delta_total, int):
            lines.append(f"- Δ total: {delta_total:+d}")
        lines.append(f"- Stable: {rmt_info.get('stable', True)}")
        lines.append("")

    guard_overhead_info = certificate.get("guard_overhead", {}) or {}
    if guard_overhead_info:
        lines.append("### Guard Overhead")
        lines.append("")
        evaluated_flag = bool(guard_overhead_info.get("evaluated", True))
        if not evaluated_flag:
            # Make explicit when overhead was not evaluated by policy/profile
            lines.append("- Evaluated: false (skipped by policy/profile)")
        bare_ppl = guard_overhead_info.get("bare_ppl")
        guarded_ppl = guard_overhead_info.get("guarded_ppl")
        if isinstance(bare_ppl, int | float) and math.isfinite(float(bare_ppl)):
            lines.append(f"- Bare Primary Metric: {bare_ppl:.3f}")
        if isinstance(guarded_ppl, int | float) and math.isfinite(float(guarded_ppl)):
            lines.append(f"- Guarded Primary Metric: {guarded_ppl:.3f}")
        ratio = guard_overhead_info.get("overhead_ratio")
        percent = guard_overhead_info.get("overhead_percent")
        if (
            isinstance(ratio, int | float)
            and math.isfinite(float(ratio))
            and isinstance(percent, int | float)
            and math.isfinite(float(percent))
        ):
            lines.append(f"- Overhead: {ratio:.4f}x ({percent:+.2f}%)")
        elif isinstance(ratio, int | float) and math.isfinite(float(ratio)):
            lines.append(f"- Overhead: {ratio:.4f}x")
        overhead_source = guard_overhead_info.get("source")
        if overhead_source:
            lines.append(f"- Source: {overhead_source}")
        plan_ctx = certificate.get("provenance", {}).get("window_plan", {})
        if isinstance(plan_ctx, dict) and plan_ctx:
            plan_preview = (
                plan_ctx.get("preview_n")
                if plan_ctx.get("preview_n") is not None
                else plan_ctx.get("actual_preview")
            )
            plan_final = (
                plan_ctx.get("final_n")
                if plan_ctx.get("final_n") is not None
                else plan_ctx.get("actual_final")
            )
            plan_profile = plan_ctx.get("profile")
            lines.append(
                f"- Window Plan Used: profile={plan_profile}, preview={plan_preview}, final={plan_final}"
            )
        lines.append("")

    compression_diag = (
        certificate.get("structure", {}).get("compression_diagnostics", {})
        if isinstance(certificate.get("structure"), dict)
        else {}
    )
    inference_flags = compression_diag.get("inferred") or {}
    inference_sources = compression_diag.get("inference_source") or {}
    inference_log = compression_diag.get("inference_log") or []
    if inference_flags or inference_sources or inference_log:
        lines.append("## Inference")
        lines.append("")
        if inference_flags:
            lines.append("- **Fields Inferred:**")
            for field, flag in inference_flags.items():
                lines.append(f"  - {field}: {'yes' if flag else 'no'}")
        if inference_sources:
            lines.append("- **Sources:**")
            for field, source in inference_sources.items():
                lines.append(f"  - {field}: {source}")
        if inference_log:
            lines.append("- **Inference Log:**")
            for entry in inference_log:
                lines.append(f"  - {entry}")
        lines.append("")

    # Model and Configuration
    lines.append("## Model Information")
    lines.append("")
    meta = certificate["meta"]
    lines.append(f"- **Model ID:** {meta.get('model_id')}")
    lines.append(f"- **Adapter:** {meta.get('adapter')}")
    lines.append(f"- **Device:** {meta.get('device')}")
    lines.append(f"- **Timestamp:** {meta.get('ts')}")
    commit_value = meta.get("commit") or ""
    if commit_value:
        short_sha = str(commit_value)[:12]
        lines.append(f"- **Commit:** {short_sha}")
    else:
        lines.append("- **Commit:** (not set)")
    lines.append(f"- **Seed:** {meta.get('seed')}")
    seeds_map = meta.get("seeds", {})
    if isinstance(seeds_map, dict) and seeds_map:
        lines.append(
            "- **Seeds:** "
            f"python={seeds_map.get('python')}, "
            f"numpy={seeds_map.get('numpy')}, "
            f"torch={seeds_map.get('torch')}"
        )
    invarlock_version = meta.get("invarlock_version")
    if invarlock_version:
        lines.append(f"- **InvarLock Version:** {invarlock_version}")
    env_flags = meta.get("env_flags")
    if isinstance(env_flags, dict) and env_flags:
        lines.append("- **Env Flags:**")
        lines.append("  ```yaml")
        for k, v in env_flags.items():
            lines.append(f"  {k}: {v}")
        lines.append("  ```")
    # Determinism flags (if present)
    cuda_flags = meta.get("cuda_flags")
    if isinstance(cuda_flags, dict) and cuda_flags:
        parts = []
        for key in (
            "deterministic_algorithms",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "cudnn_allow_tf32",
            "cuda_matmul_allow_tf32",
            "CUBLAS_WORKSPACE_CONFIG",
        ):
            if key in cuda_flags and cuda_flags[key] is not None:
                parts.append(f"{key}={cuda_flags[key]}")
        if parts:
            lines.append(f"- **Determinism Flags:** {', '.join(parts)}")
    lines.append("")

    # Edit Configuration (removed duplicate Edit Information section)

    # Auto-tuning Configuration
    auto = certificate["auto"]
    if auto["tier"] != "none":
        lines.append("## Auto-Tuning Configuration")
        lines.append("")
        lines.append(f"- **Tier:** {auto['tier']}")
        lines.append(f"- **Probes Used:** {auto['probes_used']}")
    if auto.get("target_pm_ratio"):
        lines.append(
            f"- **Auto Policy Target Ratio (informational):** {auto['target_pm_ratio']:.3f}"
        )
        # Tiny relax breadcrumb for dev-only demos
        try:
            if bool(auto.get("tiny_relax")):
                lines.append("- Tiny relax: enabled (dev-only)")
        except Exception:
            pass
        lines.append("")

    resolved_policy = certificate.get("resolved_policy")
    if resolved_policy:
        lines.append("## Resolved Policy")
        lines.append("")
        lines.append("```yaml")
        resolved_yaml = yaml.safe_dump(
            resolved_policy, sort_keys=True, width=80, default_flow_style=False
        ).strip()
        for line in resolved_yaml.splitlines():
            lines.append(line)
        lines.append("```")
        lines.append("")

    policy_provenance = certificate.get("policy_provenance", {})
    if policy_provenance:
        lines.append("## Policy Provenance")
        lines.append("")
        lines.append(f"- **Tier:** {policy_provenance.get('tier')}")
        overrides_list = policy_provenance.get("overrides") or []
        if overrides_list:
            lines.append(f"- **Overrides:** {', '.join(overrides_list)}")
        else:
            lines.append("- **Overrides:** (none)")
        digest_value = policy_provenance.get("policy_digest")
        if digest_value:
            lines.append(f"- **Policy Digest:** `{digest_value}`")
        else:
            lines.append("- **Policy Digest:** (not recorded)")
        if policy_provenance.get("resolved_at"):
            lines.append(f"- **Resolved At:** {policy_provenance.get('resolved_at')}")
        lines.append("")

    # Dataset Information
    lines.append("## Dataset Configuration")
    lines.append("")
    dataset = certificate.get("dataset", {}) or {}
    prov = (
        (dataset.get("provider") or "unknown")
        if isinstance(dataset, dict)
        else "unknown"
    )
    lines.append(f"- **Provider:** {prov}")
    try:
        seq_len_val = (
            int(dataset.get("seq_len"))
            if isinstance(dataset.get("seq_len"), int | float)
            else dataset.get("seq_len")
        )
    except Exception:  # pragma: no cover - defensive
        seq_len_val = dataset.get("seq_len")
    if seq_len_val is not None:
        lines.append(f"- **Sequence Length:** {seq_len_val}")
    windows_blk = (
        dataset.get("windows", {}) if isinstance(dataset.get("windows"), dict) else {}
    )
    win_prev = windows_blk.get("preview")
    win_final = windows_blk.get("final")
    if win_prev is not None and win_final is not None:
        lines.append(f"- **Windows:** {win_prev} preview + {win_final} final")
    if windows_blk.get("seed") is not None:
        lines.append(f"- **Seed:** {windows_blk.get('seed')}")
    hash_blk = dataset.get("hash", {}) if isinstance(dataset.get("hash"), dict) else {}
    if hash_blk.get("preview_tokens") is not None:
        lines.append(f"- **Preview Tokens:** {hash_blk.get('preview_tokens'):,}")
    if hash_blk.get("final_tokens") is not None:
        lines.append(f"- **Final Tokens:** {hash_blk.get('final_tokens'):,}")
    if hash_blk.get("total_tokens") is not None:
        lines.append(f"- **Total Tokens:** {hash_blk.get('total_tokens'):,}")
    if hash_blk.get("dataset"):
        lines.append(f"- **Dataset Hash:** {hash_blk.get('dataset')}")
    tokenizer = dataset.get("tokenizer", {})
    if tokenizer.get("name") or tokenizer.get("hash"):
        vocab_size = tokenizer.get("vocab_size")
        vocab_suffix = f" (vocab {vocab_size})" if isinstance(vocab_size, int) else ""
        lines.append(
            f"- **Tokenizer:** {tokenizer.get('name', 'unknown')}{vocab_suffix}"
        )
        if tokenizer.get("hash"):
            lines.append(f"  - Hash: {tokenizer['hash']}")
        lines.append(
            f"  - BOS/EOS: {tokenizer.get('bos_token')} / {tokenizer.get('eos_token')}"
        )
        if tokenizer.get("pad_token") is not None:
            lines.append(f"  - PAD: {tokenizer.get('pad_token')}")
        if tokenizer.get("add_prefix_space") is not None:
            lines.append(f"  - add_prefix_space: {tokenizer.get('add_prefix_space')}")
    lines.append("")

    provenance_info = certificate.get("provenance", {}) or {}
    if provenance_info:
        lines.append("## Run Provenance")
        lines.append("")
        baseline_info = provenance_info.get("baseline", {}) or {}
        if baseline_info:
            lines.append(f"- **Baseline Run ID:** {baseline_info.get('run_id')}")
            if baseline_info.get("report_hash"):
                lines.append(f"  - Report Hash: `{baseline_info.get('report_hash')}`")
            if baseline_info.get("report_path"):
                lines.append(f"  - Report Path: {baseline_info.get('report_path')}")
        edited_info = provenance_info.get("edited", {}) or {}
        if edited_info:
            lines.append(f"- **Edited Run ID:** {edited_info.get('run_id')}")
            if edited_info.get("report_hash"):
                lines.append(f"  - Report Hash: `{edited_info.get('report_hash')}`")
            if edited_info.get("report_path"):
                lines.append(f"  - Report Path: {edited_info.get('report_path')}")
        window_plan = provenance_info.get("window_plan")
        if isinstance(window_plan, dict) and window_plan:
            preview_val = window_plan.get(
                "preview_n", window_plan.get("actual_preview")
            )
            final_val = window_plan.get("final_n", window_plan.get("actual_final"))
            lines.append(
                f"- **Window Plan:** profile={window_plan.get('profile')}, preview={preview_val}, final={final_val}"
            )
        provider_digest = provenance_info.get("provider_digest")
        if isinstance(provider_digest, dict) and provider_digest:
            ids_d = provider_digest.get("ids_sha256")
            tok_d = provider_digest.get("tokenizer_sha256")
            mask_d = provider_digest.get("masking_sha256")

            lines.append("- **Provider Digest:**")
            if tok_d:
                lines.append(
                    f"  - tokenizer_sha256: `{_short_digest(tok_d)}` (full in JSON)"
                )
            if ids_d:
                lines.append(f"  - ids_sha256: `{_short_digest(ids_d)}` (full in JSON)")
            if mask_d:
                lines.append(
                    f"  - masking_sha256: `{_short_digest(mask_d)}` (full in JSON)"
                )
        # Surface confidence label prominently
        try:
            conf = certificate.get("confidence", {}) or {}
            if isinstance(conf, dict) and conf.get("label"):
                lines.append(f"- **Confidence:** {conf.get('label')}")
        except Exception:
            pass
        # Surface policy version + thresholds hash (short)
        try:
            pd = certificate.get("policy_digest", {}) or {}
            if isinstance(pd, dict) and pd:
                pv = pd.get("policy_version")
                th = pd.get("thresholds_hash")
                if pv:
                    lines.append(f"- **Policy Version:** {pv}")
                if isinstance(th, str) and th:
                    short = th if len(th) <= 16 else (th[:8] + "…" + th[-8:])
                    lines.append(f"- **Thresholds Digest:** `{short}` (full in JSON)")
                if pd.get("changed"):
                    lines.append("- Note: policy changed")
        except Exception:
            pass
        lines.append("")

    # Structural Changes heading is printed with content later; avoid empty header here

    # Primary Metric (metric-v1) snapshot, if present
    try:
        pm = certificate.get("primary_metric")
        if isinstance(pm, dict) and pm:
            kind = pm.get("kind", "unknown")
            lines.append(f"## Primary Metric ({kind})")
            lines.append("")
            unit = pm.get("unit", "-")
            paired = pm.get("paired", False)
            reps = None
            # Snapshot only; bootstrap reps live in ppl.stats.bootstrap for ppl metrics
            # Mark estimated metrics (e.g., pseudo accuracy counts) clearly
            estimated_flag = False
            try:
                if bool(pm.get("estimated")):
                    estimated_flag = True
                elif str(pm.get("counts_source", "")).lower() == "pseudo_config":
                    estimated_flag = True
            except Exception:
                estimated_flag = False
            est_suffix = " (estimated)" if estimated_flag else ""
            lines.append(f"- Kind: {kind} (unit: {unit}){est_suffix}")
            gating_basis = pm.get("gating_basis") or pm.get("basis")
            if gating_basis:
                lines.append(f"- Basis: {gating_basis}")
            if isinstance(paired, bool):
                lines.append(f"- Paired: {paired}")
            reps = pm.get("reps")
            if isinstance(reps, int | float):
                lines.append(f"- Bootstrap Reps: {int(reps)}")
            ci = pm.get("ci") or pm.get("display_ci")
            if (
                isinstance(ci, list | tuple)
                and len(ci) == 2
                and all(isinstance(x, int | float) for x in ci)
            ):
                lines.append(f"- CI: {ci[0]:.3f}–{ci[1]:.3f}")
            prev = pm.get("preview")
            fin = pm.get("final")
            ratio = pm.get("ratio_vs_baseline")

            lines.append("")
            if estimated_flag and str(kind).lower() in {"accuracy", "vqa_accuracy"}:
                lines.append(
                    "- Note: Accuracy derived from pseudo counts (quick dev preset); use a labeled preset for measured accuracy."
                )
            lines.append("| Field | Value |")
            lines.append("|-------|-------|")
            lines.append(f"| Preview | {_fmt_by_kind(prev, str(kind))} |")
            lines.append(f"| Final | {_fmt_by_kind(fin, str(kind))} |")
            # For accuracy, ratio field is actually a delta (as per helper); clarify inline
            if kind in {"accuracy", "vqa_accuracy"}:
                lines.append(f"| Δ vs Baseline | {_fmt_by_kind(ratio, str(kind))} |")
                # When baseline accuracy is near-zero, clarify display rule
                try:
                    base_pt = pm.get("baseline_point")
                    if isinstance(base_pt, int | float) and base_pt < 0.05:
                        lines.append(
                            "- Note: baseline < 5%; ratio suppressed; showing Δpp"
                        )
                except Exception:
                    pass
            else:
                try:
                    lines.append(f"| Ratio vs Baseline | {float(ratio):.3f} |")
                except Exception:
                    lines.append("| Ratio vs Baseline | N/A |")
            lines.append("")
            # Secondary metrics (informational)
            try:
                secs = certificate.get("secondary_metrics")
                if isinstance(secs, list) and secs:
                    lines.append("## Secondary Metrics (informational)")
                    lines.append("")
                    lines.append("| Kind | Preview | Final | vs Baseline | CI |")
                    lines.append("|------|---------|-------|-------------|----|")
                    for m in secs:
                        if not isinstance(m, dict):
                            continue
                        k = m.get("kind", "?")
                        pv = _fmt_by_kind(m.get("preview"), str(k))
                        fv = _fmt_by_kind(m.get("final"), str(k))
                        rb = m.get("ratio_vs_baseline")
                        try:
                            rb_str = (
                                f"{float(rb):.3f}"
                                if (str(k).startswith("ppl"))
                                else _fmt_by_kind(rb, str(k))
                            )
                        except Exception:
                            rb_str = "N/A"
                        ci = m.get("display_ci") or m.get("ci")
                        if isinstance(ci, tuple | list) and len(ci) == 2:
                            ci_str = f"{float(ci[0]):.3f}-{float(ci[1]):.3f}"
                        else:
                            ci_str = "–"
                        lines.append(f"| {k} | {pv} | {fv} | {rb_str} | {ci_str} |")
                    lines.append("")
            except Exception:
                pass
    except Exception:
        pass

    # System Overhead section (latency/throughput)
    sys_over = certificate.get("system_overhead", {}) or {}
    if isinstance(sys_over, dict) and sys_over:
        _append_system_overhead_section(lines, sys_over)

    # Accuracy Subgroups (informational)
    try:
        cls = certificate.get("classification", {})
        sub = cls.get("subgroups") if isinstance(cls, dict) else None
        if isinstance(sub, dict) and sub:
            _append_accuracy_subgroups(lines, sub)
    except Exception:
        pass
    # Structural Changes
    try:
        structure = certificate.get("structure", {}) or {}
        params_changed = int(structure.get("params_changed", 0) or 0)
        layers_modified = int(structure.get("layers_modified", 0) or 0)
        bitwidth_changes = 0
        try:
            bitwidth_changes = int(len(structure.get("bitwidths", []) or []))
        except Exception:
            bitwidth_changes = 0
        # Decide whether to show the section
        has_changes = any(
            v > 0 for v in (params_changed, layers_modified, bitwidth_changes)
        )
        edit_name = str(certificate.get("edit_name", "unknown"))
        if has_changes:
            lines.append("## Structural Changes")
            lines.append("")
            lines.append("| Change Type | Count |")
            lines.append("|-------------|-------|")
            lines.append(f"| Parameters Changed | {params_changed:,} |")
            if edit_name == "quant_rtn":
                # For quantization: prefer a single clear line reconciling target vs applied
                # using diagnostics when available. Fallback to bitwidth-change count.
                try:
                    t_an = (structure.get("compression_diagnostics", {}) or {}).get(
                        "target_analysis", {}
                    )
                except Exception:
                    t_an = {}
                eligible = None
                modified = None
                if isinstance(t_an, dict) and t_an:
                    eligible = t_an.get("modules_eligible")
                    modified = t_an.get("modules_modified")
                if isinstance(modified, int) and isinstance(eligible, int):
                    lines.append(
                        f"| Linear Modules Quantized | {modified} of {eligible} targeted |"
                    )
                else:
                    total_bitwidth_changes = bitwidth_changes
                    if total_bitwidth_changes > 0 and layers_modified > 0:
                        modules_per_layer = total_bitwidth_changes // max(
                            layers_modified, 1
                        )
                        lines.append(
                            f"| Linear Modules Quantized | {total_bitwidth_changes} ({modules_per_layer} per block × {layers_modified} blocks) |"
                        )
                    elif total_bitwidth_changes > 0:
                        lines.append(
                            f"| Linear Modules Quantized | {total_bitwidth_changes} |"
                        )
            else:
                lines.append(f"| Layers Modified | {layers_modified} |")
            lines.append("")
    except Exception:
        # Best-effort; omit section on error
        pass

    # Add detailed breakdowns if available
    if structure.get("bitwidths") and edit_name != "quant_rtn":
        lines.append(f"| Bit-width Changes | {len(structure['bitwidths'])} layers |")
    if structure.get("ranks"):
        lines.append(f"| Rank Changes | {len(structure['ranks'])} layers |")

    lines.append("")

    # Compression Diagnostics
    compression_diag = structure.get("compression_diagnostics", {})
    if edit_name == "noop":
        lines.append("### Compression Diagnostics")
        lines.append("")
        lines.append("Not applicable (no parameters modified).")
        lines.append("")
    elif compression_diag:
        lines.append("### Compression Diagnostics")
        lines.append("")

        # Algorithm execution status
        status = compression_diag.get("execution_status", "unknown")
        status_emoji = (
            "✅" if status == "successful" else "❌" if status == "failed" else "⚠️"
        )
        lines.append(f"**Execution Status:** {status_emoji} {status.upper()}")
        lines.append("")

        # Target module analysis
        target_analysis = compression_diag.get("target_analysis", {})
        if target_analysis:
            lines.append("**Target Module Analysis:**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(
                f"| Modules Found | {target_analysis.get('modules_found', 0)} |"
            )
            lines.append(
                f"| Modules Eligible | {target_analysis.get('modules_eligible', 0)} |"
            )
            lines.append(
                f"| Modules Modified | {target_analysis.get('modules_modified', 0)} |"
            )
            try:
                _eligible = int(target_analysis.get("modules_eligible", 0))
                _modified = int(target_analysis.get("modules_modified", 0))
                lines.append(f"| Targets → Applied | {_eligible} → {_modified} |")
            except Exception:
                pass
            lines.append(f"| Scope | {target_analysis.get('scope', 'unknown')} |")
            lines.append("")

        # Parameter effectiveness
        param_analysis = compression_diag.get("parameter_analysis", {})
        if param_analysis:
            lines.append("**Parameter Effectiveness:**")
            lines.append("")
            for param, info in param_analysis.items():
                if isinstance(info, dict):
                    lines.append(
                        f"- **{param}:** {info.get('value', 'N/A')} ({info.get('effectiveness', 'unknown')})"
                    )
                else:
                    lines.append(f"- **{param}:** {info}")
            lines.append("")

        # Algorithm-specific details
        algo_details = compression_diag.get("algorithm_details", {})
        if algo_details:
            lines.append("**Algorithm Details:**")
            lines.append("")
            for key, value in algo_details.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        # Informational recommendations (non-normative)
        warnings = compression_diag.get("warnings", [])
        if warnings:
            lines.append("**ℹ️ Informational:**")
            lines.append("")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")

    # Variance Guard (Spectral/RMT summaries are already provided above)
    variance = certificate["variance"]
    lines.append("## Variance Guard")

    # Display whether VE was enabled after A/B test
    lines.append(f"- **Enabled:** {'Yes' if variance['enabled'] else 'No'}")

    if variance["enabled"]:
        # VE was enabled - show the gain
        gain_value = variance.get("gain", "N/A")
        if isinstance(gain_value, int | float):
            lines.append(f"- **Gain:** {gain_value:.3f}")
        else:
            lines.append(f"- **Gain:** {gain_value}")
    else:
        # VE was not enabled - show succinct reason if available, else a clear disabled message
        ppl_no_ve = variance.get("ppl_no_ve")
        ppl_with_ve = variance.get("ppl_with_ve")
        ratio_ci = variance.get("ratio_ci")
        if ppl_no_ve is not None and ppl_with_ve is not None and ratio_ci:
            lines.append(f"- **Primary metric without VE:** {ppl_no_ve:.3f}")
            lines.append(f"- **Primary metric with VE:** {ppl_with_ve:.3f}")
            gain_value = variance.get("gain")
            if isinstance(gain_value, int | float):
                lines.append(f"- **Gain (insufficient):** {gain_value:.3f}")
        else:
            lines.append(
                "- Variance Guard: Disabled (predictive gate not evaluated for this edit)."
            )
            # Add concise rationale aligned with Balanced predictive gate contract
            try:
                ve_policy = certificate.get("policies", {}).get("variance", {})
                min_effect = ve_policy.get("min_effect_lognll")
                if isinstance(min_effect, int | float):
                    lines.append(
                        f"- Predictive gate (Balanced): one-sided; enables only if CI excludes 0 and |mean Δ| ≥ {float(min_effect):.4g}."
                    )
                else:
                    lines.append(
                        "- Predictive gate (Balanced): one-sided; enables only if CI excludes 0 and |mean Δ| ≥ min_effect."
                    )
                lines.append(
                    "- Predictive Gate: evaluated=false (disabled under current policy/edit)."
                )
            except Exception:
                pass

    if variance.get("ratio_ci"):
        ratio_lo, ratio_hi = variance["ratio_ci"]
        lines.append(f"- **Ratio CI:** [{ratio_lo:.3f}, {ratio_hi:.3f}]")

    if variance.get("calibration") and variance.get("enabled"):
        calib = variance["calibration"]
        coverage = calib.get("coverage")
        requested = calib.get("requested")
        status = calib.get("status", "unknown")
        lines.append(f"- **Calibration:** {coverage}/{requested} windows ({status})")

    lines.append("")

    # MoE Observability (non-gating)
    moe = certificate.get("moe", {}) if isinstance(certificate.get("moe"), dict) else {}
    if moe:
        lines.append("## MoE Observability")
        lines.append("")
        # Core router fields
        for key in ("top_k", "capacity_factor", "expert_drop_rate"):
            if key in moe:
                lines.append(f"- **{key}:** {moe[key]}")
        # Utilization summary
        if "utilization_count" in moe or "utilization_mean" in moe:
            uc = moe.get("utilization_count")
            um = moe.get("utilization_mean")
            parts = []
            if uc is not None:
                parts.append(f"N={int(uc)}")
            if isinstance(um, int | float):
                parts.append(f"mean={um:.3f}")
            if parts:
                lines.append(f"- **Utilization:** {'; '.join(parts)}")
        # Delta summaries when available
        for key, label in (
            ("delta_load_balance_loss", "Δ load_balance_loss"),
            ("delta_router_entropy", "Δ router_entropy"),
            ("delta_utilization_mean", "Δ utilization mean"),
        ):
            if key in moe and isinstance(moe.get(key), int | float):
                lines.append(f"- **{label}:** {float(moe[key]):+.4f}")
        lines.append("")

    # Policy Summary
    lines.append("## Applied Policies")
    lines.append("")
    policies = certificate["policies"]
    for guard_name, policy in policies.items():
        lines.append(f"### {guard_name.title()}")
        lines.append("")
        policy_yaml = (
            yaml.safe_dump(policy, sort_keys=True, width=80).strip().splitlines()
        )
        lines.append("```yaml")
        for line in policy_yaml:
            lines.append(line)
        lines.append("```")
        lines.append("")

    # Artifacts
    lines.append("## Artifacts")
    lines.append("")
    artifacts = certificate["artifacts"]
    if artifacts.get("events_path"):
        lines.append(f"- **Events Log:** `{artifacts['events_path']}`")
    if artifacts.get("report_path"):
        lines.append(f"- **Full Report:** `{artifacts['report_path']}`")
    lines.append(f"- **Certificate Generated:** {artifacts['generated_at']}")
    lines.append("")

    # Certificate Hash for Integrity
    cert_hash = _compute_certificate_hash(certificate)
    lines.append("## Certificate Integrity")
    lines.append("")
    lines.append(f"**Certificate Hash:** `{cert_hash}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*This InvarLock safety certificate provides a comprehensive assessment of model compression safety.*"
    )
    lines.append(
        "*All metrics are compared against the uncompressed baseline model for safety validation.*"
    )

    return "\n".join(lines)
