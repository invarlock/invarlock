from __future__ import annotations

import json
import math

# mypy: ignore-errors
from typing import Any

import yaml

from .render_dataset_section import append_dataset_and_provenance_section
from .report_console import (
    compute_report_hash as _compute_report_hash,
)
from .report_summary import build_quality_gates_summary, build_safety_dashboard_summary


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


def _render_executive_dashboard(cert: dict[str, Any]) -> str:
    """Render executive summary dashboard table."""
    lines: list[str] = []
    _append_safety_dashboard_section(lines, cert)
    return "\n".join(lines).rstrip()


def _append_safety_dashboard_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    """Append a concise, first-screen summary table for the evaluation report."""
    summary = build_safety_dashboard_summary(evaluation_report)

    lines.append("| Check | Status | Quick Summary |")
    lines.append("|-------|--------|---------------|")
    lines.append(f"| Overall | {summary.overall_status} | Canonical gate outcomes |")
    for row in summary.rows:
        lines.append(f"| {row.label} | {row.status} | {row.summary} |")
    lines.append("")


def _append_primary_metric_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    """Append the Primary Metric section early for quick triage."""
    pm = evaluation_report.get("primary_metric")
    if not isinstance(pm, dict) or not pm:
        return

    kind = pm.get("kind", "unknown")
    lines.append("## Primary Metric")
    lines.append("")
    unit = pm.get("unit", "-")
    paired = pm.get("paired", False)

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

    if kind in {"accuracy", "vqa_accuracy"}:
        lines.append(f"| Δ vs Baseline | {_fmt_by_kind(ratio, str(kind))} |")
        try:
            base_pt = pm.get("baseline_point")
            if isinstance(base_pt, int | float) and base_pt < 0.05:
                lines.append("- Note: baseline < 5%; ratio suppressed; showing Δpp")
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
        secs = evaluation_report.get("secondary_metrics")
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


def _append_policy_configuration_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    resolved_policy = evaluation_report.get("resolved_policy")
    policy_provenance = evaluation_report.get("policy_provenance", {}) or {}
    has_prov = isinstance(policy_provenance, dict) and bool(policy_provenance)
    has_resolved = isinstance(resolved_policy, dict) and bool(resolved_policy)
    if not (has_prov or has_resolved):
        return

    lines.append("## Policy Configuration")
    lines.append("")

    tier = None
    if has_prov:
        tier = policy_provenance.get("tier")
    if not tier:
        tier = (evaluation_report.get("auto", {}) or {}).get("tier")
    digest_value = None
    if has_prov:
        digest_value = policy_provenance.get("policy_digest")
    if not digest_value:
        digest_value = (evaluation_report.get("policy_digest", {}) or {}).get(
            "thresholds_hash"
        )

    summary_parts: list[str] = []
    if tier:
        summary_parts.append(f"**Tier:** {tier}")
    if digest_value:
        summary_parts.append(f"**Digest:** `{_short_digest(str(digest_value))}`")
    if summary_parts:
        lines.append(" | ".join(summary_parts))

    if has_prov:
        overrides_list = policy_provenance.get("overrides") or []
        if overrides_list:
            lines.append(f"- **Overrides:** {', '.join(overrides_list)}")
        else:
            lines.append("- **Overrides:** (none)")
        if policy_provenance.get("resolved_at"):
            lines.append(f"- **Resolved At:** {policy_provenance.get('resolved_at')}")

    if has_resolved:
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Resolved Policy YAML</summary>")
        lines.append("")
        lines.append("```yaml")
        resolved_yaml = yaml.safe_dump(
            resolved_policy, sort_keys=True, width=80, default_flow_style=False
        ).strip()
        for line in resolved_yaml.splitlines():
            lines.append(line)
        lines.append("```")
        lines.append("")
        lines.append("</details>")

    lines.append("")


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


def _get_generated_at(evaluation_report: dict[str, Any]) -> str:
    artifacts = evaluation_report.get("artifacts")
    if isinstance(artifacts, dict):
        generated_at = artifacts.get("generated_at")
        if generated_at:
            return str(generated_at)
    policy_provenance = evaluation_report.get("policy_provenance")
    if isinstance(policy_provenance, dict):
        resolved_at = policy_provenance.get("resolved_at")
        if resolved_at:
            return str(resolved_at)
    return "(not recorded)"


def _get_window_plan_summary(evaluation_report: dict[str, Any]) -> str | None:
    try:
        plan_ctx = (
            evaluation_report.get("window_plan")
            or evaluation_report.get("dataset", {}).get("windows", {})
            or evaluation_report.get("ppl", {}).get("window_plan")
        )
        if not isinstance(plan_ctx, dict):
            return None
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
        if profile is None and preview_n is None and final_n is None:
            return None
        seq_len = evaluation_report.get("dataset", {}).get(
            "seq_len"
        ) or evaluation_report.get("dataset", {}).get("sequence_length")
        seq_len_suffix = f", seq_len={seq_len}" if seq_len else ""
        return f"Window Plan: {profile}, {preview_n}/{final_n}{seq_len_suffix}"
    except Exception:
        return None


def _append_report_header(lines: list[str], evaluation_report: dict[str, Any]) -> None:
    lines.append("# InvarLock Evaluation Report")
    lines.append("")
    lines.append(
        "> *Basis: “point” gates check the point estimate; “upper” gates check the CI "
        "upper bound; “point & upper” requires both to pass.*"
    )
    lines.append("")
    lines.append(f"**Schema Version:** {evaluation_report['schema_version']}")
    lines.append(f"**Run ID:** `{evaluation_report['run_id']}`")
    lines.append(f"**Generated:** {_get_generated_at(evaluation_report)}")
    lines.append(f"**Edit Type:** {evaluation_report.get('edit_name', 'Unknown')}")
    lines.append("")
    lines.append(
        "> Full evidence: see [`evaluation.report.json`](evaluation.report.json) for complete provenance, digests, and raw measurements."
    )
    lines.append("")


def _append_executive_summary_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    lines.append("## Executive Summary")
    lines.append("")
    summary = build_safety_dashboard_summary(evaluation_report)
    lines.append(f"**Overall Status:** {summary.overall_status}")
    window_plan_summary = _get_window_plan_summary(evaluation_report)
    if window_plan_summary:
        lines.append(f"- {window_plan_summary}")
    lines.append("")

    dashboard = _render_executive_dashboard(evaluation_report)
    if dashboard:
        lines.extend(dashboard.splitlines())
        lines.append("")


def _append_quality_gates_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    summary = build_quality_gates_summary(evaluation_report)

    lines.append("## Quality Gates")
    lines.append("")
    lines.append("| Gate | Status | Measured | Threshold | Basis | Description |")
    lines.append("|------|--------|----------|-----------|-------|-------------|")
    for row in summary.rows:
        lines.append(
            f"| {row.label} | {row.status} | {row.measured} | {row.threshold} | {row.basis} | {row.description} |"
        )
    if summary.hysteresis_applied:
        lines.append("- Note: hysteresis applied to gate boundary")
    lines.append("")


def render_report_markdown(evaluation_report: dict[str, Any]) -> str:
    """
    Render an evaluation report as a formatted Markdown report with pretty tables.

    Render an already-normalized evaluation report into Markdown.
    """
    lines: list[str] = []
    appendix_lines: list[str] = []
    edit_name = str(evaluation_report.get("edit_name") or "").lower()

    _append_report_header(lines, evaluation_report)

    plugins = evaluation_report.get("plugins", {})
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

    _append_executive_summary_section(lines, evaluation_report)

    _append_quality_gates_section(lines, evaluation_report)

    lines.append("## Guard Check Details")
    lines.append("")
    lines.append("| Guard Check | Status | Measured | Threshold | Description |")
    lines.append("|--------------|--------|----------|-----------|-------------|")

    inv_summary = evaluation_report.get("invariants", {}) or {}
    validation = evaluation_report.get("validation", {})
    inv_status = "✅ PASS" if validation.get("invariants_pass", False) else "❌ FAIL"
    inv_counts = inv_summary.get("summary", {}) if isinstance(inv_summary, dict) else {}
    if not isinstance(inv_counts, dict):
        inv_counts = {}
    inv_measure = (
        str(inv_summary.get("status", "not recorded")).upper()
        if isinstance(inv_summary, dict)
        else "NOT RECORDED"
    )
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
    invariants_failures = (
        inv_summary.get("failures") if isinstance(inv_summary, dict) else []
    ) or []
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
    spectral_summary = evaluation_report.get("spectral", {}) or {}
    caps_applied = (
        spectral_summary.get("caps_applied")
        if isinstance(spectral_summary, dict)
        else None
    )
    caps_measure = f"{caps_applied} violations" if caps_applied is not None else "N/A"
    spectral_threshold = (
        f"< {spectral_summary.get('max_caps')}"
        if isinstance(spectral_summary, dict)
        and spectral_summary.get("max_caps") is not None
        else "< 5"
    )
    lines.append(
        f"| Spectral Stability | {spec_status} | {caps_measure} | {spectral_threshold} | Weight matrix spectral norms |"
    )

    # Catastrophic spike safety stop row is now driven by primary metric flags
    if isinstance(evaluation_report.get("primary_metric"), dict):
        pm_ok = bool(validation.get("primary_metric_acceptable", True))
        pm_ratio = evaluation_report.get("primary_metric", {}).get("ratio_vs_baseline")
        if isinstance(pm_ratio, int | float):
            lines.append(
                f"| Catastrophic Spike Gate (hard stop) | {'✅ PASS' if pm_ok else '❌ FAIL'} | {pm_ratio:.3f}x | ≤ 2.0x | Hard stop @ 2.0× |"
            )

    # RMT Health remains a first-screen summary row alongside the other guard gates.
    rmt_status = "✅ PASS" if validation.get("rmt_stable", False) else "❌ FAIL"
    rmt_state = evaluation_report.get("rmt", {}).get("status", "unknown").title()
    lines.append(
        f"| RMT Health | {rmt_status} | {rmt_state} | ε-rule | Random Matrix Theory guard status |"
    )

    # Pairing + Bootstrap snapshot (quick audit surface)
    try:
        stats = (
            evaluation_report.get("dataset", {}).get("windows", {}).get("stats", {})
            or evaluation_report.get("ppl", {}).get("stats", {})
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
            parts: list[str] = []
            if paired_windows is not None:
                try:
                    parts.append(f"{int(paired_windows)} windows")
                except Exception:
                    parts.append(f"windows={paired_windows}")
            if isinstance(match_frac, int | float) and math.isfinite(float(match_frac)):
                parts.append(f"{float(match_frac) * 100.0:.1f}% match")
            elif match_frac is not None:
                parts.append(f"match={match_frac}")
            if isinstance(overlap_frac, int | float) and math.isfinite(
                float(overlap_frac)
            ):
                parts.append(f"{float(overlap_frac) * 100.0:.1f}% overlap")
            elif overlap_frac is not None:
                parts.append(f"overlap={overlap_frac}")
            lines.append(f"- ✅ Pairing: {', '.join(parts) if parts else 'N/A'}")
        if isinstance(bootstrap, dict):
            reps = bootstrap.get("replicates")
            bseed = bootstrap.get("seed")
            if reps is not None or bseed is not None:
                bits: list[str] = []
                if reps is not None:
                    try:
                        bits.append(f"{int(reps)} replicates")
                    except Exception:
                        bits.append(f"replicates={reps}")
                if bseed is not None:
                    try:
                        bits.append(f"seed={int(bseed)}")
                    except Exception:
                        bits.append(f"seed={bseed}")
                lines.append(f"- ✅ Bootstrap: {', '.join(bits) if bits else 'N/A'}")
        # Optional: show log-space paired Δ CI next to ratio CI for clarity
        delta_ci = evaluation_report.get("primary_metric", {}).get(
            "ci"
        ) or evaluation_report.get("ppl", {}).get("logloss_delta_ci")
        if (
            isinstance(delta_ci, tuple | list)
            and len(delta_ci) == 2
            and all(isinstance(x, int | float) for x in delta_ci)
        ):
            lines.append(
                f"- ℹ️ Log Δ (paired) CI: [{delta_ci[0]:.6f}, {delta_ci[1]:.6f}]"
            )
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

    _append_primary_metric_section(lines, evaluation_report)

    # Guard observability snapshots
    lines.append("## Guard Observability")
    lines.append("")

    spectral_info = evaluation_report.get("spectral", {}) or {}
    if spectral_info:
        lines.append("### Spectral Guard Summary")
        lines.append("")
        lines.append("| Metric | Value | Status |")
        lines.append("|--------|-------|--------|")

        spectral_ok = bool(validation.get("spectral_stable", False))
        caps_applied = spectral_info.get("caps_applied")
        max_caps = spectral_info.get("max_caps")
        caps_val = (
            f"{caps_applied}/{max_caps}"
            if caps_applied is not None and max_caps is not None
            else "-"
        )
        lines.append(
            f"| Caps Applied | {caps_val} | {'✅ OK' if spectral_ok else '❌ FAIL'} |"
        )

        summary = spectral_info.get("summary", {}) or {}
        caps_exceeded = summary.get("caps_exceeded")
        if caps_exceeded is not None:
            cap_status = "✅ OK" if not bool(caps_exceeded) else "⚠️ WARN"
            lines.append(f"| Caps Exceeded | {caps_exceeded} | {cap_status} |")

        top_scores = spectral_info.get("top_z_scores") or {}
        max_family: str | None = None
        max_module: str | None = None
        max_abs_z: float | None = None
        if isinstance(top_scores, dict):
            for family, entries in top_scores.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    z_val = entry.get("z")
                    if not (
                        isinstance(z_val, int | float) and math.isfinite(float(z_val))
                    ):
                        continue
                    z_abs = abs(float(z_val))
                    if max_abs_z is None or z_abs > max_abs_z:
                        max_abs_z = z_abs
                        max_family = str(family)
                        max_module = (
                            str(entry.get("module")) if entry.get("module") else None
                        )

        family_caps = spectral_info.get("family_caps") or {}
        kappa = None
        if max_family and isinstance(family_caps, dict):
            try:
                kappa = (family_caps.get(max_family, {}) or {}).get("kappa")
            except Exception:
                kappa = None
        kappa_f = (
            float(kappa)
            if isinstance(kappa, int | float) and math.isfinite(float(kappa))
            else None
        )

        if max_abs_z is not None:
            max_val = f"{max_abs_z:.3f}"
            if max_family:
                max_val += f" ({max_family})"
            if max_module:
                max_val += f" – {max_module}"
            if kappa_f is None:
                max_status = "ℹ️ No κ"
            elif max_abs_z <= kappa_f:
                max_status = f"✅ Within κ={kappa_f:.3f}"
            else:
                max_status = f"❌ Exceeds κ={kappa_f:.3f}"
            lines.append(f"| Max |z| | {max_val} | {max_status} |")

        mt_info = spectral_info.get("multiple_testing", {}) or {}
        if isinstance(mt_info, dict) and mt_info:
            mt_method = mt_info.get("method")
            mt_alpha = mt_info.get("alpha")
            mt_m = mt_info.get("m")
            parts: list[str] = []
            if mt_method:
                parts.append(f"method={mt_method}")
            if isinstance(mt_alpha, int | float) and math.isfinite(float(mt_alpha)):
                parts.append(f"α={float(mt_alpha):.3g}")
            if isinstance(mt_m, int | float) and math.isfinite(float(mt_m)):
                parts.append(f"m={int(mt_m)}")
            lines.append(
                f"| Multiple Testing | {', '.join(parts) if parts else '—'} | ℹ️ INFO |"
            )

        lines.append("")

        caps_by_family = spectral_info.get("caps_applied_by_family") or {}
        quantiles = spectral_info.get("family_z_quantiles") or {}
        if any(
            bool(x)
            for x in (caps_by_family, quantiles, family_caps, top_scores)
            if isinstance(x, dict)
        ):
            lines.append("<details>")
            lines.append("<summary>Per-family details</summary>")
            lines.append("")
            lines.append("| Family | κ | q95 | Max |z| | Violations |")
            lines.append("|--------|---|-----|--------|------------|")

            families: set[str] = set()
            for block in (caps_by_family, quantiles, family_caps, top_scores):
                if isinstance(block, dict):
                    families.update(str(k) for k in block.keys())

            for family in sorted(families):
                kappa = None
                if isinstance(family_caps, dict):
                    kappa = (family_caps.get(family, {}) or {}).get("kappa")
                kappa_str = (
                    f"{float(kappa):.3f}"
                    if isinstance(kappa, int | float) and math.isfinite(float(kappa))
                    else "-"
                )

                q95 = None
                max_z = None
                if isinstance(quantiles, dict):
                    stats = quantiles.get(family) or {}
                    if isinstance(stats, dict):
                        q95 = stats.get("q95")
                        max_z = stats.get("max")
                q95_str = f"{q95:.3f}" if isinstance(q95, int | float) else "-"
                max_str = f"{max_z:.3f}" if isinstance(max_z, int | float) else "-"

                violations = None
                if isinstance(caps_by_family, dict):
                    violations = caps_by_family.get(family)
                v_str = (
                    str(int(violations)) if isinstance(violations, int | float) else "0"
                )

                lines.append(
                    f"| {family} | {kappa_str} | {q95_str} | {max_str} | {v_str} |"
                )

            if isinstance(top_scores, dict) and top_scores:
                lines.append("")
                lines.append("Top |z| per family:")
                for family in sorted(top_scores.keys()):
                    entries = top_scores[family]
                    if not isinstance(entries, list) or not entries:
                        continue
                    formatted_entries = []
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        module_name = entry.get("module", "unknown")
                        z_val = entry.get("z")
                        if isinstance(z_val, int | float) and math.isfinite(
                            float(z_val)
                        ):
                            z_str = f"{z_val:.3f}"
                        else:
                            z_str = "n/a"
                        formatted_entries.append(f"{module_name} (|z|={z_str})")
                    lines.append(f"- {family}: {', '.join(formatted_entries)}")

            lines.append("")
            lines.append("</details>")
            lines.append("")

    rmt_info = evaluation_report.get("rmt", {}) or {}
    if rmt_info:
        lines.append("### RMT Guard")
        lines.append("")
        families = rmt_info.get("families") or {}
        stable = bool(rmt_info.get("stable", True))
        status = "✅ OK" if stable else "❌ FAIL"
        mode = rmt_info.get("mode")
        if isinstance(mode, str) and mode.strip():
            lines.append(f"- Mode: `{mode.strip()}`")
        measurement_contract = (
            rmt_info.get("measurement_contract")
            if isinstance(rmt_info.get("measurement_contract"), dict)
            else {}
        )
        if measurement_contract:
            contract_parts: list[str] = []
            estimator = measurement_contract.get("estimator")
            if isinstance(estimator, dict) and estimator:
                contract_parts.append(
                    f"estimator={json.dumps(estimator, sort_keys=True)}"
                )
            activation_sampling = measurement_contract.get("activation_sampling")
            if isinstance(activation_sampling, dict) and activation_sampling:
                contract_parts.append(
                    f"activation_sampling={json.dumps(activation_sampling, sort_keys=True)}"
                )
            if contract_parts:
                lines.append(f"- Measurement Contract: {'; '.join(contract_parts)}")
        delta_total = rmt_info.get("delta_total")
        if isinstance(delta_total, int):
            lines.append(f"- Δ total: {delta_total:+d}")
        lines.append(f"- Status: {status}")
        lines.append(f"- Families: {len(families)}")
        if families:
            edge_risk_mode = any(
                isinstance(data, dict) and ("edge_base" in data or "edge_cur" in data)
                for data in families.values()
            )
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>RMT family details</summary>")
            lines.append("")
            if edge_risk_mode:
                lines.append("| Family | ε_f | Edge Base | Edge Cur | Δ |")
                lines.append("|--------|-----|-----------|----------|---|")
            else:
                lines.append("| Family | ε_f | Bare | Guarded | Δ |")
                lines.append("|--------|-----|------|---------|---|")
            for family, data in families.items():
                epsilon_val = data.get("epsilon")
                epsilon_str = (
                    f"{epsilon_val:.3f}"
                    if isinstance(epsilon_val, int | float)
                    else "-"
                )
                if edge_risk_mode:
                    edge_base = data.get("edge_base")
                    edge_cur = data.get("edge_cur")
                    delta_val = data.get("delta")
                    edge_base_str = (
                        f"{edge_base:.3f}"
                        if isinstance(edge_base, int | float)
                        else "-"
                    )
                    edge_cur_str = (
                        f"{edge_cur:.3f}" if isinstance(edge_cur, int | float) else "-"
                    )
                    delta_str = (
                        f"{delta_val:+.3f}"
                        if isinstance(delta_val, int | float)
                        else "-"
                    )
                    lines.append(
                        f"| {family} | {epsilon_str} | {edge_base_str} | {edge_cur_str} | {delta_str} |"
                    )
                    continue
                bare_count = data.get("bare", 0)
                guarded_count = data.get("guarded", 0)
                delta_count = None
                try:
                    bare_str = str(int(bare_count))
                except (TypeError, ValueError):
                    bare_str = "-"
                try:
                    guarded_str = str(int(guarded_count))
                except (TypeError, ValueError):
                    guarded_str = "-"
                try:
                    delta_count = int(guarded_count) - int(bare_count)  # type: ignore[arg-type]
                except Exception:
                    delta_count = None
                delta_str = f"{delta_count:+d}" if isinstance(delta_count, int) else "-"
                lines.append(
                    f"| {family} | {epsilon_str} | {bare_str} | {guarded_str} | {delta_str} |"
                )
            lines.append("")
            lines.append("</details>")
            lines.append("")
        else:
            lines.append("")

    guard_overhead_info = evaluation_report.get("guard_overhead", {}) or {}
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
        plan_ctx = evaluation_report.get("provenance", {}).get("window_plan", {})
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
        evaluation_report.get("structure", {}).get("compression_diagnostics", {})
        if isinstance(evaluation_report.get("structure"), dict)
        else {}
    )
    inference_flags = compression_diag.get("inferred") or {}
    inference_sources = compression_diag.get("inference_source") or {}
    inference_log = compression_diag.get("inference_log") or []
    if inference_flags or inference_sources or inference_log:
        appendix_lines.append("### Inference Diagnostics")
        appendix_lines.append("")
        if inference_flags:
            appendix_lines.append("- **Fields Inferred:**")
            for field, flag in inference_flags.items():
                appendix_lines.append(f"  - {field}: {'yes' if flag else 'no'}")
        if inference_sources:
            appendix_lines.append("- **Sources:**")
            for field, source in inference_sources.items():
                appendix_lines.append(f"  - {field}: {source}")
        if inference_log:
            appendix_lines.append("- **Inference Log:**")
            for entry in inference_log:
                appendix_lines.append(f"  - {entry}")
        appendix_lines.append("")

    # Model and Configuration
    lines.append("## Model Information")
    lines.append("")
    meta = evaluation_report.get("meta", {}) or {}
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
    cuda_flags = meta.get("cuda_flags")

    # Compressed determinism/environment summary for readability
    det_parts: list[str] = []
    for label, keys in (
        ("torch_det", ("torch_deterministic_algorithms", "deterministic_algorithms")),
        ("cudnn_det", ("cudnn_deterministic",)),
        ("cudnn_bench", ("cudnn_benchmark",)),
        ("tf32_matmul", ("cuda_matmul_allow_tf32",)),
        ("tf32_cudnn", ("cudnn_allow_tf32",)),
        ("cublas_ws", ("CUBLAS_WORKSPACE_CONFIG",)),
    ):
        val = None
        for key in keys:
            if isinstance(env_flags, dict) and env_flags.get(key) is not None:
                val = env_flags.get(key)
                break
            if isinstance(cuda_flags, dict) and cuda_flags.get(key) is not None:
                val = cuda_flags.get(key)
                break
        if val is not None:
            det_parts.append(f"{label}={val}")
    if det_parts:
        lines.append(f"- **Determinism:** {', '.join(det_parts)}")

    full_flags: dict[str, Any] = {}
    if isinstance(env_flags, dict) and env_flags:
        full_flags["env_flags"] = env_flags
    if isinstance(cuda_flags, dict) and cuda_flags:
        full_flags["cuda_flags"] = cuda_flags
    if full_flags:
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Environment flags (full)</summary>")
        lines.append("")
        lines.append("```yaml")
        flags_yaml = yaml.safe_dump(full_flags, sort_keys=True, width=80).strip()
        for line in flags_yaml.splitlines():
            lines.append(line)
        lines.append("```")
        lines.append("")
        lines.append("</details>")
    lines.append("")

    # Edit Configuration (removed duplicate Edit Information section)

    # Auto-tuning Configuration
    auto = evaluation_report.get("auto", {}) or {}
    auto_tier = auto.get("tier")
    if auto_tier and auto_tier != "none":
        lines.append("## Auto-Tuning Configuration")
        lines.append("")
        lines.append(f"- **Tier:** {auto_tier}")
        lines.append(f"- **Probes Used:** {auto.get('probes_used', 0)}")
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

    append_dataset_and_provenance_section(lines, evaluation_report)

    # Structural Changes heading is printed with content later; avoid empty header here

    # System Overhead section (latency/throughput)
    sys_over = evaluation_report.get("system_overhead", {}) or {}
    if isinstance(sys_over, dict) and sys_over:
        _append_system_overhead_section(lines, sys_over)

    # Accuracy Subgroups (informational)
    try:
        cls = evaluation_report.get("classification", {})
        sub = cls.get("subgroups") if isinstance(cls, dict) else None
        if isinstance(sub, dict) and sub:
            _append_accuracy_subgroups(lines, sub)
    except Exception:
        pass
    # Structural Changes
    try:
        structure = evaluation_report.get("structure", {}) or {}
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
        edit_name = str(evaluation_report.get("edit_name", "unknown"))
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
    variance = evaluation_report.get("variance", {}) or {}
    if not isinstance(variance, dict):
        variance = {}
    appendix_lines.append("### Variance Guard")
    appendix_lines.append("")

    # Display whether VE was enabled after A/B test
    variance_enabled = bool(variance.get("enabled"))
    appendix_lines.append(f"- **Enabled:** {'Yes' if variance_enabled else 'No'}")

    if variance_enabled:
        # VE was enabled - show the gain
        gain_value = variance.get("gain", "N/A")
        if isinstance(gain_value, int | float):
            appendix_lines.append(f"- **Gain:** {gain_value:.3f}")
        else:
            appendix_lines.append(f"- **Gain:** {gain_value}")
    else:
        # VE was not enabled - show succinct reason if available, else a clear disabled message
        ppl_no_ve = variance.get("ppl_no_ve")
        ppl_with_ve = variance.get("ppl_with_ve")
        ratio_ci = variance.get("ratio_ci")
        if ppl_no_ve is not None and ppl_with_ve is not None and ratio_ci:
            appendix_lines.append(f"- **Primary metric without VE:** {ppl_no_ve:.3f}")
            appendix_lines.append(f"- **Primary metric with VE:** {ppl_with_ve:.3f}")
            gain_value = variance.get("gain")
            if isinstance(gain_value, int | float):
                appendix_lines.append(f"- **Gain (insufficient):** {gain_value:.3f}")
        else:
            appendix_lines.append(
                "- Variance Guard: Disabled (predictive gate not evaluated for this edit)."
            )
            # Add concise rationale aligned with Balanced predictive gate contract
            try:
                ve_policy = evaluation_report.get("policies", {}).get("variance", {})
                min_effect = ve_policy.get("min_effect_lognll")
                if isinstance(min_effect, int | float):
                    appendix_lines.append(
                        f"- Predictive gate (Balanced): one-sided; enables only if CI excludes 0 and |mean Δ| ≥ {float(min_effect):.4g}."
                    )
                else:
                    appendix_lines.append(
                        "- Predictive gate (Balanced): one-sided; enables only if CI excludes 0 and |mean Δ| ≥ min_effect."
                    )
                appendix_lines.append(
                    "- Predictive Gate: evaluated=false (disabled under current policy/edit)."
                )
            except Exception:
                pass

    if variance.get("ratio_ci"):
        ratio_lo, ratio_hi = variance["ratio_ci"]
        appendix_lines.append(f"- **Ratio CI:** [{ratio_lo:.3f}, {ratio_hi:.3f}]")

    if variance.get("calibration") and variance.get("enabled"):
        calib = variance["calibration"]
        coverage = calib.get("coverage")
        requested = calib.get("requested")
        status = calib.get("status", "unknown")
        appendix_lines.append(
            f"- **Calibration:** {coverage}/{requested} windows ({status})"
        )
    appendix_lines.append("")

    lines.append("")

    # MoE Observability (non-gating)
    moe = (
        evaluation_report.get("moe", {})
        if isinstance(evaluation_report.get("moe"), dict)
        else {}
    )
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

    _append_policy_configuration_section(lines, evaluation_report)

    appendix_lines.append("### Artifacts")
    appendix_lines.append("")
    artifacts = evaluation_report["artifacts"]
    if artifacts.get("events_path"):
        appendix_lines.append(f"- **Events Log:** `{artifacts['events_path']}`")
    if artifacts.get("report_path"):
        appendix_lines.append(f"- **Full Report:** `{artifacts['report_path']}`")
    appendix_lines.append(
        f"- **Report Generated:** {_get_generated_at(evaluation_report)}"
    )
    appendix_lines.append("")

    if appendix_lines:
        lines.append("## Appendix")
        lines.append("")
        lines.extend(appendix_lines)

    # Report Hash for Integrity
    cert_hash = _compute_report_hash(evaluation_report)
    lines.append("## Evaluation Report Integrity")
    lines.append("")
    lines.append(f"**Report Hash:** `{cert_hash}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*This InvarLock Evaluation Report summarizes baseline‑paired evaluation results for a subject model relative to the provided baseline snapshot under the configured profile/preset.*"
    )
    lines.append(
        "*It reports regression-risk indicators for the measured signals; it is not a broad AI safety, alignment, or content-safety guarantee.*"
    )

    return "\n".join(lines)
