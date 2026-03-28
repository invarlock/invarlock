from __future__ import annotations

import math
from typing import Any

import yaml

from .render_dataset_section import append_dataset_and_provenance_section
from .render_guard_sections import (
    append_guard_check_details_section,
    append_guard_observability_sections,
)
from .render_helpers import _fmtv, _p, _short_digest
from .render_model_context import append_model_context_sections
from .render_primary_metric_section import (
    append_primary_metric_section as _append_primary_metric_section,
)
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
        except (
            AttributeError,
            ImportError,
            KeyError,
            OSError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            b_val = float("nan")
        try:
            e_val = float(e_raw)
        except (
            AttributeError,
            ImportError,
            KeyError,
            OSError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
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
        except (
            AttributeError,
            ImportError,
            KeyError,
            OSError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            npv = 0
        try:
            nfi = int(rec.get("n_final", 0))
        except (
            AttributeError,
            ImportError,
            KeyError,
            OSError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            nfi = 0
        dp = rec.get("delta_pp")
        try:
            dp_str = f"{float(dp):+.1f} pp"
        except (
            AttributeError,
            ImportError,
            KeyError,
            OSError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
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
    except (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
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

    append_guard_check_details_section(lines, evaluation_report)

    _append_primary_metric_section(lines, evaluation_report)

    append_guard_observability_sections(lines, evaluation_report)

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

    append_model_context_sections(lines, evaluation_report)

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
    except (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        pass
    # Structural Changes
    try:
        structure = evaluation_report.get("structure", {}) or {}
        params_changed = int(structure.get("params_changed", 0) or 0)
        layers_modified = int(structure.get("layers_modified", 0) or 0)
        bitwidth_changes = 0
        try:
            bitwidth_changes = int(len(structure.get("bitwidths", []) or []))
        except (
            AttributeError,
            ImportError,
            KeyError,
            OSError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
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
                except (
                    AttributeError,
                    ImportError,
                    KeyError,
                    OSError,
                    OverflowError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
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
    except (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
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
            except (
                AttributeError,
                ImportError,
                KeyError,
                OSError,
                OverflowError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
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
            except (
                AttributeError,
                ImportError,
                KeyError,
                OSError,
                OverflowError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
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
