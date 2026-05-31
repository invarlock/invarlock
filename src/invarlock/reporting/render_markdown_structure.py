"""Structural and appendix sections for Markdown report rendering."""

from __future__ import annotations

from typing import Any


def get_generated_at(evaluation_report: dict[str, Any]) -> str:
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


def append_structural_changes_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
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
        pass


def append_compression_diagnostics_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    structure = evaluation_report.get("structure", {}) or {}
    compression_diag = structure.get("compression_diagnostics", {})
    edit_name = str(evaluation_report.get("edit_name", "unknown"))

    if edit_name == "noop":
        lines.append("### Compression Diagnostics")
        lines.append("")
        lines.append("Not applicable (no parameters modified).")
        lines.append("")
    elif compression_diag:
        lines.append("### Compression Diagnostics")
        lines.append("")

        status = compression_diag.get("execution_status", "unknown")
        status_emoji = (
            "✅" if status == "successful" else "❌" if status == "failed" else "⚠️"
        )
        lines.append(f"**Execution Status:** {status_emoji} {status.upper()}")
        lines.append("")

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

        algo_details = compression_diag.get("algorithm_details", {})
        if algo_details:
            lines.append("**Algorithm Details:**")
            lines.append("")
            for key, value in algo_details.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        warnings = compression_diag.get("warnings", [])
        if warnings:
            lines.append("**ℹ️ Informational:**")
            lines.append("")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")


def append_moe_observability_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    moe = (
        evaluation_report.get("moe", {})
        if isinstance(evaluation_report.get("moe"), dict)
        else {}
    )
    if not moe:
        return
    lines.append("## MoE Observability")
    lines.append("")
    for key in ("top_k", "capacity_factor", "expert_drop_rate"):
        if key in moe:
            lines.append(f"- **{key}:** {moe[key]}")
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
    for key, label in (
        ("delta_load_balance_loss", "Δ load_balance_loss"),
        ("delta_router_entropy", "Δ router_entropy"),
        ("delta_utilization_mean", "Δ utilization mean"),
    ):
        if key in moe and isinstance(moe.get(key), int | float):
            lines.append(f"- **{label}:** {float(moe[key]):+.4f}")
    lines.append("")


def append_appendix_sections(
    lines: list[str], appendix_lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    _append_inference_diagnostics_section(appendix_lines, evaluation_report)
    _append_variance_guard_appendix(appendix_lines, evaluation_report)

    appendix_lines.append("### Artifacts")
    appendix_lines.append("")
    artifacts = evaluation_report["artifacts"]
    if artifacts.get("events_path"):
        appendix_lines.append(f"- **Events Log:** `{artifacts['events_path']}`")
    if artifacts.get("report_path"):
        appendix_lines.append(f"- **Full Report:** `{artifacts['report_path']}`")
    appendix_lines.append(
        f"- **Report Generated:** {get_generated_at(evaluation_report)}"
    )
    appendix_lines.append("")

    if appendix_lines:
        lines.append("## Appendix")
        lines.append("")
        lines.extend(appendix_lines)


def _append_inference_diagnostics_section(
    appendix_lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    compression_diag = (
        evaluation_report.get("structure", {}).get("compression_diagnostics", {})
        if isinstance(evaluation_report.get("structure"), dict)
        else {}
    )
    inference_flags = compression_diag.get("inferred") or {}
    inference_sources = compression_diag.get("inference_source") or {}
    inference_log = compression_diag.get("inference_log") or []
    if not (inference_flags or inference_sources or inference_log):
        return

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


def _append_variance_guard_appendix(
    appendix_lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    variance = evaluation_report.get("variance", {}) or {}
    if not isinstance(variance, dict):
        variance = {}
    appendix_lines.append("### Variance Guard")
    appendix_lines.append("")

    variance_enabled = bool(variance.get("enabled"))
    appendix_lines.append(f"- **Enabled:** {'Yes' if variance_enabled else 'No'}")

    if variance_enabled:
        gain_value = variance.get("gain", "N/A")
        if isinstance(gain_value, int | float):
            appendix_lines.append(f"- **Gain:** {gain_value:.3f}")
        else:
            appendix_lines.append(f"- **Gain:** {gain_value}")
    else:
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
