from __future__ import annotations

from typing import Any

import yaml

from .render_guard_sections import (
    append_guard_check_details_section,
    append_guard_observability_sections,
)
from .render_markdown_structure import (
    append_appendix_sections as _append_appendix_sections,
)
from .render_markdown_structure import (
    append_compression_diagnostics_section as _append_compression_diagnostics_section,
)
from .render_markdown_structure import (
    append_moe_observability_section as _append_moe_observability_section,
)
from .render_markdown_structure import (
    append_structural_changes_section as _append_structural_changes_section,
)
from .render_markdown_structure import (
    get_generated_at as _get_generated_at,
)
from .render_markdown_tables import (
    append_accuracy_subgroups as _append_accuracy_subgroups,
)
from .render_markdown_tables import (
    append_system_overhead_section as _append_system_overhead_section,
)
from .report_summary import build_quality_gates_summary, build_safety_dashboard_summary
from .report_summary import (
    compute_report_hash as _compute_report_hash,
)
from .utils import _fmt_by_kind, _short_digest

_MODEL_CONTEXT_PARSE_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
)


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


def _primary_metric_is_pseudo_accuracy(evaluation_report: dict[str, Any]) -> bool:
    primary_metric = evaluation_report.get("primary_metric")
    if not isinstance(primary_metric, dict):
        return False
    kind = str(primary_metric.get("kind") or "").strip().lower()
    counts_source = str(primary_metric.get("counts_source") or "").strip().lower()
    return kind == "accuracy" and (
        counts_source == "pseudo_config" or bool(primary_metric.get("estimated"))
    )


def _is_non_assurance_report(evaluation_report: dict[str, Any]) -> bool:
    assurance = evaluation_report.get("assurance")
    if not isinstance(assurance, dict):
        return True
    mode = str(assurance.get("mode") or "").strip().lower()
    runtime_status = (
        str(assurance.get("runtime_provenance_verification_status") or "")
        .strip()
        .lower()
    )
    assurance_verdict = (
        str(
            assurance.get("verified_assurance_verdict")
            or assurance.get("verdict")
            or ""
        )
        .strip()
        .lower()
    )
    if mode != "strict":
        return True
    if runtime_status and runtime_status not in {"verified", "pass", "ok"}:
        return True
    if assurance_verdict and assurance_verdict not in {"verified", "pass", "ok"}:
        return True
    return False


def _is_estimated_metric(primary_metric: dict[str, Any]) -> bool:
    try:
        if bool(primary_metric.get("estimated")):
            return True
        return str(primary_metric.get("counts_source", "")).lower() == "pseudo_config"
    except _MODEL_CONTEXT_PARSE_EXCEPTIONS:
        return False


def _format_secondary_metric_ratio(metric: dict[str, Any], kind: str) -> str:
    ratio = metric.get("ratio_vs_baseline")
    try:
        if kind.startswith("ppl"):
            if isinstance(ratio, int | float):
                return f"{float(ratio):.3f}"
            return "N/A"
        return _fmt_by_kind(ratio, kind)
    except _MODEL_CONTEXT_PARSE_EXCEPTIONS:
        return "N/A"


def _append_primary_metric_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    primary_metric = evaluation_report.get("primary_metric")
    if not isinstance(primary_metric, dict) or not primary_metric:
        return

    kind = primary_metric.get("kind", "unknown")
    lines.append("## Primary Metric")
    lines.append("")
    unit = primary_metric.get("unit", "-")
    paired = primary_metric.get("paired", False)
    estimated_flag = _is_estimated_metric(primary_metric)
    estimated_suffix = " (estimated)" if estimated_flag else ""

    lines.append(f"- Kind: {kind} (unit: {unit}){estimated_suffix}")
    gating_basis = primary_metric.get("gating_basis") or primary_metric.get("basis")
    if gating_basis:
        lines.append(f"- Basis: {gating_basis}")
    if isinstance(paired, bool):
        lines.append(f"- Paired: {paired}")
    reps = primary_metric.get("reps")
    if isinstance(reps, int | float):
        lines.append(f"- Bootstrap Reps: {int(reps)}")
    ci = primary_metric.get("ci") or primary_metric.get("display_ci")
    if (
        isinstance(ci, list | tuple)
        and len(ci) == 2
        and all(isinstance(value, int | float) for value in ci)
    ):
        lines.append(f"- CI: {ci[0]:.3f}–{ci[1]:.3f}")

    preview = primary_metric.get("preview")
    final = primary_metric.get("final")
    ratio = primary_metric.get("ratio_vs_baseline")

    lines.append("")
    kind_name = str(kind).lower()
    if estimated_flag and kind_name == "accuracy":
        lines.append(
            "- Note: Accuracy derived from pseudo counts (quick dev preset); use a labeled preset for measured accuracy."
        )
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Preview | {_fmt_by_kind(preview, str(kind))} |")
    lines.append(f"| Final | {_fmt_by_kind(final, str(kind))} |")

    if kind == "accuracy":
        lines.append(f"| Δ vs Baseline | {_fmt_by_kind(ratio, str(kind))} |")
        try:
            baseline_point = primary_metric.get("baseline_point")
        except _MODEL_CONTEXT_PARSE_EXCEPTIONS:
            baseline_point = None
        if isinstance(baseline_point, int | float) and baseline_point < 0.05:
            lines.append("- Note: baseline < 5%; ratio suppressed; showing Δpp")
    else:
        try:
            if isinstance(ratio, int | float):
                lines.append(f"| Ratio vs Baseline | {float(ratio):.3f} |")
            else:
                lines.append("| Ratio vs Baseline | N/A |")
        except _MODEL_CONTEXT_PARSE_EXCEPTIONS:
            lines.append("| Ratio vs Baseline | N/A |")
    lines.append("")

    secondary_metrics = evaluation_report.get("secondary_metrics")
    if not isinstance(secondary_metrics, list) or not secondary_metrics:
        return

    lines.append("## Secondary Metrics (informational)")
    lines.append("")
    lines.append("| Kind | Preview | Final | vs Baseline | CI |")
    lines.append("|------|---------|-------|-------------|----|")
    for metric in secondary_metrics:
        if not isinstance(metric, dict):
            continue
        metric_kind = str(metric.get("kind", "?"))
        preview_value = _fmt_by_kind(metric.get("preview"), metric_kind)
        final_value = _fmt_by_kind(metric.get("final"), metric_kind)
        ratio_value = _format_secondary_metric_ratio(metric, metric_kind)
        ci = metric.get("display_ci") or metric.get("ci")
        if isinstance(ci, tuple | list) and len(ci) == 2:
            ci_value = f"{float(ci[0]):.3f}-{float(ci[1]):.3f}"
        else:
            ci_value = "–"
        lines.append(
            f"| {metric_kind} | {preview_value} | {final_value} | {ratio_value} | {ci_value} |"
        )
    lines.append("")


def _dataset_hash_source_label(source: Any) -> str | None:
    source_map = {
        "explicit_preview_final_hashes": "provider-derived explicit preview/final hashes",
        "explicit_token_ids": "content-derived token IDs",
        "config_fallback": "config-derived fallback",
    }
    key = str(source or "").strip()
    return source_map.get(key)


def _append_dataset_and_provenance_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    """Append the dataset/provenance Markdown block."""
    dataset = evaluation_report.get("dataset", {}) or {}
    provenance_info = evaluation_report.get("provenance", {}) or {}

    has_dataset = isinstance(dataset, dict) and bool(dataset)
    has_provenance = isinstance(provenance_info, dict) and bool(provenance_info)
    if not (has_dataset or has_provenance):
        return

    lines.append("## Dataset and Provenance")
    lines.append("")

    if has_dataset:
        provider = dataset.get("provider") or "unknown"
        lines.append(f"- **Provider:** {provider}")
        seq_len_raw = dataset.get("seq_len")
        seq_len_val = (
            int(seq_len_raw) if isinstance(seq_len_raw, int | float) else seq_len_raw
        )
        if seq_len_val is not None:
            lines.append(f"- **Sequence Length:** {seq_len_val}")
        windows_blk = (
            dataset.get("windows", {})
            if isinstance(dataset.get("windows"), dict)
            else {}
        )
        win_prev = windows_blk.get("preview")
        win_final = windows_blk.get("final")
        if win_prev is not None and win_final is not None:
            lines.append(f"- **Windows:** {win_prev} preview + {win_final} final")
        if windows_blk.get("seed") is not None:
            lines.append(f"- **Seed:** {windows_blk.get('seed')}")
        hash_blk = (
            dataset.get("hash", {}) if isinstance(dataset.get("hash"), dict) else {}
        )
        if hash_blk.get("preview_tokens") is not None:
            lines.append(f"- **Preview Tokens:** {hash_blk.get('preview_tokens'):,}")
        if hash_blk.get("final_tokens") is not None:
            lines.append(f"- **Final Tokens:** {hash_blk.get('final_tokens'):,}")
        if hash_blk.get("total_tokens") is not None:
            lines.append(f"- **Total Tokens:** {hash_blk.get('total_tokens'):,}")
        if hash_blk.get("dataset"):
            lines.append(f"- **Dataset Hash:** {hash_blk.get('dataset')}")
        hash_source = _dataset_hash_source_label(hash_blk.get("source"))
        if hash_source:
            lines.append(f"- **Hash Source:** {hash_source}")
        tokenizer = dataset.get("tokenizer", {})
        if isinstance(tokenizer, dict) and (
            tokenizer.get("name") or tokenizer.get("hash")
        ):
            vocab_size = tokenizer.get("vocab_size")
            vocab_suffix = (
                f" (vocab {vocab_size})" if isinstance(vocab_size, int) else ""
            )
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
                lines.append(
                    f"  - add_prefix_space: {tokenizer.get('add_prefix_space')}"
                )

    if has_provenance:
        baseline_info = provenance_info.get("baseline", {}) or {}
        edited_info = provenance_info.get("edited", {}) or {}

        if baseline_info or edited_info:
            lines.append("")
        if baseline_info:
            lines.append(f"- **Baseline Run ID:** {baseline_info.get('run_id')}")
            if baseline_info.get("report_hash"):
                lines.append(f"  - Report Hash: `{baseline_info.get('report_hash')}`")
            if baseline_info.get("report_path"):
                lines.append(f"  - Report Path: {baseline_info.get('report_path')}")
        if edited_info:
            lines.append(f"- **Edited Run ID:** {edited_info.get('run_id')}")
            if edited_info.get("report_hash"):
                lines.append(f"  - Report Hash: `{edited_info.get('report_hash')}`")
            if edited_info.get("report_path"):
                lines.append(f"  - Report Path: {edited_info.get('report_path')}")

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

        confidence = evaluation_report.get("confidence", {}) or {}
        if isinstance(confidence, dict) and confidence.get("label"):
            lines.append(f"- **Confidence:** {confidence.get('label')}")

        policy_digest = evaluation_report.get("policy_digest", {}) or {}
        if isinstance(policy_digest, dict) and policy_digest:
            policy_version = policy_digest.get("policy_version")
            thresholds_hash = policy_digest.get("thresholds_hash")
            if policy_version:
                lines.append(f"- **Policy Version:** {policy_version}")
            if isinstance(thresholds_hash, str) and thresholds_hash:
                lines.append(
                    f"- **Thresholds Digest:** `{_short_digest(thresholds_hash)}` (full in JSON)"
                )
            if policy_digest.get("changed"):
                lines.append("- Note: policy changed")

    lines.append("")


def _append_report_warning_banners(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    warnings: list[str] = []
    if _primary_metric_is_pseudo_accuracy(evaluation_report):
        warnings.append("ESTIMATED / PSEUDO ACCURACY — NOT MEASURED LABEL ACCURACY")
    if _is_non_assurance_report(evaluation_report):
        warnings.append("NON-ASSURANCE REPORT")
    if not warnings:
        return
    for warning in warnings:
        lines.append(f"> **{warning}**")
    lines.append("")


def _append_model_context_sections(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
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
    version = meta.get("invarlock_version")
    if version:
        lines.append(f"- **InvarLock Version:** {version}")
    env_flags = meta.get("env_flags")
    cuda_flags = meta.get("cuda_flags")

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
        try:
            if bool(auto.get("tiny_relax")):
                lines.append("- Tiny relax: enabled (dev-only)")
        except _MODEL_CONTEXT_PARSE_EXCEPTIONS:
            pass
        lines.append("")


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


def _append_plugin_provenance_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    plugins = evaluation_report.get("plugins", {})
    if not (isinstance(plugins, dict) and plugins):
        lines.append("")
        return

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


def render_report_markdown(evaluation_report: dict[str, Any]) -> str:
    """
    Render an evaluation report as a formatted Markdown report with pretty tables.

    Render an already-normalized evaluation report into Markdown.
    """
    lines: list[str] = []
    appendix_lines: list[str] = []

    _append_report_header(lines, evaluation_report)

    _append_report_warning_banners(lines, evaluation_report)

    _append_plugin_provenance_section(lines, evaluation_report)

    _append_executive_summary_section(lines, evaluation_report)

    _append_quality_gates_section(lines, evaluation_report)

    append_guard_check_details_section(lines, evaluation_report)

    _append_primary_metric_section(lines, evaluation_report)

    append_guard_observability_sections(lines, evaluation_report)

    _append_model_context_sections(lines, evaluation_report)

    _append_dataset_and_provenance_section(lines, evaluation_report)

    # Structural Changes heading is printed with content later; avoid empty header here

    sys_over = evaluation_report.get("system_overhead", {}) or {}
    if isinstance(sys_over, dict) and sys_over:
        _append_system_overhead_section(lines, sys_over)

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
    _append_structural_changes_section(lines, evaluation_report)

    structure = evaluation_report.get("structure", {}) or {}
    edit_name = str(evaluation_report.get("edit_name", "unknown"))
    if structure.get("bitwidths") and edit_name != "quant_rtn":
        lines.append(f"| Bit-width Changes | {len(structure['bitwidths'])} layers |")
    if structure.get("ranks"):
        lines.append(f"| Rank Changes | {len(structure['ranks'])} layers |")

    lines.append("")

    lines.append("")

    _append_compression_diagnostics_section(lines, evaluation_report)

    _append_moe_observability_section(lines, evaluation_report)

    _append_policy_configuration_section(lines, evaluation_report)

    _append_appendix_sections(lines, appendix_lines, evaluation_report)

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
