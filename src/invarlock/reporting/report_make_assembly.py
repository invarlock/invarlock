from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from invarlock.core import auto_tuning as auto_tuning_mod
from invarlock.eval import tail_stats as tail_stats_mod

from . import guard_warnings as guard_warnings_mod
from . import policy_utils as report_policy_utils_mod
from . import report_builder_support as report_builder_support_mod
from . import report_edit_summary as report_edit_summary_mod
from . import report_normalization as report_normalization_mod
from . import report_overhead as report_overhead_mod
from . import report_policy as report_policy_mod
from . import report_provenance as report_provenance_mod
from . import report_validation as report_validation_mod
from .report_schema import load_validation_allowlist
from .report_types import RunReport


def _copy_meta_provenance_fields(
    report: RunReport,
    meta: dict[str, Any],
    build_diagnostics: list[dict[str, Any]],
    *,
    record_blocking_diagnostic,
    non_fatal_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        env_flags = (
            report.get("meta", {}).get("env_flags")
            if isinstance(report.get("meta"), dict)
            else None
        )
        if isinstance(env_flags, dict) and env_flags:
            meta["env_flags"] = env_flags
    except non_fatal_exceptions:  # pragma: no cover
        record_blocking_diagnostic(
            code="meta.env_flags_unavailable",
            message="Environment flag provenance could not be copied into the evaluation report.",
        )

    try:
        det = (
            report.get("meta", {}).get("determinism")
            if isinstance(report.get("meta"), dict)
            else None
        )
        if isinstance(det, dict) and det:
            meta["determinism"] = det
    except non_fatal_exceptions:  # pragma: no cover
        record_blocking_diagnostic(
            code="meta.determinism_unavailable",
            message="Determinism provenance could not be copied into the evaluation report.",
        )

    try:
        ctx = report.get("context") if isinstance(report, dict) else None
        ctx_profile = (
            str(ctx.get("profile") or "").strip().lower()
            if isinstance(ctx, dict)
            else ""
        )
        if ctx_profile:
            meta["profile"] = ctx_profile
    except non_fatal_exceptions:  # pragma: no cover
        record_blocking_diagnostic(
            code="meta.profile_unavailable",
            message="Execution profile provenance could not be copied into the evaluation report.",
        )

    tokenizer_hash_meta = report["meta"].get("tokenizer_hash")
    if not tokenizer_hash_meta:
        dataset_section = report.get("data", {})
        if isinstance(dataset_section, dict):
            tokenizer_hash_meta = dataset_section.get("tokenizer_hash")
    if isinstance(tokenizer_hash_meta, str) and tokenizer_hash_meta:
        meta["tokenizer_hash"] = tokenizer_hash_meta

    model_profile_meta = report["meta"].get("model_profile")
    if isinstance(model_profile_meta, dict) and model_profile_meta:
        meta["model_profile"] = model_profile_meta

    cuda_flags = report["meta"].get("cuda_flags")
    if isinstance(cuda_flags, dict) and cuda_flags:
        meta["cuda_flags"] = cuda_flags


def _resolve_policy_inputs(
    report: RunReport,
    build_diagnostics: list[dict[str, Any]],
    *,
    non_fatal_exceptions: tuple[type[BaseException], ...],
) -> tuple[str | None, dict[str, Any] | None]:
    profile = None
    explicit_overrides: dict[str, Any] | None = None
    try:
        ctx = report.get("context") if isinstance(report, dict) else None
        if isinstance(ctx, dict) and ctx.get("profile"):
            profile = str(ctx.get("profile"))
    except non_fatal_exceptions:
        profile = None
        report_builder_support_mod.append_build_diagnostic(
            build_diagnostics,
            code="policy.profile_from_context_failed",
            message="Profile extraction from run context failed; policy resolution fell back to default profile handling.",
            severity="error",
        )
    try:
        window_plan = (
            report.get("metrics", {}).get("window_plan")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        if (
            profile is None
            and isinstance(window_plan, dict)
            and window_plan.get("profile")
        ):
            profile = str(window_plan.get("profile"))
    except non_fatal_exceptions:
        profile = None
        report_builder_support_mod.append_build_diagnostic(
            build_diagnostics,
            code="policy.profile_from_window_plan_failed",
            message="Window-plan profile extraction failed; policy resolution fell back to context/default profile handling.",
            severity="error",
        )
    try:
        meta_cfg = (
            report.get("meta", {}).get("config")
            if isinstance(report.get("meta"), dict)
            else None
        )
        if isinstance(meta_cfg, dict) and isinstance(meta_cfg.get("guards"), dict):
            explicit_overrides = meta_cfg.get("guards")
        cfg2 = report.get("config")
        if explicit_overrides is None and isinstance(cfg2, dict):
            if isinstance(cfg2.get("guards"), dict):
                explicit_overrides = cfg2.get("guards")
    except non_fatal_exceptions:
        explicit_overrides = None
        report_builder_support_mod.append_build_diagnostic(
            build_diagnostics,
            code="policy.explicit_overrides_unavailable",
            message="Explicit guard overrides could not be extracted from the run configuration.",
            severity="error",
        )
    return profile, explicit_overrides


def _resolve_policy_edit_and_telemetry_context(
    report: RunReport,
    report_map: dict[str, Any],
    meta: dict[str, Any],
    auto: dict[str, Any],
    spectral: dict[str, Any],
    rmt: dict[str, Any],
    variance: dict[str, Any],
    policies: dict[str, Any],
    variance_policy_digest: str,
    build_diagnostics: list[dict[str, Any]],
    *,
    record_blocking_diagnostic: Callable[[str, str], None],
    non_fatal_exceptions: tuple[type[BaseException], ...],
) -> dict[str, Any]:
    load_validation_allowlist()
    validation_allowlist_source = "contracts"
    profile, explicit_overrides = _resolve_policy_inputs(
        report,
        build_diagnostics,
        non_fatal_exceptions=non_fatal_exceptions,
    )

    resolved_policy = report_policy_utils_mod._build_resolved_policies(
        auto.get("tier", "balanced"),
        spectral,
        rmt,
        variance,
        profile=profile,
        explicit_overrides=explicit_overrides,
    )
    overrides_list = report_policy_utils_mod._extract_policy_overrides(report)
    resolved_digest = report_policy_utils_mod._compute_policy_digest(
        {
            "resolved_policy": resolved_policy,
            "overrides": overrides_list,
        }
    )
    policy_provenance = {
        "tier": auto.get("tier", "balanced"),
        "overrides": overrides_list,
        "policy_digest": resolved_digest,
        "validation_allowlist_source": validation_allowlist_source,
    }
    auto["policy_digest"] = resolved_digest

    for guard_name in ("spectral", "rmt", "variance"):
        if guard_name in resolved_policy:
            policies[guard_name] = copy.deepcopy(resolved_policy[guard_name])
            if guard_name == "variance" and variance_policy_digest:
                policies[guard_name]["policy_digest"] = variance_policy_digest

    plugin_provenance: dict[str, Any] = {}
    meta_plugins = report_map.get("meta")
    if isinstance(meta_plugins, dict):
        raw_plugin_provenance = meta_plugins.get("plugins")
        if isinstance(raw_plugin_provenance, dict):
            plugin_provenance = raw_plugin_provenance

    edit_metadata = report_edit_summary_mod.extract_edit_metadata(
        report, plugin_provenance
    )
    edit_section = report.get("edit") if isinstance(report, dict) else {}
    edit_name = (
        report_builder_support_mod.optional_text(edit_section.get("name"))
        if isinstance(edit_section, dict)
        else None
    )
    if isinstance(edit_metadata, dict) and edit_name is None:
        edit_metadata.pop("name", None)

    telemetry = report_builder_support_mod.extract_telemetry(report, meta.get("device"))

    return {
        "profile": profile,
        "resolved_policy": resolved_policy,
        "policy_provenance": policy_provenance,
        "plugin_provenance": plugin_provenance,
        "edit_metadata": edit_metadata,
        "edit_name": edit_name,
        "telemetry": telemetry,
    }


def _build_report_assembly_context(
    report: RunReport,
    report_map: dict[str, Any],
    baseline_raw: RunReport | dict[str, Any],
    baseline_raw_map: dict[str, Any],
    baseline_normalized: dict[str, Any],
    baseline_ref: dict[str, Any],
    dataset_info: dict[str, Any],
    telemetry: dict[str, Any],
    policy_provenance: dict[str, Any],
    ppl_analysis: dict[str, Any],
    resolved_policy: dict[str, Any],
    auto: dict[str, Any],
    invariants: dict[str, Any],
    spectral: dict[str, Any],
    rmt: dict[str, Any],
    variance: dict[str, Any],
    ppl_metrics: dict[str, Any] | Any,
    provenance_env_flags: dict[str, Any] | None,
    blocking_state: dict[str, bool],
) -> dict[str, Any]:
    evaluate_metric_tail = tail_stats_mod.evaluate_metric_tail
    window_capacity_ctx = (
        report.get("metrics", {}).get("window_capacity")
        if isinstance(report.get("metrics"), dict)
        else None
    )

    artifacts_payload = report_builder_support_mod.build_artifacts_payload(report)
    raw_guard_ctx = report.get("guard_overhead")
    guard_overhead_section, _ = report_overhead_mod.prepare_guard_overhead_section(
        raw_guard_ctx
    )
    schedule_digest = report_builder_support_mod.attach_schedule_digest(
        report, guard_overhead_section
    )

    policy_provenance["resolved_at"] = artifacts_payload["generated_at"]

    current_run_id = report_normalization_mod._generate_run_id(report)
    provenance = report_provenance_mod.build_provenance_block(
        report_map,
        baseline_raw_map,
        baseline_ref,
        artifacts_payload,
        policy_provenance,
        schedule_digest,
        ppl_analysis,
        current_run_id,
        compute_report_digest_fn=report_provenance_mod.compute_report_digest,
        collect_backend_versions_fn=report_provenance_mod.collect_backend_versions,
        compute_edit_digest_fn=report_provenance_mod.compute_edit_digest,
        env_flags_payload=provenance_env_flags,
    )

    moe_section = report_builder_support_mod.build_moe_section(
        report, baseline_raw, baseline_normalized
    )
    capacity_tokens, capacity_examples = (
        report_builder_support_mod.resolve_capacity_context(
            window_capacity_ctx,
            dataset_info,
        )
    )

    pm_acceptance_range = report_policy_mod.resolve_pm_acceptance_range_from_report(
        report_map,
    )
    pm_drift_band = report_policy_mod.resolve_pm_drift_band_from_report(
        report_map,
        drift_band_default=report_policy_mod.PM_DRIFT_BAND_DEFAULT,
    )
    tiny_relax = report_policy_mod.resolve_tiny_relax_from_report(report_map)
    pm_tail_result = report_builder_support_mod.evaluate_primary_metric_tail(
        report,
        baseline_normalized,
        resolved_policy,
        evaluate_metric_tail,
    )

    target_ratio_raw = auto.get("target_pm_ratio")
    target_ratio = (
        float(target_ratio_raw) if isinstance(target_ratio_raw, int | float) else None
    )
    validation_flags = report_validation_mod.compute_validation_flags(
        ppl=ppl_analysis,
        spectral=spectral,
        rmt=rmt,
        invariants=invariants,
        tier=str(auto.get("tier", "balanced")),
        _ppl_metrics=ppl_metrics if isinstance(ppl_metrics, dict) else None,
        target_ratio=target_ratio,
        guard_overhead=guard_overhead_section,
        primary_metric=(
            report_map.get("metrics", {}).get("primary_metric")
            if isinstance(report_map.get("metrics"), dict)
            else None
        ),
        moe=moe_section,
        dataset_capacity={
            "tokens_available": capacity_tokens,
            "examples_available": capacity_examples,
        },
        pm_acceptance_range=pm_acceptance_range,
        pm_drift_band=pm_drift_band,
        pm_tail=pm_tail_result,
        tiny_relax=tiny_relax,
        pm_drift_band_default=report_policy_mod.PM_DRIFT_BAND_DEFAULT,
        get_tier_policies_fn=auto_tuning_mod.get_tier_policies,
    )
    baseline_warning_context = dict(baseline_normalized)
    if isinstance(baseline_raw_map.get("guards"), list):
        baseline_warning_context["guards"] = baseline_raw_map.get("guards")
    if isinstance(baseline_raw_map.get("metrics"), dict):
        baseline_warning_context["metrics"] = baseline_raw_map.get("metrics")

    guard_warning_summary = guard_warnings_mod.build_guard_warnings(
        subject={
            "spectral": spectral,
            "rmt": rmt,
            "invariants": invariants,
            "variance": variance,
        },
        baseline=baseline_warning_context,
        validation=validation_flags,
    )
    validation_flags["guard_warnings_present"] = bool(
        guard_warning_summary.get("present", False)
    )
    validation_flags["guard_warning_policy_acceptable"] = True

    _allowed_validation = load_validation_allowlist()
    validation_filtered = {
        k: bool(v) for k, v in validation_flags.items() if k in _allowed_validation
    }
    if blocking_state["blocking"]:
        validation_filtered["primary_metric_acceptable"] = False

    return {
        "artifacts_payload": artifacts_payload,
        "raw_guard_ctx": raw_guard_ctx,
        "guard_overhead_section": guard_overhead_section,
        "current_run_id": current_run_id,
        "provenance": provenance,
        "moe_section": moe_section,
        "capacity_tokens": capacity_tokens,
        "capacity_examples": capacity_examples,
        "pm_acceptance_range": pm_acceptance_range,
        "pm_drift_band": pm_drift_band,
        "tiny_relax": tiny_relax,
        "pm_tail_result": pm_tail_result,
        "guard_warning_summary": guard_warning_summary,
        "validation_filtered": validation_filtered,
    }
