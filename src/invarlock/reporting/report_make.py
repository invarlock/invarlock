"""
InvarLock Evaluation Report Generation
=====================================

Generate standardized evaluation reports from RunReport and baseline
comparison.
Evaluation reports are standalone, portable artifacts that record statistical
gates and evidence for CI/CD checks and audits (not formal verification).
"""

from __future__ import annotations

# Core evaluation report building and analysis orchestration lives here.
import copy
from collections.abc import Callable
from typing import Any, cast

import invarlock.eval.primary_metric as primary_metric_mod
from invarlock.core.exceptions import ValidationError

from . import dataset_hashing as dataset_hashing_mod
from . import guards_invariants as guards_invariants_mod
from . import guards_spectral as guards_spectral_mod
from . import policy_utils as report_policy_utils_mod
from . import report_builder_support as report_builder_support_mod
from . import report_edit_summary as report_edit_summary_mod
from . import report_enrichment as report_enrichment_mod
from . import report_make_assembly as report_make_assembly_mod
from . import report_metric_impact as report_metric_impact_mod
from . import report_normalization as report_normalization_mod
from . import report_policy as report_policy_mod
from . import report_primary_metric_analysis as report_primary_metric_analysis_mod
from . import report_primary_metric_policy as report_primary_metric_policy_mod
from . import report_provenance as report_provenance_mod
from . import report_schema as report_schema_mod
from .guards_rmt import _extract_rmt_analysis
from .guards_variance import _extract_variance_analysis
from .report_build_context import EvaluationReportBuilder
from .report_types import RunReport

POLICY_VERSION = report_provenance_mod.POLICY_VERSION
REPORT_SCHEMA_VERSION = report_schema_mod.REPORT_SCHEMA_VERSION
REPORT_JSON_SCHEMA = report_schema_mod.REPORT_JSON_SCHEMA
TIER_RATIO_LIMITS = report_policy_mod.TIER_RATIO_LIMITS

# Canonical preview→final drift band used when not explicitly configured.
PM_DRIFT_BAND_DEFAULT: tuple[float, float] = (0.95, 1.05)

VARIANCE_CANONICAL_KEYS = (
    "deadband",
    "min_abs_adjust",
    "max_scale_step",
    "min_effect_lognll",
    "predictive_one_sided",
    "topk_backstop",
    "max_adjusted_modules",
)

_MAKE_REPORT_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    RuntimeError,
    OSError,
)


_TOP_LEVEL_GUARD_NAMES = frozenset({"spectral", "rmt", "variance", "invariants"})
_GUARD_OUTCOME_FIELDS = (
    "passed",
    "decision",
    "policy",
    "diagnostics",
    "violations",
    "details",
    "supported",
    "reason",
    "assurance_blocking",
    "status",
)


def _build_blocking_diagnostic_recorder(
    build_diagnostics: list[dict[str, Any]],
) -> tuple[dict[str, bool], Callable[[str, str], None]]:
    state = {"blocking": False}

    def _record_blocking_diagnostic(code: str, message: str) -> None:
        state["blocking"] = True
        report_builder_support_mod.append_build_diagnostic(
            build_diagnostics,
            code=code,
            message=message,
            severity="error",
        )

    return state, _record_blocking_diagnostic


def _normalize_make_report_inputs(
    report: RunReport,
    baseline: RunReport | dict[str, Any],
    *,
    non_fatal_exceptions: tuple[type[BaseException], ...],
) -> tuple[
    RunReport,
    dict[str, Any],
    RunReport | dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    RunReport | None,
]:
    normalized_report = report_normalization_mod.validated_run_report_view(report)
    report_map = cast(dict[str, Any], normalized_report)

    baseline_raw = baseline
    baseline_raw_map = cast(dict[str, Any], baseline_raw)
    try:
        baseline_normalized = report_normalization_mod.normalize_baseline(baseline_raw)
    except non_fatal_exceptions as exc:
        raise ValidationError(
            code="E231",
            message=(
                "Baseline normalization failed; evaluation report assembly "
                "requires a concrete finite baseline metric or valid baseline "
                "evaluation evidence."
            ),
            details={"error": str(exc)},
        ) from exc

    baseline_report: RunReport | None = None
    try:
        if (
            isinstance(baseline_raw, dict)
            and "meta" in baseline_raw
            and "metrics" in baseline_raw
            and "edit" in baseline_raw
        ):
            baseline_report = report_normalization_mod.validated_run_report_view(
                baseline_raw
            )
    except non_fatal_exceptions as exc:
        raise ValidationError(
            code="E232",
            message=(
                "Baseline report normalization failed; evaluation report assembly "
                "requires a valid baseline report."
            ),
            details={"error": str(exc)},
        ) from exc

    return (
        normalized_report,
        report_map,
        baseline_raw,
        baseline_raw_map,
        baseline_normalized,
        baseline_report,
    )


def _extract_report_build_sections(
    report: RunReport,
    report_map: dict[str, Any],
    baseline_raw: RunReport | dict[str, Any],
    baseline_normalized: dict[str, Any],
    baseline_report: RunReport | None,
    build_diagnostics: list[dict[str, Any]],
    *,
    record_blocking_diagnostic: Callable[[str, str], None],
    non_fatal_exceptions: tuple[type[BaseException], ...],
) -> dict[str, Any]:
    meta = report_builder_support_mod.extract_report_meta(report, build_diagnostics)
    report_make_assembly_mod._copy_meta_provenance_fields(
        report,
        meta,
        build_diagnostics,
        record_blocking_diagnostic=record_blocking_diagnostic,
        non_fatal_exceptions=non_fatal_exceptions,
    )

    auto_config = report["meta"].get("auto")
    if auto_config:
        auto: dict[str, Any] = {
            "tier": auto_config.get("tier", "balanced"),
            "probes_used": auto_config.get("probes", auto_config.get("probes_used", 0)),
            "target_pm_ratio": auto_config.get("target_pm_ratio"),
        }
    else:
        auto = {"tier": "none", "probes_used": 0, "target_pm_ratio": None}

    dataset_info = dataset_hashing_mod._extract_dataset_info(report_map)
    try:
        if isinstance(dataset_info, dict):
            windows = dataset_info.get("windows")
            if isinstance(windows, dict):
                windows.setdefault("stats", {})
    except non_fatal_exceptions:  # pragma: no cover
        report_builder_support_mod.append_build_diagnostic(
            build_diagnostics,
            code="dataset.windows_stats_unavailable",
            message="Dataset window statistics could not be initialized in the evaluation report.",
        )

    baseline_ref = report_builder_support_mod.build_baseline_reference(
        report,
        baseline_raw,
        baseline_normalized,
        compute_primary_metric_from_report_fn=primary_metric_mod.compute_primary_metric_from_report,
    )
    ppl_analysis, window_plan_profile = (
        report_primary_metric_analysis_mod.build_primary_metric_analysis(
            report_map,
            baseline_normalized,
            baseline_ref,
            dataset_info,
        )
    )
    ppl_metrics = report_map.get("metrics", {})

    invariants = guards_invariants_mod._extract_invariants(
        report,
        baseline=baseline_report,
    )
    spectral = guards_spectral_mod._extract_spectral_analysis(
        report,
        baseline_normalized,
    )
    rmt = _extract_rmt_analysis(report, baseline_normalized)
    variance = _extract_variance_analysis(report)

    structure = report_edit_summary_mod.extract_structural_deltas(report)
    compression_diag = structure.get("compression_diagnostics", {})
    structure["compression_diagnostics"] = compression_diag

    policies = report_policy_utils_mod._extract_effective_policies(report)
    variance_policy = policies.get("variance")
    guard_variance_policy = None
    for guard in report.get("guards", []):
        if guard.get("name", "").lower() == "variance" and isinstance(
            guard.get("policy"), dict
        ):
            guard_variance_policy = guard.get("policy")
            break

    variance_policy_digest = ""
    if isinstance(variance_policy, dict):
        variance_policy_digest = (
            report_policy_utils_mod._compute_variance_policy_digest(variance_policy)
        )
        if not variance_policy_digest and isinstance(guard_variance_policy, dict):
            variance_policy_digest = (
                report_policy_utils_mod._compute_variance_policy_digest(
                    guard_variance_policy
                )
            )
            if variance_policy_digest:
                for key in VARIANCE_CANONICAL_KEYS:
                    if (
                        isinstance(guard_variance_policy, dict)
                        and key in guard_variance_policy
                        and key not in variance_policy
                    ):
                        variance_policy[key] = guard_variance_policy[key]
        if variance_policy_digest:
            policies["variance"]["policy_digest"] = variance_policy_digest

    return {
        "meta": meta,
        "auto": auto,
        "dataset_info": dataset_info,
        "baseline_ref": baseline_ref,
        "ppl_analysis": ppl_analysis,
        "window_plan_profile": window_plan_profile,
        "ppl_metrics": ppl_metrics,
        "invariants": invariants,
        "spectral": spectral,
        "rmt": rmt,
        "variance": variance,
        "structure": structure,
        "policies": policies,
        "variance_policy_digest": variance_policy_digest,
    }


def _collect_guard_outcomes(guards: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(guards, list):
        return {}
    outcomes: dict[str, dict[str, Any]] = {}
    for entry in guards:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip().lower()
        if name not in _TOP_LEVEL_GUARD_NAMES:
            continue
        outcome = outcomes.setdefault(name, {})
        for field in _GUARD_OUTCOME_FIELDS:
            if field not in entry:
                continue
            value = entry[field]
            if field == "passed":
                if value is False or "passed" not in outcome:
                    outcome[field] = value
                continue
            if field == "decision":
                if value in {"block", "rollback"} or "decision" not in outcome:
                    outcome[field] = value
                continue
            outcome.setdefault(field, copy.deepcopy(value))
    return outcomes


def _attach_top_level_guard_outcomes(evaluation_report: dict[str, Any]) -> None:
    outcomes = _collect_guard_outcomes(evaluation_report.get("guards"))
    for guard_name, outcome in outcomes.items():
        section = evaluation_report.get(guard_name)
        if not isinstance(section, dict):
            continue
        for field, value in outcome.items():
            section.setdefault(field, copy.deepcopy(value))


def _build_evaluation_report(
    *,
    report_map: dict[str, Any],
    current_run_id: str,
    meta: dict[str, Any],
    auto: dict[str, Any],
    dataset_info: dict[str, Any],
    edit_metadata: dict[str, Any] | None,
    telemetry: dict[str, Any],
    baseline_ref: dict[str, Any],
    invariants: dict[str, Any],
    spectral: dict[str, Any],
    rmt: dict[str, Any],
    variance: dict[str, Any],
    structure: dict[str, Any],
    policies: dict[str, Any],
    resolved_policy: dict[str, Any],
    policy_provenance: dict[str, Any],
    policy_resolution: dict[str, Any],
    provenance: dict[str, Any],
    plugin_provenance: dict[str, Any],
    edit_name: str | None,
    artifacts_payload: dict[str, Any],
    validation_filtered: dict[str, Any],
    guard_warning_summary: dict[str, Any] | None = None,
    guard_metric_impact_section: dict[str, Any],
    pm_tail_result: dict[str, Any],
) -> dict[str, Any]:
    subject_ref: dict[str, Any] = {
        "model_id": meta.get("model_id"),
        "adapter": meta.get("adapter"),
    }
    model_identity = meta.get("model_identity")
    if model_identity is not None:
        subject_ref["model_identity"] = copy.deepcopy(model_identity)
    evaluation_report = {
        "schema_version": report_schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": current_run_id,
        "meta": meta,
        "subject_ref": subject_ref,
        "auto": auto,
        "dataset": dataset_info,
        "edit": edit_metadata,
        "telemetry": telemetry,
        "baseline_ref": baseline_ref,
        "invariants": invariants,
        "spectral": spectral,
        "rmt": rmt,
        "variance": variance,
        "structure": structure,
        "policies": policies,
        "resolved_policy": resolved_policy,
        "policy_provenance": policy_provenance,
        "policy_resolution": policy_resolution,
        "provenance": provenance,
        "plugins": plugin_provenance,
        "guards": (
            copy.deepcopy(report_map.get("guards", []))
            if isinstance(report_map.get("guards"), list)
            else []
        ),
        "artifacts": artifacts_payload,
        "validation": validation_filtered,
        "guard_warnings": guard_warning_summary
        or {"present": False, "warning_count": 0, "warnings": []},
        "guard_metric_impact": guard_metric_impact_section,
        "primary_metric_tail": pm_tail_result,
        "context": (
            copy.deepcopy(report_map.get("context", {}))
            if isinstance(report_map.get("context"), dict)
            else {}
        ),
        "evaluation_windows": (
            copy.deepcopy(report_map.get("evaluation_windows", {}))
            if isinstance(report_map.get("evaluation_windows"), dict)
            else {}
        ),
    }
    if isinstance(report_map.get("evaluation_realism"), dict):
        evaluation_report["evaluation_realism"] = copy.deepcopy(
            report_map["evaluation_realism"]
        )
    provider_digest = provenance.get("provider_digest")
    dataset_evidence = (
        provider_digest.get("dataset_evidence")
        if isinstance(provider_digest, dict)
        else None
    )
    if isinstance(dataset_evidence, dict):
        evaluation_report["dataset_evidence"] = copy.deepcopy(dataset_evidence)
    _attach_top_level_guard_outcomes(evaluation_report)
    if edit_name is not None:
        evaluation_report["edit_name"] = edit_name
    report_builder_support_mod.ensure_report_build_evidence(evaluation_report)
    return evaluation_report


def _finalize_evaluation_report(
    evaluation_report: dict[str, Any],
    *,
    report_map: dict[str, Any],
    report: RunReport,
    baseline_raw_map: dict[str, Any],
    baseline_normalized: dict[str, Any],
    baseline_ref: dict[str, Any],
    telemetry: dict[str, Any],
    resolved_policy: dict[str, Any],
    auto: dict[str, Any],
    policy_provenance: dict[str, Any],
    raw_guard_ctx: Any,
    ppl_analysis: dict[str, Any],
    window_plan_profile: Any,
    pm_drift_band: dict[str, Any] | None,
    tiny_relax: bool,
    current_run_id: str,
    build_diagnostics: list[dict[str, Any]],
    record_blocking_diagnostic,
    non_fatal_exceptions: tuple[type[BaseException], ...],
) -> None:
    if tiny_relax:
        try:
            auto_section = evaluation_report.setdefault("auto", {})
            if isinstance(auto_section, dict):
                auto_section["tiny_relax"] = True
            prov = evaluation_report.setdefault("provenance", {})
            if isinstance(prov, dict):
                flags = prov.setdefault("flags", [])
                if isinstance(flags, list) and "tiny_relax" not in flags:
                    flags.append("tiny_relax")
        except non_fatal_exceptions:  # pragma: no cover
            record_blocking_diagnostic(
                code="provenance.tiny_relax_flag_unavailable",
                message="Tiny-relax provenance could not be attached to the evaluation report.",
            )

    report_enrichment_mod.attach_guard_metric_impact(
        evaluation_report,
        raw_guard_ctx,
        report_map,
        report_metric_impact_mod.compute_guard_metric_impact_from_guard,
    )

    try:
        report_primary_metric_policy_mod.propagate_pairing_stats(
            evaluation_report, ppl_analysis
        )
    except non_fatal_exceptions:  # pragma: no cover
        record_blocking_diagnostic(
            code="pairing.stats_unavailable",
            message="Pairing statistics could not be propagated into the evaluation report.",
        )

    report_enrichment_mod.attach_policy_digest(
        evaluation_report,
        auto,
        resolved_policy,
        baseline_raw_map,
        baseline_normalized,
        report_policy_utils_mod._compute_thresholds_payload,
        report_policy_utils_mod._compute_thresholds_hash,
        report_provenance_mod.POLICY_VERSION,
    )
    report_enrichment_mod.attach_secondary_metrics(evaluation_report, report_map)
    report_enrichment_mod.attach_classification(evaluation_report, report_map)
    report_enrichment_mod.attach_system_overhead(
        evaluation_report,
        report_map,
        baseline_raw_map,
        telemetry,
    )

    from .primary_metric_utils import attach_primary_metric as _attach_pm

    _attach_pm(
        evaluation_report,
        report_map,
        baseline_raw_map,
        baseline_ref,
        ppl_analysis,
    )
    try:
        if isinstance(pm_drift_band, dict) and pm_drift_band:
            pm_block = evaluation_report.get("primary_metric")
            if isinstance(pm_block, dict):
                pm_block.setdefault("drift_band", dict(pm_drift_band))
    except non_fatal_exceptions:  # pragma: no cover
        record_blocking_diagnostic(
            code="primary_metric.drift_band_unavailable",
            message="Primary-metric drift-band metadata could not be attached to the evaluation report.",
        )
    report_primary_metric_policy_mod.enforce_display_ci_alignment(
        ppl_analysis.get("stats", {}).get("pairing", "run_metrics"),
        evaluation_report.get("primary_metric"),
        ppl_analysis.get("logloss_delta_ci"),
        window_plan_profile,
        evaluation_report=evaluation_report,
    )
    report_enrichment_mod.ensure_primary_metric_display_ci(evaluation_report)
    report_enrichment_mod.attach_telemetry_summary_line(
        evaluation_report, report_map, current_run_id
    )
    report_enrichment_mod.attach_confidence_label(
        evaluation_report, report_enrichment_mod.compute_confidence_label
    )
    if build_diagnostics:
        meta_section = evaluation_report.setdefault("meta", {})
        if isinstance(meta_section, dict):
            meta_section["build_diagnostics"] = build_diagnostics
    try:
        EvaluationReportBuilder(evaluation_report).finalize_assurance()
    except non_fatal_exceptions:
        record_blocking_diagnostic(
            code="assurance.section_unavailable",
            message="Assurance verdict metadata could not be attached to the evaluation report.",
        )


def make_report(
    report: RunReport,
    baseline: RunReport | dict[str, Any],
    *,
    provenance_env_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate an evaluation report from a RunReport and baseline comparison."""
    build_diagnostics: list[dict[str, Any]] = []
    blocking_state, record_blocking_diagnostic = _build_blocking_diagnostic_recorder(
        build_diagnostics
    )
    (
        report,
        report_map,
        baseline_raw,
        baseline_raw_map,
        baseline_normalized,
        baseline_report,
    ) = _normalize_make_report_inputs(
        report,
        baseline,
        non_fatal_exceptions=_MAKE_REPORT_NON_FATAL_EXCEPTIONS,
    )
    sections = _extract_report_build_sections(
        report,
        report_map,
        baseline_raw,
        baseline_normalized,
        baseline_report,
        build_diagnostics,
        record_blocking_diagnostic=record_blocking_diagnostic,
        non_fatal_exceptions=_MAKE_REPORT_NON_FATAL_EXCEPTIONS,
    )
    policy_context = (
        report_make_assembly_mod._resolve_policy_edit_and_telemetry_context(
            report,
            report_map,
            sections["meta"],
            sections["auto"],
            sections["spectral"],
            sections["rmt"],
            sections["variance"],
            sections["policies"],
            sections["variance_policy_digest"],
            build_diagnostics,
            non_fatal_exceptions=_MAKE_REPORT_NON_FATAL_EXCEPTIONS,
        )
    )
    assembly_context = report_make_assembly_mod._build_report_assembly_context(
        report,
        report_map,
        baseline_raw,
        baseline_raw_map,
        baseline_normalized,
        sections["baseline_ref"],
        sections["dataset_info"],
        policy_context["telemetry"],
        policy_context["policy_provenance"],
        sections["ppl_analysis"],
        policy_context["resolved_policy"],
        sections["auto"],
        sections["invariants"],
        sections["spectral"],
        sections["rmt"],
        sections["variance"],
        sections["ppl_metrics"],
        provenance_env_flags,
        blocking_state,
    )

    evaluation_report = _build_evaluation_report(
        report_map=report_map,
        current_run_id=assembly_context["current_run_id"],
        meta=sections["meta"],
        auto=sections["auto"],
        dataset_info=sections["dataset_info"],
        edit_metadata=policy_context["edit_metadata"],
        telemetry=policy_context["telemetry"],
        baseline_ref=sections["baseline_ref"],
        invariants=sections["invariants"],
        spectral=sections["spectral"],
        rmt=sections["rmt"],
        variance=sections["variance"],
        structure=sections["structure"],
        policies=sections["policies"],
        resolved_policy=policy_context["resolved_policy"],
        policy_provenance=policy_context["policy_provenance"],
        policy_resolution=policy_context["policy_resolution"],
        provenance=assembly_context["provenance"],
        plugin_provenance=policy_context["plugin_provenance"],
        edit_name=policy_context["edit_name"],
        artifacts_payload=assembly_context["artifacts_payload"],
        validation_filtered=assembly_context["validation_filtered"],
        guard_warning_summary=assembly_context["guard_warning_summary"],
        guard_metric_impact_section=assembly_context["guard_metric_impact_section"],
        pm_tail_result=assembly_context["pm_tail_result"],
    )
    _finalize_evaluation_report(
        evaluation_report,
        report_map=report_map,
        report=report,
        baseline_raw_map=baseline_raw_map,
        baseline_normalized=baseline_normalized,
        baseline_ref=sections["baseline_ref"],
        telemetry=policy_context["telemetry"],
        resolved_policy=policy_context["resolved_policy"],
        auto=sections["auto"],
        policy_provenance=policy_context["policy_provenance"],
        raw_guard_ctx=assembly_context["raw_guard_ctx"],
        ppl_analysis=sections["ppl_analysis"],
        window_plan_profile=sections["window_plan_profile"],
        pm_drift_band=assembly_context["pm_drift_band"],
        tiny_relax=assembly_context["tiny_relax"],
        current_run_id=assembly_context["current_run_id"],
        build_diagnostics=build_diagnostics,
        record_blocking_diagnostic=record_blocking_diagnostic,
        non_fatal_exceptions=_MAKE_REPORT_NON_FATAL_EXCEPTIONS,
    )

    return evaluation_report
