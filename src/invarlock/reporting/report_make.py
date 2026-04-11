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
from typing import Any

from . import report_make_assembly as report_make_assembly_mod
from . import report_make_inputs as report_make_inputs_mod
from . import report_make_output as report_make_output_mod
from . import report_policy as report_policy_mod
from . import report_provenance as report_provenance_mod
from . import report_schema as report_schema_mod
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


def make_report(
    report: RunReport,
    baseline: RunReport | dict[str, Any],
    *,
    provenance_env_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate an evaluation report from a RunReport and baseline comparison."""
    build_diagnostics: list[dict[str, Any]] = []
    blocking_state, record_blocking_diagnostic = (
        report_make_inputs_mod._build_blocking_diagnostic_recorder(build_diagnostics)
    )
    (
        report,
        report_map,
        baseline_raw,
        baseline_raw_map,
        baseline_normalized,
        baseline_report,
    ) = report_make_inputs_mod._normalize_make_report_inputs(
        report,
        baseline,
        non_fatal_exceptions=_MAKE_REPORT_NON_FATAL_EXCEPTIONS,
    )
    sections = report_make_inputs_mod._extract_report_build_sections(
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
            record_blocking_diagnostic=record_blocking_diagnostic,
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
        sections["ppl_metrics"],
        provenance_env_flags,
        blocking_state,
    )

    evaluation_report = report_make_output_mod._build_evaluation_report(
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
        provenance=assembly_context["provenance"],
        plugin_provenance=policy_context["plugin_provenance"],
        edit_name=policy_context["edit_name"],
        artifacts_payload=assembly_context["artifacts_payload"],
        validation_filtered=assembly_context["validation_filtered"],
        guard_overhead_section=assembly_context["guard_overhead_section"],
        pm_tail_result=assembly_context["pm_tail_result"],
    )
    report_make_output_mod._finalize_evaluation_report(
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
