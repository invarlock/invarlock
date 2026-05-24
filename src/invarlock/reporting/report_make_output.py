from __future__ import annotations

import copy
from typing import Any

from invarlock.core.assurance_contract import (
    build_assurance_section,
    resolve_report_runtime_provenance_declared,
)

from . import policy_utils as report_policy_utils_mod
from . import report_build_evidence as report_build_evidence_mod
from . import report_confidence as report_confidence_mod
from . import report_enrichment as report_enrichment_mod
from . import report_overhead as report_overhead_mod
from . import report_primary_metric_policy as report_primary_metric_policy_mod
from . import report_provenance as report_provenance_mod
from . import report_schema as report_schema_mod
from .report_types import RunReport


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
    provenance: dict[str, Any],
    plugin_provenance: dict[str, Any],
    edit_name: str | None,
    artifacts_payload: dict[str, Any],
    validation_filtered: dict[str, bool],
    guard_overhead_section: dict[str, Any],
    pm_tail_result: dict[str, Any],
) -> dict[str, Any]:
    evaluation_report = {
        "schema_version": report_schema_mod.REPORT_SCHEMA_VERSION,
        "run_id": current_run_id,
        "meta": meta,
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
        "provenance": provenance,
        "plugins": plugin_provenance,
        "guards": (
            copy.deepcopy(report_map.get("guards", []))
            if isinstance(report_map.get("guards"), list)
            else []
        ),
        "artifacts": artifacts_payload,
        "validation": validation_filtered,
        "guard_overhead": guard_overhead_section,
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
    if edit_name is not None:
        evaluation_report["edit_name"] = edit_name
    report_build_evidence_mod.ensure_report_build_evidence(evaluation_report)
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

    report_enrichment_mod.attach_quality_overhead(
        evaluation_report,
        raw_guard_ctx,
        report_map,
        report_overhead_mod.compute_quality_overhead_from_guard,
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
        evaluation_report, report_confidence_mod.compute_confidence_label
    )
    if build_diagnostics:
        meta_section = evaluation_report.setdefault("meta", {})
        if isinstance(meta_section, dict):
            meta_section["build_diagnostics"] = build_diagnostics
    try:
        report_build_evidence_mod.ensure_report_build_evidence(evaluation_report)
        fallback_fields_used = (
            report_build_evidence_mod.report_build_has_evidence_events(
                evaluation_report
            )
        )
        evaluation_report["assurance"] = build_assurance_section(
            evaluation_report,
            fallback_fields_used=fallback_fields_used,
            runtime_provenance_verified=None,
            runtime_provenance_declared=resolve_report_runtime_provenance_declared(
                evaluation_report
            ),
            runtime_provenance_verification_status="pending",
        )
    except non_fatal_exceptions:
        record_blocking_diagnostic(
            code="assurance.section_unavailable",
            message="Assurance verdict metadata could not be attached to the evaluation report.",
        )
