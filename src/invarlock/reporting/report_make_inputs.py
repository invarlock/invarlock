from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import invarlock.eval.primary_metric as primary_metric_mod
from invarlock.core.exceptions import ValidationError

from . import dataset_hashing as dataset_hashing_mod
from . import guards_invariants as guards_invariants_mod
from . import guards_rmt as guards_rmt_mod
from . import guards_spectral as guards_spectral_mod
from . import guards_variance as guards_variance_mod
from . import policy_utils as report_policy_utils_mod
from . import report_builder_support as report_builder_support_mod
from . import report_edit_summary as report_edit_summary_mod
from . import report_make_assembly as report_make_assembly_mod
from . import report_normalization as report_normalization_mod
from . import report_primary_metric_analysis as report_primary_metric_analysis_mod
from .report_types import RunReport

VARIANCE_CANONICAL_KEYS = (
    "deadband",
    "min_abs_adjust",
    "max_scale_step",
    "min_effect_lognll",
    "predictive_one_sided",
    "topk_backstop",
    "max_adjusted_modules",
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
    normalized_report = report_normalization_mod.normalize_and_validate_run_report(
        report
    )
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
            baseline_report = (
                report_normalization_mod.normalize_and_validate_run_report(baseline_raw)
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
    rmt = guards_rmt_mod._extract_rmt_analysis(report, baseline_normalized)
    variance = guards_variance_mod._extract_variance_analysis(report)

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
