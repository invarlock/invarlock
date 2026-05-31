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
import math
from collections.abc import Callable
from typing import Any, cast

import invarlock.eval.primary_metric as primary_metric_mod
from invarlock.core.auto_tuning import get_tier_policies
from invarlock.core.exceptions import ValidationError

from . import dataset_hashing as dataset_hashing_mod
from . import guards_invariants as guards_invariants_mod
from . import guards_spectral as guards_spectral_mod
from . import policy_utils as report_policy_utils_mod
from . import report_builder_support as report_builder_support_mod
from . import report_edit_summary as report_edit_summary_mod
from . import report_enrichment as report_enrichment_mod
from . import report_make_assembly as report_make_assembly_mod
from . import report_normalization as report_normalization_mod
from . import report_overhead as report_overhead_mod
from . import report_policy as report_policy_mod
from . import report_primary_metric_analysis as report_primary_metric_analysis_mod
from . import report_primary_metric_policy as report_primary_metric_policy_mod
from . import report_provenance as report_provenance_mod
from . import report_schema as report_schema_mod
from .report_builder_support import EvaluationReportBuilder
from .report_types import RunReport
from .verify_check_helpers_consistency import (
    _baseline_guard_payload,
    _measurement_contract_digest,
)

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
_GUARD_PARSE_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
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


def _to_float_or_none(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    try:
        return float(value)
    except _GUARD_PARSE_EXCEPTIONS:
        return None


def _extract_rmt_analysis(
    report: RunReport, baseline: dict[str, Any]
) -> dict[str, Any]:
    """Extract RMT analysis using activation edge-risk epsilon-band semantics."""
    tier = report_policy_utils_mod._resolve_policy_tier(report)
    tier_policies = get_tier_policies()
    tier_defaults = tier_policies.get(tier, tier_policies.get("balanced", {}))

    default_epsilon_map = (
        tier_defaults.get("rmt", {}).get("epsilon_by_family")
        if isinstance(tier_defaults, dict)
        else {}
    )
    default_epsilon_map = {
        str(family): float(value)
        for family, value in (default_epsilon_map or {}).items()
        if isinstance(value, int | float) and math.isfinite(float(value))
    }

    epsilon_default = 0.1
    try:
        eps_def = (
            tier_defaults.get("rmt", {}).get("epsilon_default")
            if isinstance(tier_defaults, dict)
            else None
        )
        if isinstance(eps_def, int | float) and math.isfinite(float(eps_def)):
            epsilon_default = float(eps_def)
    except _GUARD_PARSE_EXCEPTIONS:
        pass

    baseline_rmt = _baseline_guard_payload(baseline, "rmt")
    baseline_edge_by_family: dict[str, float] = {}
    baseline_contract = None
    if isinstance(baseline_rmt, dict) and baseline_rmt:
        bc = baseline_rmt.get("measurement_contract")
        if isinstance(bc, dict) and bc:
            baseline_contract = bc
        base = baseline_rmt.get("edge_risk_by_family") or baseline_rmt.get(
            "edge_risk_by_family_base"
        )
        if isinstance(base, dict):
            for k, v in base.items():
                if isinstance(v, int | float) and math.isfinite(float(v)):
                    baseline_edge_by_family[str(k)] = float(v)

    rmt_guard = None
    guard_metrics: dict[str, Any] = {}
    guard_policy: dict[str, Any] = {}
    for guard in report.get("guards", []) or []:
        if str(guard.get("name", "")).lower() == "rmt":
            rmt_guard = guard
            guard_metrics = guard.get("metrics", {}) or {}
            guard_policy = guard.get("policy", {}) or {}
            break

    policy_out: dict[str, Any] | None = None
    if isinstance(guard_policy, dict) and guard_policy:
        policy_out = dict(guard_policy)
        epsilon_default_value = _to_float_or_none(policy_out.get("epsilon_default"))
        if epsilon_default_value is not None and math.isfinite(epsilon_default_value):
            epsilon_default = epsilon_default_value

    metric_epsilon_default = _to_float_or_none(guard_metrics.get("epsilon_default"))
    if metric_epsilon_default is not None and math.isfinite(metric_epsilon_default):
        epsilon_default = metric_epsilon_default

    edge_base: dict[str, float] = {}
    edge_cur: dict[str, float] = {}
    if isinstance(guard_metrics, dict) and guard_metrics:
        base = guard_metrics.get("edge_risk_by_family_base") or {}
        cur = guard_metrics.get("edge_risk_by_family") or {}
        if isinstance(base, dict):
            for k, v in base.items():
                if isinstance(v, int | float) and math.isfinite(float(v)):
                    edge_base[str(k)] = float(v)
        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(v, int | float) and math.isfinite(float(v)):
                    edge_cur[str(k)] = float(v)
    if not edge_base and baseline_edge_by_family:
        edge_base = dict(baseline_edge_by_family)

    epsilon_map: dict[str, float] = {}
    eps_src = guard_metrics.get("epsilon_by_family") or {}
    if not eps_src and isinstance(guard_policy, dict):
        eps_src = guard_policy.get("epsilon_by_family") or {}
    if isinstance(eps_src, dict):
        for k, v in eps_src.items():
            if isinstance(v, int | float) and math.isfinite(float(v)):
                epsilon_map[str(k)] = float(v)

    epsilon_violations = guard_metrics.get("epsilon_violations") or []
    if not (isinstance(epsilon_violations, list) and epsilon_violations):
        epsilon_violations = []
        families = set(edge_cur) | set(edge_base)
        for family in families:
            base = float(edge_base.get(family, 0.0) or 0.0)
            cur = float(edge_cur.get(family, 0.0) or 0.0)
            if base <= 0.0:
                continue
            eps_value = _to_float_or_none(
                epsilon_map.get(
                    family,
                    default_epsilon_map.get(family, epsilon_default),
                )
            )
            eps = epsilon_default if eps_value is None else eps_value
            threshold = (1.0 + eps) * base
            if cur > threshold:
                delta_ratio = (cur / base) - 1.0
                epsilon_violations.append(
                    {
                        "family": family,
                        "edge_base": base,
                        "edge_cur": cur,
                        "delta": float(delta_ratio),
                        "allowed": threshold,
                        "epsilon": eps,
                    }
                )

    stable = bool(guard_metrics.get("stable", not epsilon_violations))

    families_all = sorted(
        set(edge_base) | set(edge_cur) | set(epsilon_map) | set(default_epsilon_map)
    )
    family_breakdown: dict[str, dict[str, Any]] = {}
    ratios: list[float] = []
    deltas: list[float] = []
    for family in families_all:
        base = float(edge_base.get(family, 0.0) or 0.0)
        cur = float(edge_cur.get(family, 0.0) or 0.0)
        eps_value = _to_float_or_none(
            epsilon_map.get(family, default_epsilon_map.get(family, epsilon_default))
        )
        eps = epsilon_default if eps_value is None else eps_value
        allowed: float | None = (1.0 + eps) * base if base > 0.0 else None
        ratio: float | None = (cur / base) if base > 0.0 else None
        delta: float | None = ((cur / base) - 1.0) if base > 0.0 else None
        if isinstance(ratio, float) and math.isfinite(ratio):
            ratios.append(ratio)
        if isinstance(delta, float) and math.isfinite(delta):
            deltas.append(delta)
        family_breakdown[family] = {
            "edge_base": base,
            "edge_cur": cur,
            "epsilon": eps,
            "allowed": allowed,
            "ratio": ratio,
            "delta": delta,
        }

    measurement_contract = None
    try:
        mc = (
            guard_metrics.get("measurement_contract")
            if isinstance(guard_metrics, dict)
            else None
        )
        if isinstance(mc, dict) and mc:
            measurement_contract = mc
    except _GUARD_PARSE_EXCEPTIONS:
        measurement_contract = None

    mc_hash = _measurement_contract_digest(measurement_contract)
    baseline_hash = _measurement_contract_digest(baseline_contract)

    result: dict[str, Any] = {
        "tier": tier,
        "edge_risk_by_family_base": dict(edge_base),
        "edge_risk_by_family": dict(edge_cur),
        "epsilon_default": float(epsilon_default),
        "epsilon_by_family": dict(epsilon_map),
        "epsilon_violations": list(epsilon_violations),
        "stable": stable,
        "status": "stable" if stable else "unstable",
        "max_edge_ratio": max(ratios) if ratios else None,
        "max_edge_delta": max(deltas) if deltas else None,
        "mean_edge_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "families": family_breakdown,
        "evaluated": bool(rmt_guard),
    }
    if policy_out:
        result["policy"] = policy_out
    if measurement_contract is not None:
        result["measurement_contract"] = measurement_contract
    mode = None
    if isinstance(measurement_contract, dict):
        raw_mode = measurement_contract.get("kind")
        if isinstance(raw_mode, str) and raw_mode.strip():
            mode = raw_mode.strip()
    if mode is None and result["evaluated"]:
        mode = "activation_edge_risk"
    if mode is not None:
        result["mode"] = mode
    if mc_hash:
        result["measurement_contract_hash"] = mc_hash
    if baseline_hash:
        result["baseline_measurement_contract_hash"] = baseline_hash
    if mc_hash and baseline_hash:
        result["measurement_contract_match"] = bool(mc_hash == baseline_hash)
    return result


def _extract_variance_analysis(report: RunReport) -> dict[str, Any]:
    ve_enabled = False
    gain = None
    ppl_no_ve = None
    ppl_with_ve = None
    ratio_ci: Any = None
    calibration: dict[str, Any] = {}
    guard_metrics: dict[str, Any] = {}
    guard_policy: dict[str, Any] | None = None
    for guard in report.get("guards", []) or []:
        if "variance" in str(guard.get("name", "")).lower():
            metrics = guard.get("metrics", {}) or {}
            guard_metrics = metrics if isinstance(metrics, dict) else {}
            gp = guard.get("policy", {}) or {}
            if isinstance(gp, dict) and gp:
                guard_policy = dict(gp)
            ve_enabled = bool(guard_metrics.get("ve_enabled", bool(guard_metrics)))
            gain = guard_metrics.get("ab_gain", guard_metrics.get("gain", None))
            ppl_no_ve = guard_metrics.get("ppl_no_ve", None)
            ppl_with_ve = guard_metrics.get("ppl_with_ve", None)
            ratio_ci = guard_metrics.get("ratio_ci", ratio_ci)
            calibration_candidate = guard_metrics.get("calibration", calibration)
            if isinstance(calibration_candidate, dict):
                calibration = calibration_candidate
            break
    if gain is None:
        metrics_variance = (report.get("metrics", {}) or {}).get("variance", {})
        if isinstance(metrics_variance, dict):
            ve_enabled = metrics_variance.get("ve_enabled", ve_enabled)
            gain = metrics_variance.get("gain", gain)
            ppl_no_ve = metrics_variance.get("ppl_no_ve", ppl_no_ve)
            ppl_with_ve = metrics_variance.get("ppl_with_ve", ppl_with_ve)
            if not guard_metrics:
                guard_metrics = metrics_variance
    result: dict[str, Any] = {"enabled": ve_enabled, "gain": gain}
    if isinstance(ratio_ci, tuple | list) and len(ratio_ci) == 2:
        try:
            result["ratio_ci"] = (float(ratio_ci[0]), float(ratio_ci[1]))
        except _GUARD_PARSE_EXCEPTIONS:
            pass
    if calibration:
        result["calibration"] = calibration
    if not ve_enabled and ppl_no_ve is not None and ppl_with_ve is not None:
        result["ppl_no_ve"] = ppl_no_ve
        result["ppl_with_ve"] = ppl_with_ve
    metadata_fields = [
        "tap",
        "target_modules",
        "target_module_names",
        "focus_modules",
        "scope",
        "proposed_scales",
        "proposed_scales_pre_edit",
        "proposed_scales_post_edit",
        "monitor_only",
        "max_calib_used",
        "mode",
        "min_rel_gain",
        "alpha",
    ]
    for field in metadata_fields:
        value = guard_metrics.get(field)
        if value not in (None, {}, []):
            result[field] = value
    predictive_gate = guard_metrics.get("predictive_gate")
    if predictive_gate:
        result["predictive_gate"] = predictive_gate
    ab_section: dict[str, Any] = {}
    if guard_metrics.get("ab_seed_used") is not None:
        ab_section["seed"] = guard_metrics["ab_seed_used"]
    if guard_metrics.get("ab_windows_used") is not None:
        ab_section["windows_used"] = guard_metrics["ab_windows_used"]
    if guard_metrics.get("ab_provenance"):
        prov = guard_metrics["ab_provenance"]
        if isinstance(prov, dict):
            prov_out = dict(prov)

            if "window_ids" not in prov_out:
                window_ids: set[int] = set()

                def _collect(node: Any) -> None:
                    if isinstance(node, dict):
                        ids = node.get("window_ids")
                        if isinstance(ids, list):
                            for wid in ids:
                                if isinstance(wid, int | float):
                                    window_ids.add(int(wid))
                        for value in node.values():
                            _collect(value)
                        return
                    if isinstance(node, list):
                        for value in node:
                            _collect(value)

                _collect(prov_out)
                if window_ids:
                    prov_out["window_ids"] = sorted(window_ids)

            ab_section["provenance"] = prov_out
        else:
            ab_section["provenance"] = prov
    if guard_metrics.get("ab_point_estimates"):
        ab_section["point_estimates"] = guard_metrics["ab_point_estimates"]
    if ab_section:
        result["ab_test"] = ab_section
    if guard_policy:
        result["policy"] = guard_policy
    return result


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
    provenance: dict[str, Any],
    plugin_provenance: dict[str, Any],
    edit_name: str | None,
    artifacts_payload: dict[str, Any],
    validation_filtered: dict[str, Any],
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
        provenance=assembly_context["provenance"],
        plugin_provenance=policy_context["plugin_provenance"],
        edit_name=policy_context["edit_name"],
        artifacts_payload=assembly_context["artifacts_payload"],
        validation_filtered=assembly_context["validation_filtered"],
        guard_overhead_section=assembly_context["guard_overhead_section"],
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
