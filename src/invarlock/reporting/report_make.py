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
import hashlib
import math
from typing import Any, cast

from invarlock.core.auto_tuning import get_tier_policies
from invarlock.core.exceptions import MetricsError, ValidationError
from invarlock.eval import tail_stats as tail_stats_mod
from invarlock.eval.primary_metric import compute_primary_metric_from_report

from . import guards_invariants as guards_invariants_mod
from . import guards_rmt as guards_rmt_mod
from . import guards_spectral as guards_spectral_mod
from . import guards_variance as guards_variance_mod
from . import policy_utils as report_policy_utils_mod
from . import report_build_context as report_build_context_mod
from . import report_edit_summary as report_edit_summary_mod
from . import report_enrichment as report_enrichment_mod
from . import report_normalization as report_normalization_mod
from . import report_overhead as report_overhead_mod
from . import report_policy as report_policy_mod
from . import report_primary_metric_analysis as report_primary_metric_analysis_mod
from . import report_provenance as report_provenance_mod
from . import report_schema as report_schema_mod
from . import report_validation as report_validation_mod
from .dataset_hashing import _extract_dataset_info
from .report_confidence import compute_confidence_label as _compute_confidence_label
from .report_primary_metric_policy import (
    enforce_display_ci_alignment as _enforce_display_ci_alignment,
)
from .report_primary_metric_policy import (
    propagate_pairing_stats as _propagate_pairing_stats,
)
from .report_types import RunReport
from .report_validation_allowlist import (
    apply_validation_allowlist_schema as _apply_validation_allowlist_schema,
)
from .report_validation_allowlist import (
    load_validation_allowlist as _load_validation_allowlist,
)
from .report_validation_allowlist import (
    load_validation_allowlist_with_source as _load_validation_allowlist_with_source,
)
from .utils import _coerce_int, _sanitize_seed_bundle

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


## Helpers are imported from invarlock.reporting.utils
_collect_backend_versions = report_provenance_mod.collect_backend_versions
_compute_edit_digest = report_provenance_mod.compute_edit_digest
_compute_report_digest = report_provenance_mod.compute_report_digest


## Pairing helper available from invarlock.reporting.utils


def _compute_thresholds_payload(
    tier: str, resolved_policy: dict[str, Any]
) -> dict[str, Any]:
    from .policy_utils import _compute_thresholds_payload as _impl

    return _impl(tier, resolved_policy)


def _compute_thresholds_hash(payload: dict[str, Any]) -> str:
    from .policy_utils import _compute_thresholds_hash as _impl

    return _impl(payload)


# Tighten JSON Schema: populate validation.properties from allow-list and
# disallow unknown validation keys at schema level.
_VALIDATION_ALLOWLIST_KEYS, _VALIDATION_ALLOWLIST_SOURCE = (
    _load_validation_allowlist_with_source()
)
_apply_validation_allowlist_schema(REPORT_JSON_SCHEMA, _VALIDATION_ALLOWLIST_KEYS)


## Note: helpers like _get_section/_get_mapping/_iter_guard_entries,
## and policy helpers are provided by invarlock.reporting.utils and policy_utils.
## Import those directly in callers/tests instead of through this module.


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _extract_report_meta(
    report: RunReport, diagnostics: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Extract the evaluation report metadata block with a full seed bundle."""

    def _note(code: str, message: str) -> None:
        if diagnostics is not None:
            _append_build_diagnostic(diagnostics, code=code, message=message)

    raw_meta_section = report.get("meta")
    meta_section: dict[str, Any] = (
        dict(raw_meta_section) if isinstance(raw_meta_section, dict) else {}
    )
    seed_value = _coerce_int(meta_section.get("seed"))
    seeds_bundle = _sanitize_seed_bundle(meta_section.get("seeds"), seed_value)
    primary_seed = (
        seeds_bundle.get("python") if isinstance(seeds_bundle, dict) else None
    )
    if primary_seed is None:
        primary_seed = 0
    model_id = _optional_text(meta_section.get("model_id"))
    if model_id is None:
        _note(
            "meta.model_id_unavailable",
            "Run metadata is missing a usable model_id; evaluation report metadata leaves it null.",
        )
    adapter = _optional_text(meta_section.get("adapter"))
    if adapter is None:
        _note(
            "meta.adapter_unavailable",
            "Run metadata is missing a usable adapter; evaluation report metadata leaves it null.",
        )
    device = _optional_text(meta_section.get("device"))
    if device is None:
        _note(
            "meta.device_unavailable",
            "Run metadata is missing a usable device; evaluation report metadata leaves it null.",
        )
    return {
        "model_id": model_id,
        "adapter": adapter,
        "device": device,
        "ts": meta_section.get("ts"),
        "commit": meta_section.get("commit"),
        "seed": primary_seed,
        "seeds": seeds_bundle,
    }


# Console validation helpers live in `report_console`; Markdown rendering lives in
# `render`. This module owns report-construction helpers only.
## Private helper functions


## Dataset hashing helpers live in invarlock.reporting.dataset_hashing


def _generate_run_id(report: RunReport) -> str:
    """Generate a unique run ID from report metadata."""
    raw_meta = (
        report.get("meta") if isinstance(report, dict) else getattr(report, "meta", {})
    )
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    if isinstance(meta, dict):
        existing = meta.get("run_id")
        if isinstance(existing, str) and existing:
            return existing
        timestamp = meta.get("ts", meta.get("start_time", ""))
        timestamp_str = str(timestamp) if timestamp is not None else ""
        model_id = _optional_text(meta.get("model_id")) or ""
        commit = str(meta.get("commit", meta.get("commit_sha", "")) or "")[:16]
        base_str = f"{timestamp_str}{model_id}{commit}"
    return hashlib.sha256(base_str.encode()).hexdigest()[:16]


def _append_build_diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    severity: str = "warning",
) -> None:
    entry: dict[str, Any] = {
        "code": code,
        "message": message,
        "severity": severity,
    }
    if details:
        entry["details"] = details
    diagnostics.append(entry)


## NOTE: _compute_report_hash moved to invarlock.reporting.render and is re-exported below.


## Note: compute_window_hashes is available under invarlock.reporting.dataset_hashing.


def make_report(
    report: RunReport,
    baseline: RunReport | dict[str, Any],
) -> dict[str, Any]:
    """Generate an evaluation report from a RunReport and baseline comparison."""
    NON_FATAL_EXCEPTIONS = (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    )
    evaluate_metric_tail = tail_stats_mod.evaluate_metric_tail
    build_diagnostics: list[dict[str, Any]] = []
    provenance_blocking_issue = False

    def _record_blocking_diagnostic(code: str, message: str) -> None:
        nonlocal provenance_blocking_issue
        provenance_blocking_issue = True
        _append_build_diagnostic(
            build_diagnostics,
            code=code,
            message=message,
            severity="error",
        )

    report = report_normalization_mod.normalize_and_validate_run_report(report)
    report_map = cast(dict[str, Any], report)

    # Normalize baseline input
    baseline_raw = baseline
    baseline_raw_map = cast(dict[str, Any], baseline_raw)
    baseline_normalized = report_normalization_mod.normalize_baseline(baseline_raw)
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
    except NON_FATAL_EXCEPTIONS as exc:
        raise ValidationError(
            code="E232",
            message=(
                "Baseline report normalization failed; evaluation report assembly "
                "requires a valid baseline report."
            ),
            details={"error": str(exc)},
        ) from exc

    # Extract core metadata with full seed bundle
    meta = _extract_report_meta(report, build_diagnostics)

    # Propagate environment flags captured in the RunReport (e.g., deterministic algos,
    # TF32 controls, MPS/CUDA availability). This is useful for auditability and
    # reproducibility of evaluation runs.
    try:
        env_flags = (
            report.get("meta", {}).get("env_flags")
            if isinstance(report.get("meta"), dict)
            else None
        )
        if isinstance(env_flags, dict) and env_flags:
            meta["env_flags"] = env_flags
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        _record_blocking_diagnostic(
            code="meta.env_flags_unavailable",
            message="Environment flag provenance could not be copied into the evaluation report.",
        )

    # Determinism preset (CI/Release provenance) when present.
    try:
        det = (
            report.get("meta", {}).get("determinism")
            if isinstance(report.get("meta"), dict)
            else None
        )
        if isinstance(det, dict) and det:
            meta["determinism"] = det
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        _record_blocking_diagnostic(
            code="meta.determinism_unavailable",
            message="Determinism provenance could not be copied into the evaluation report.",
        )

    # Execution profile provenance when available via run context.
    try:
        ctx = report.get("context") if isinstance(report, dict) else None
        ctx_profile = (
            str(ctx.get("profile") or "").strip().lower()
            if isinstance(ctx, dict)
            else ""
        )
        if ctx_profile:
            meta["profile"] = ctx_profile
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        _record_blocking_diagnostic(
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

    # Extract auto-tuning configuration
    auto_config = report["meta"].get("auto")
    if auto_config:
        auto: dict[str, Any] = {
            "tier": auto_config.get("tier", "balanced"),
            "probes_used": auto_config.get("probes", auto_config.get("probes_used", 0)),
            "target_pm_ratio": auto_config.get("target_pm_ratio"),
        }
    else:
        auto = {"tier": "none", "probes_used": 0, "target_pm_ratio": None}

    # Extract dataset configuration and compute hashes
    dataset_info = _extract_dataset_info(report_map)
    try:
        if isinstance(dataset_info, dict):
            windows = dataset_info.get("windows")
            if isinstance(windows, dict):
                windows.setdefault("stats", {})
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        _append_build_diagnostic(
            build_diagnostics,
            code="dataset.windows_stats_unavailable",
            message="Dataset window statistics could not be initialized in the evaluation report.",
        )

    # Baseline reference (PM-only). Derive a primary_metric snapshot from baseline windows.
    # Prefer explicit baseline primary_metric when provided; otherwise compute from windows
    def _direct_baseline_metric(payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None

        def _is_finite_number(value: Any) -> bool:
            return isinstance(value, (int, float)) and math.isfinite(float(value))

        def _coerce_finite_float(value: Any) -> float | None:
            if not _is_finite_number(value):
                return None
            return float(value)

        report_kind = (
            report.get("metrics", {}).get("primary_metric", {}).get("kind")
            if isinstance(report.get("metrics"), dict)
            else None
        )
        default_kind = str(report_kind or "ppl_causal")

        direct_pm = payload.get("primary_metric")
        if isinstance(direct_pm, dict) and _is_finite_number(direct_pm.get("final")):
            return copy.deepcopy(direct_pm)
        if _is_finite_number(payload.get("ppl_final")):
            preview_value_raw = payload.get("ppl_preview")
            if not _is_finite_number(preview_value_raw):
                preview_value_raw = payload.get("ppl_final")
            preview_value = _coerce_finite_float(preview_value_raw)
            final_value = _coerce_finite_float(payload.get("ppl_final"))
            if preview_value is None or final_value is None:
                return None
            return {
                "kind": default_kind,
                "preview": preview_value,
                "final": final_value,
            }

        metrics_block = payload.get("metrics")
        if not isinstance(metrics_block, dict):
            return None
        block_pm = metrics_block.get("primary_metric")
        if isinstance(block_pm, dict) and _is_finite_number(block_pm.get("final")):
            return copy.deepcopy(block_pm)
        ppl_final = metrics_block.get("ppl_final")
        if _is_finite_number(ppl_final):
            ppl_preview_raw = metrics_block.get("ppl_preview", ppl_final)
            if not _is_finite_number(ppl_preview_raw):
                ppl_preview_raw = ppl_final
            ppl_preview = _coerce_finite_float(ppl_preview_raw)
            ppl_final_value = _coerce_finite_float(ppl_final)
            if ppl_preview is None or ppl_final_value is None:
                return None
            return {
                "kind": default_kind,
                "preview": ppl_preview,
                "final": ppl_final_value,
            }
        return None

    baseline_pm = None
    try:
        bm = (
            baseline_raw.get("metrics", {}).get("primary_metric")
            if isinstance(baseline_raw.get("metrics"), dict)
            else None
        )
        if (
            isinstance(bm, dict)
            and bm
            and "final" in bm
            and bm.get("final") is not None
        ):
            baseline_pm = bm
    except NON_FATAL_EXCEPTIONS as exc:
        raise MetricsError(
            code="E233",
            message=(
                "Evaluation report assembly requires a concrete baseline metric; "
                "baseline primary metric lookup failed."
            ),
            details={"error": str(exc)},
        ) from exc
    if not isinstance(baseline_pm, dict) or not baseline_pm:
        baseline_pm = _direct_baseline_metric(baseline_raw)
    if not isinstance(baseline_pm, dict) or not baseline_pm:
        baseline_pm = _direct_baseline_metric(baseline_normalized)
    if not isinstance(baseline_pm, dict) or not baseline_pm:
        try:
            baseline_metric_source: dict[str, Any] = baseline_normalized
            if (
                isinstance(baseline_raw, dict)
                and isinstance(baseline_raw_map.get("evaluation_windows"), dict)
                and _direct_baseline_metric(baseline_normalized) is None
            ):
                baseline_metric_source = baseline_raw_map
            baseline_pm = compute_primary_metric_from_report(baseline_metric_source)
        except NON_FATAL_EXCEPTIONS as exc:
            raise MetricsError(
                code="E234",
                message=(
                    "Evaluation report assembly requires a concrete baseline metric; "
                    "baseline primary metric could not be derived."
                ),
                details={"error": str(exc)},
            ) from exc
    baseline_final = baseline_pm.get("final") if isinstance(baseline_pm, dict) else None
    if not isinstance(baseline_final, (int, float)) or not math.isfinite(
        float(baseline_final)
    ):
        raise MetricsError(
            code="E235",
            message=(
                "Evaluation report assembly requires a concrete finite `final` value "
                "for the baseline primary metric."
            ),
            details={"baseline_primary_metric": baseline_pm},
        )
    baseline_ref: dict[str, Any] = {
        "run_id": _optional_text(baseline_normalized.get("run_id")),
        "model_id": _optional_text(baseline_normalized.get("model_id")),
        "primary_metric": {
            "kind": baseline_pm.get("kind", "ppl_causal"),
            "final": float(baseline_final),
        },
    }
    baseline_metrics = (
        baseline_raw.get("metrics")
        if isinstance(baseline_raw, dict)
        and isinstance(baseline_raw.get("metrics"), dict)
        else None
    )
    if isinstance(baseline_metrics, dict):
        classification_metrics = baseline_metrics.get("classification")
        if isinstance(classification_metrics, dict) and classification_metrics:
            baseline_ref["metrics"] = {
                "classification": copy.deepcopy(classification_metrics)
            }
    # Propagate baseline tokenizer hash for verify-time linting when available
    baseline_tok_hash = baseline_normalized.get("tokenizer_hash")
    if isinstance(baseline_tok_hash, str) and baseline_tok_hash:
        baseline_ref["tokenizer_hash"] = baseline_tok_hash

    ppl_analysis, window_plan_profile = (
        report_primary_metric_analysis_mod.build_primary_metric_analysis(
            report_map,
            baseline_normalized,
            baseline_ref,
            dataset_info,
        )
    )
    ppl_metrics = report_map.get("metrics", {})

    # Extract invariant status
    invariants = guards_invariants_mod._extract_invariants(
        report,
        baseline=baseline_report,
    )

    # Extract spectral analysis
    spectral = guards_spectral_mod._extract_spectral_analysis(
        report,
        baseline_normalized,
    )

    # Extract RMT analysis
    rmt = guards_rmt_mod._extract_rmt_analysis(report, baseline_normalized)

    # Extract variance guard info
    variance = guards_variance_mod._extract_variance_analysis(report)

    # Extract structural deltas
    structure = report_edit_summary_mod.extract_structural_deltas(report)
    compression_diag = structure.get("compression_diagnostics", {})
    structure["compression_diagnostics"] = compression_diag

    # Extract effective policies used
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

    # Resolve tier/profile policy (canonical) and merge observed guard policies.
    profile = None
    explicit_overrides: dict[str, Any] | None = None
    try:
        ctx = report.get("context") if isinstance(report, dict) else None
        if isinstance(ctx, dict) and ctx.get("profile"):
            profile = str(ctx.get("profile"))
    except NON_FATAL_EXCEPTIONS:
        profile = None
        provenance_blocking_issue = True
        _append_build_diagnostic(
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
    except NON_FATAL_EXCEPTIONS:
        profile = None
        provenance_blocking_issue = True
        _append_build_diagnostic(
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
    except NON_FATAL_EXCEPTIONS:
        explicit_overrides = None
        provenance_blocking_issue = True
        _append_build_diagnostic(
            build_diagnostics,
            code="policy.explicit_overrides_unavailable",
            message="Explicit guard overrides could not be extracted from the run configuration.",
            severity="error",
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
        "validation_allowlist_source": _VALIDATION_ALLOWLIST_SOURCE,
    }
    if profile in {"ci", "release"} and _VALIDATION_ALLOWLIST_SOURCE != "contracts":
        _record_blocking_diagnostic(
            code="policy.validation_allowlist_source_invalid",
            message=(
                "CI/Release evaluation reports must resolve validation allowlists "
                "from contracts-only sources."
            ),
        )
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
        _optional_text(edit_section.get("name"))
        if isinstance(edit_section, dict)
        else None
    )
    if isinstance(edit_metadata, dict) and edit_name is None:
        edit_metadata["name"] = None

    telemetry = report_build_context_mod.extract_telemetry(report, meta.get("device"))

    # Build the evaluation report
    window_capacity_ctx = (
        report.get("metrics", {}).get("window_capacity")
        if isinstance(report.get("metrics"), dict)
        else None
    )

    artifacts_payload = report_build_context_mod.build_artifacts_payload(report)

    raw_guard_ctx = report.get("guard_overhead")
    guard_overhead_section, _ = report_overhead_mod.prepare_guard_overhead_section(
        raw_guard_ctx
    )

    schedule_digest = report_build_context_mod.attach_schedule_digest(
        report, guard_overhead_section
    )

    policy_provenance["resolved_at"] = artifacts_payload["generated_at"]

    current_run_id = _generate_run_id(report)
    provenance = report_provenance_mod.build_provenance_block(
        report_map,
        baseline_raw_map,
        baseline_ref,
        artifacts_payload,
        policy_provenance,
        schedule_digest,
        ppl_analysis,
        current_run_id,
        compute_report_digest_fn=_compute_report_digest,
        collect_backend_versions_fn=_collect_backend_versions,
        compute_edit_digest_fn=_compute_edit_digest,
    )

    moe_section = report_build_context_mod.build_moe_section(
        report, baseline_raw, baseline_normalized
    )

    capacity_tokens, capacity_examples = (
        report_build_context_mod.resolve_capacity_context(
            window_capacity_ctx, dataset_info
        )
    )

    pm_acceptance_range = report_policy_mod.resolve_pm_acceptance_range_from_report(
        report_map,
    )
    pm_drift_band = report_policy_mod.resolve_pm_drift_band_from_report(
        report_map, drift_band_default=PM_DRIFT_BAND_DEFAULT
    )
    tiny_relax = report_policy_mod.resolve_tiny_relax_from_report(report_map)

    pm_tail_result = report_build_context_mod.evaluate_primary_metric_tail(
        report,
        baseline_normalized,
        resolved_policy,
        evaluate_metric_tail,
    )

    target_ratio_raw = auto.get("target_pm_ratio")
    target_ratio = (
        float(target_ratio_raw)
        if isinstance(target_ratio_raw, int | float)
        else None
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
        pm_drift_band_default=PM_DRIFT_BAND_DEFAULT,
        get_tier_policies_fn=get_tier_policies,
    )

    # Enforce validation key allow-list to prevent surface drift
    _allowed_validation = _load_validation_allowlist()
    validation_filtered = {
        k: bool(v) for k, v in validation_flags.items() if k in _allowed_validation
    }
    if provenance_blocking_issue:
        validation_filtered["primary_metric_acceptable"] = False

    evaluation_report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
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
        "edit_name": edit_name,
        "artifacts": artifacts_payload,
        "validation": validation_filtered,
        "guard_overhead": guard_overhead_section,
        "primary_metric_tail": pm_tail_result,
    }

    # Record tiny-relax provenance explicitly when active.
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
        except NON_FATAL_EXCEPTIONS:  # pragma: no cover
            _record_blocking_diagnostic(
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
        _propagate_pairing_stats(evaluation_report, ppl_analysis)
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        _record_blocking_diagnostic(
            code="pairing.stats_unavailable",
            message="Pairing statistics could not be propagated into the evaluation report.",
        )

    report_enrichment_mod.attach_policy_digest(
        evaluation_report,
        auto,
        resolved_policy,
        baseline_raw_map,
        baseline_normalized,
        _compute_thresholds_payload,
        _compute_thresholds_hash,
        POLICY_VERSION,
    )
    report_enrichment_mod.attach_secondary_metrics(evaluation_report, report_map)
    report_enrichment_mod.attach_classification(evaluation_report, report_map)
    report_enrichment_mod.attach_system_overhead(
        evaluation_report,
        report_map,
        baseline_raw_map,
        telemetry,
    )

    # Attach/normalize primary metric block (moved to helper)
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
    except NON_FATAL_EXCEPTIONS:  # pragma: no cover
        _record_blocking_diagnostic(
            code="primary_metric.drift_band_unavailable",
            message="Primary-metric drift-band metadata could not be attached to the evaluation report.",
        )
    _enforce_display_ci_alignment(
        ppl_analysis.get("stats", {}).get("pairing", "run_metrics"),
        evaluation_report.get("primary_metric"),
        ppl_analysis.get("logloss_delta_ci"),
        window_plan_profile,
    )

    report_enrichment_mod.ensure_primary_metric_display_ci(evaluation_report)
    report_enrichment_mod.attach_telemetry_summary_line(
        evaluation_report, report_map, current_run_id
    )
    report_enrichment_mod.attach_confidence_label(
        evaluation_report, _compute_confidence_label
    )
    if build_diagnostics:
        meta_section = evaluation_report.setdefault("meta", {})
        if isinstance(meta_section, dict):
            meta_section["build_diagnostics"] = build_diagnostics

    return evaluation_report
