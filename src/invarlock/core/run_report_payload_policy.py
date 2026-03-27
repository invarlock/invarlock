from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RunReportPolicyViolation:
    code: str
    message: str
    details: dict[str, Any]


def _coerce_report_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def build_run_report_context(
    *,
    profile_normalized: str,
    auto_config: Mapping[str, Any],
    run_context: Mapping[str, Any],
) -> dict[str, Any]:
    run_policy_context = (
        dict(run_context.get("run")) if isinstance(run_context.get("run"), dict) else {}
    )
    eval_policy_context = (
        dict(run_context.get("eval"))
        if isinstance(run_context.get("eval"), dict)
        else {}
    )
    return {
        "profile": profile_normalized,
        "auto": dict(auto_config),
        "assurance": dict(run_context.get("assurance") or {}),
        "run": run_policy_context,
        "eval": eval_policy_context,
    }


def build_run_report_meta(
    *,
    model_id: str,
    adapter: str,
    resolved_device: Any,
    commit_value: str,
    seed_bundle: Mapping[str, Any],
    auto_config: Mapping[str, Any],
    guard_overhead_threshold: float,
    model_profile: Any,
    timestamp: str | None = None,
    invarlock_version: str | None = None,
    env_flags: Mapping[str, object] | None = None,
    determinism_meta: Mapping[str, Any] | None = None,
    pm_acceptance_range: tuple[float, float] | None = None,
    pm_drift_band: tuple[float, float] | None = None,
) -> dict[str, Any]:
    meta_payload: dict[str, Any] = {
        "model_id": model_id,
        "adapter": adapter,
        "device": str(resolved_device),
        "commit": commit_value,
        "seed": seed_bundle["python"],
        "seeds": dict(seed_bundle),
        "ts": timestamp or datetime.now().isoformat(),
        "auto": dict(auto_config),
        "guard_overhead_threshold": guard_overhead_threshold,
        "model_profile": {
            "family": getattr(model_profile, "family", ""),
            "default_loss": getattr(model_profile, "default_loss", ""),
            "module_selectors": getattr(model_profile, "module_selectors", {}),
            "invariants": list(getattr(model_profile, "invariants", ()) or ()),
            "cert_lints": [
                dict(lint) for lint in (getattr(model_profile, "cert_lints", ()) or ())
            ],
        },
    }
    if invarlock_version:
        meta_payload["invarlock_version"] = invarlock_version
    if env_flags:
        meta_payload["env_flags"] = dict(env_flags)
    if determinism_meta:
        meta_payload["determinism"] = dict(determinism_meta)
    if pm_acceptance_range:
        meta_payload["pm_acceptance_range"] = pm_acceptance_range
    if pm_drift_band:
        meta_payload["pm_drift_band"] = pm_drift_band
    return meta_payload


def build_run_report_data(
    *,
    canonical_dataset_id: str,
    resolved_split: str,
    seq_len: int,
    stride: int,
    preview_count: int,
    final_count: int,
    dataset_meta_context: Mapping[str, Any] | None,
    tokenizer_hash: str | None,
) -> tuple[dict[str, Any], str | None]:
    data_payload: dict[str, Any] = {
        "dataset": canonical_dataset_id,
        "split": resolved_split,
        "seq_len": seq_len,
        "stride": stride,
        "preview_n": preview_count,
        "final_n": final_count,
    }
    resolved_tokenizer_hash = tokenizer_hash
    if isinstance(dataset_meta_context, Mapping):
        data_payload.update(dataset_meta_context)
        dataset_tokenizer_hash = dataset_meta_context.get("tokenizer_hash")
        if (
            not resolved_tokenizer_hash
            and isinstance(dataset_tokenizer_hash, str)
            and dataset_tokenizer_hash
        ):
            resolved_tokenizer_hash = dataset_tokenizer_hash
    return data_payload, resolved_tokenizer_hash


def build_snapshot_provenance(
    snapshot_provenance: Mapping[str, Any] | None,
) -> dict[str, bool]:
    snapshot_provenance = snapshot_provenance or {}
    return {
        "restore_failed": bool(snapshot_provenance.get("restore_failed")),
        "reload_path_used": bool(snapshot_provenance.get("reload_path_used")),
    }


def build_edit_payload(
    *,
    core_edit: Mapping[str, Any] | None,
    edit_name: str,
    edit_label: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    report_edit: dict[str, Any] = {}
    context_edit: dict[str, Any] | None = None

    if isinstance(core_edit, Mapping) and core_edit:
        edit_deltas = core_edit.get("deltas", {})
        if not isinstance(edit_deltas, Mapping):
            edit_deltas = {}
        report_edit.update(
            {
                "name": edit_name,
                "plan_digest": core_edit.get("plan_digest", str(hash(str(core_edit)))),
                "deltas": {
                    "params_changed": edit_deltas.get("params_changed", 0),
                    "sparsity": edit_deltas.get("sparsity"),
                    "bitwidth_map": edit_deltas.get("bitwidth_map"),
                    "layers_modified": edit_deltas.get("layers_modified", 0),
                },
            }
        )
        for key in (
            "algorithm",
            "algorithm_version",
            "implementation",
            "scope",
            "ranking",
            "grouping",
            "budgets",
            "seed",
            "mask_digest",
        ):
            if key in core_edit:
                report_edit[key] = copy.deepcopy(core_edit[key])
        context_edit = {
            "name": edit_name,
            "params_changed": edit_deltas.get("params_changed", 0),
            "layers_modified": edit_deltas.get("layers_modified", 0),
        }

    if edit_label:
        report_edit["name"] = edit_label
        report_edit["algorithm"] = edit_label
        context_edit = dict(context_edit or {})
        context_edit["name"] = edit_label

    return report_edit, context_edit


def validate_pairing_report_metrics(
    metrics_section: Mapping[str, Any] | None,
    *,
    baseline_requested: bool,
    profile: str | None,
    preview_count_report: Any,
    final_count_report: Any,
    expected_preview: Any,
    expected_final: Any,
) -> list[RunReportPolicyViolation]:
    metrics = dict(metrics_section) if isinstance(metrics_section, Mapping) else {}
    violations: list[RunReportPolicyViolation] = []

    match_fraction = metrics.get("window_match_fraction")
    if match_fraction is not None:
        try:
            if float(match_fraction) != 1.0:
                violations.append(
                    RunReportPolicyViolation(
                        code="E001",
                        message=(
                            "PAIRING-SCHEDULE-MISMATCH: "
                            f"window_match_fraction={float(match_fraction):.3f}"
                        ),
                        details={"window_match_fraction": float(match_fraction)},
                    )
                )
        except (TypeError, ValueError, OverflowError):
            pass

    overlap_fraction = metrics.get("window_overlap_fraction")
    if overlap_fraction is not None:
        try:
            if float(overlap_fraction) > 1e-9:
                violations.append(
                    RunReportPolicyViolation(
                        code="E001",
                        message=(
                            "PAIRING-SCHEDULE-MISMATCH: "
                            f"window_overlap_fraction={float(overlap_fraction):.3f}"
                        ),
                        details={"window_overlap_fraction": float(overlap_fraction)},
                    )
                )
        except (TypeError, ValueError, OverflowError):
            pass

    profile_normalized = (profile or "").strip().lower()
    if baseline_requested and profile_normalized in {"ci", "release"}:
        pairing_reason = metrics.get("window_pairing_reason")
        if pairing_reason is not None:
            violations.append(
                RunReportPolicyViolation(
                    code="E001",
                    message=(
                        "PAIRING-SCHEDULE-MISMATCH: baseline pairing requested but "
                        f"run was not paired (window_pairing_reason={pairing_reason})"
                    ),
                    details={"window_pairing_reason": pairing_reason},
                )
            )

        paired_windows_val = metrics.get("paired_windows")
        paired_windows_int = _coerce_report_count(paired_windows_val)
        if paired_windows_int is None or paired_windows_int <= 0:
            violations.append(
                RunReportPolicyViolation(
                    code="E001",
                    message=(
                        "PAIRED-WINDOWS-COLLAPSED: paired_windows<=0 under paired "
                        "baseline. Check device stability, dataset windows, or "
                        "edit scope."
                    ),
                    details={
                        "paired_windows": paired_windows_val,
                        "profile": profile_normalized,
                    },
                )
            )

    preview_used = _coerce_report_count(preview_count_report)
    preview_expected = _coerce_report_count(expected_preview)
    final_used = _coerce_report_count(final_count_report)
    final_expected = _coerce_report_count(expected_final)
    if (
        preview_used is not None
        and preview_expected is not None
        and preview_used != preview_expected
    ) or (
        final_used is not None
        and final_expected is not None
        and final_used != final_expected
    ):
        violations.append(
            RunReportPolicyViolation(
                code="E001",
                message=(
                    "PAIRING-SCHEDULE-MISMATCH: counts do not match configuration "
                    "after stratification"
                ),
                details={
                    "preview_used": preview_used if preview_used is not None else -1,
                    "preview_expected": (
                        preview_expected if preview_expected is not None else -1
                    ),
                    "final_used": final_used if final_used is not None else -1,
                    "final_expected": (
                        final_expected if final_expected is not None else -1
                    ),
                },
            )
        )

    return violations


def build_dataset_window_stats(
    *,
    match_fraction: Any,
    overlap_fraction: Any,
    window_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if match_fraction is not None:
        try:
            stats["window_match_fraction"] = float(match_fraction)
        except (TypeError, ValueError, OverflowError):
            pass
    if overlap_fraction is not None:
        try:
            stats["window_overlap_fraction"] = float(overlap_fraction)
        except (TypeError, ValueError, OverflowError):
            pass

    if isinstance(window_plan, Mapping) and "coverage_ok" in window_plan:
        stats["coverage"] = bool(window_plan.get("coverage_ok"))
        stats["preview_total_tokens"] = window_plan.get("preview_total_tokens")
        stats["final_total_tokens"] = window_plan.get("final_total_tokens")
        stats["min_tokens_target"] = window_plan.get("min_tokens_target")
        stats["tokens_floor_met"] = window_plan.get("tokens_floor_met")

    return stats


def merge_core_timing_metrics(
    timings: Mapping[str, Any], core_metrics: Mapping[str, Any] | None
) -> dict[str, Any]:
    merged = dict(timings)
    core_timings = (
        core_metrics.get("timings") if isinstance(core_metrics, Mapping) else None
    )
    if not isinstance(core_timings, Mapping):
        return merged
    for key in ("prepare", "prepare_guards", "edit", "guards", "eval", "finalize"):
        if key not in core_timings:
            continue
        try:
            merged[key] = float(core_timings[key])
        except (AttributeError, TypeError, ValueError, OverflowError):
            merged[key] = core_timings[key]
    return merged


def build_metrics_payload(
    *,
    core_metrics: Mapping[str, Any] | None,
    window_plan_context: Mapping[str, Any] | None,
    dataset_meta_context: Mapping[str, Any] | None,
    resolved_loss_type: str | None,
    latency_default: float = 0.0,
    memory_default: float = 0.0,
) -> dict[str, Any]:
    metrics = core_metrics if isinstance(core_metrics, Mapping) else {}
    metrics_payload: dict[str, Any] = {
        "latency_ms_per_tok": metrics.get("latency_ms_per_tok", latency_default),
        "memory_mb_peak": metrics.get("memory_mb_peak", memory_default),
        "spectral": {},
        "rmt": {},
        "invariants": {},
    }
    window_plan_ctx = window_plan_context
    if isinstance(window_plan_ctx, Mapping):
        metrics_payload["window_plan"] = dict(window_plan_ctx)
        capacity_meta = window_plan_ctx.get("capacity")
        if isinstance(capacity_meta, Mapping):
            metrics_payload["window_capacity"] = dict(capacity_meta)
        stats_section = metrics_payload.setdefault("stats", {})
        if isinstance(stats_section, dict):
            stats_section.update(
                {
                    "requested_preview": window_plan_ctx.get("requested_preview"),
                    "requested_final": window_plan_ctx.get("requested_final"),
                    "actual_preview": window_plan_ctx.get("actual_preview"),
                    "actual_final": window_plan_ctx.get("actual_final"),
                    "coverage_ok": window_plan_ctx.get("coverage_ok"),
                    "preview_total_tokens": window_plan_ctx.get("preview_total_tokens"),
                    "final_total_tokens": window_plan_ctx.get("final_total_tokens"),
                    "min_tokens_target": window_plan_ctx.get("min_tokens_target"),
                    "tokens_floor_met": window_plan_ctx.get("tokens_floor_met"),
                    "dedupe_adjustments": window_plan_ctx.get("dedupe_adjustments"),
                }
            )
    optional_keys = [
        "logloss_preview",
        "logloss_final",
        "logloss_delta",
        "logloss_preview_ci",
        "logloss_final_ci",
        "logloss_delta_ci",
        "bootstrap",
        "window_overlap_fraction",
        "window_match_fraction",
        "window_pairing_reason",
        "window_pairing_preview",
        "window_pairing_final",
        "paired_windows",
        "paired_delta_summary",
        "primary_metric_tail",
        "preview_total_tokens",
        "final_total_tokens",
        "masked_tokens_total",
        "masked_tokens_preview",
        "masked_tokens_final",
        "timings",
        "guard_timings",
        "memory_snapshots",
        "gpu_memory_mb_peak",
        "gpu_memory_reserved_mb_peak",
        "reduction",
    ]
    for key in optional_keys:
        if key in metrics:
            metrics_payload[key] = metrics[key]
    metrics_payload["loss_type"] = resolved_loss_type
    if metrics_payload.get("loss_type") is None and isinstance(
        dataset_meta_context, Mapping
    ):
        metrics_payload["loss_type"] = dataset_meta_context.get(
            "loss_type", resolved_loss_type
        )
    if isinstance(dataset_meta_context, Mapping):
        for meta_key in (
            "masked_tokens_total",
            "masked_tokens_preview",
            "masked_tokens_final",
        ):
            if (
                meta_key not in metrics_payload
                and dataset_meta_context.get(meta_key) is not None
            ):
                metrics_payload[meta_key] = dataset_meta_context[meta_key]
    return metrics_payload


def build_guard_entries(core_guards: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(core_guards, Mapping):
        return []
    entries: list[dict[str, Any]] = []
    for guard_name, guard_result in core_guards.items():
        if not isinstance(guard_result, Mapping):
            continue
        guard_entry = {
            "name": guard_name,
            "passed": guard_result.get("passed"),
            "action": guard_result.get("action"),
            "policy": guard_result.get("policy", {}),
            "metrics": guard_result.get("metrics", {}),
            "actions": guard_result.get("actions", []),
            "violations": guard_result.get("violations", []),
            "warnings": guard_result.get("warnings", []),
            "errors": guard_result.get("errors", []),
            "details": guard_result.get("details", {}),
        }
        for extra_key in ("final_z_scores", "module_family_map"):
            if extra_key in guard_result:
                guard_entry[extra_key] = guard_result[extra_key]
        entries.append(guard_entry)
    return entries


def build_flags_payload(core_guards: Mapping[str, Any] | None) -> dict[str, Any]:
    guard_values = core_guards.values() if isinstance(core_guards, Mapping) else ()
    return {
        "guard_recovered": any(
            not guard.get("passed", True)
            for guard in guard_values
            if isinstance(guard, Mapping)
        ),
        "rollback_reason": None,
    }


def build_artifacts_payload(
    *,
    event_path: Any,
    mask_artifact_path: Any | None = None,
) -> dict[str, Any]:
    payload = {
        "events_path": str(event_path) if event_path else "",
        "logs_path": "",
        "checkpoint_path": None,
    }
    if mask_artifact_path:
        payload["masks_path"] = str(mask_artifact_path)
    return payload


__all__ = [
    "build_artifacts_payload",
    "build_edit_payload",
    "build_flags_payload",
    "build_guard_entries",
    "build_metrics_payload",
    "build_run_report_context",
    "build_run_report_data",
    "build_run_report_meta",
    "build_snapshot_provenance",
    "merge_core_timing_metrics",
]
