from __future__ import annotations

import copy
from collections.abc import Mapping
from datetime import datetime
from typing import Any


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
        decision = guard_result.get("decision")
        if not isinstance(decision, str) or not decision:
            decision = "allow" if bool(guard_result.get("passed", False)) else "block"
        guard_entry = {
            "name": guard_name,
            "passed": guard_result.get("passed"),
            "decision": decision,
            "policy": guard_result.get("policy", {}),
            "metrics": guard_result.get("metrics", {}),
            "diagnostics": guard_result.get("diagnostics", []),
            "violations": guard_result.get("violations", []),
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
