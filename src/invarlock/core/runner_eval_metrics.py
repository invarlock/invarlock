from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .bootstrap import compute_paired_delta_log_ci, logspace_to_ratio_ci
from .exceptions import InvarlockError
from .runner_eval_latency import (
    _raise_latency_error,
    measure_latency,
    samples_to_dataloader,
)
from .runner_eval_metrics_multimodal import (
    _build_multimodal_eval_result,
    _evaluate_vision_text_arm,
    _is_multimodal_batch,
    _model_kwargs,
    _normalize_answer_text,
    _normalize_reference_answers,
    _prediction_answer_text,
    _resolve_adapter_hook,
    _resolve_metric_kind,
)
from .runner_eval_metrics_stats import (
    _BootstrapDeltaResult,
    _compute_bootstrap_delta_stats,
    _evaluate_pairing_and_coverage,
    _pairing_error_result,
    _PairingCoverageResult,
)
from .runner_eval_windows import compute_slice_summary, resolve_limit, slice_calibration
from .runner_pairing import (
    BOOTSTRAP_COVERAGE_REQUIREMENTS,
    assess_bootstrap_coverage,
    compute_window_pairing_metrics,
)
from .types import LogLevel


@dataclass(frozen=True)
class _EvalWindowSelection:
    preview_data: Any
    final_data: Any
    preview_n: int
    final_n: int
    requested_preview: int
    requested_final: int
    total_available: int


@dataclass(frozen=True)
class _EvalRuntimeContext:
    device: Any
    process: Any
    initial_memory: float
    preview_data: Any
    final_data: Any
    preview_n: int
    final_n: int
    resolved_loss_mode: str
    bootstrap_enabled: bool
    bootstrap_method: str
    bootstrap_replicates: int
    bootstrap_alpha: float
    bootstrap_seed: int
    ci_band: float
    single_method: str
    delta_method: str
    profile_label: str
    pairing_context: dict[str, Any]


@dataclass(frozen=True)
class _SliceMetricsResult:
    eval_error: dict[str, Any] | None
    preview_log_losses: list[float]
    final_log_losses: list[float]
    preview_tokens_ct: int
    final_tokens_ct: int
    preview_batches_ct: int
    final_batches_ct: int
    preview_window_ids: list[int]
    final_window_ids: list[int]
    preview_tokens: list[list[int]]
    final_tokens: list[list[int]]
    preview_token_counts: list[int]
    final_token_counts: list[int]
    preview_attention_masks: list[list[int]]
    final_attention_masks: list[list[int]]
    preview_mask_counts: list[int]
    final_mask_counts: list[int]
    preview_labels: list[list[int]]
    final_labels: list[list[int]]
    preview_actual_token_counts: list[int]
    final_actual_token_counts: list[int]
    preview_actual_tokens_ct: int
    final_actual_tokens_ct: int
    preview_masked_total: int
    final_masked_total: int
    preview_mean_log: float
    final_mean_log: float
    delta_mean_log: float
    pm_preview: float
    pm_final: float
    ppl_ratio: float
    paired_windows_attempted: int
    pm_invalid: bool


def _resolve_eval_device(model: Any, config: Any, torch_mod: Any) -> Any:
    device = next(model.parameters()).device
    eval_device_override = None
    if config and isinstance(getattr(config, "context", None), dict):
        eval_section = config.context.get("eval")
        if isinstance(eval_section, dict):
            override = eval_section.get("device_override")
            if isinstance(override, str) and override.strip():
                eval_device_override = override.strip()
    if eval_device_override:
        override_device = torch_mod.device(eval_device_override)
        if override_device != device:
            model.to(override_device)
            return override_device
    return device


def _select_eval_windows(
    runner: Any,
    calibration_data: Any,
    *,
    preview_n: int | None,
    final_n: int | None,
    allow_materialize: bool,
) -> _EvalWindowSelection:
    if not hasattr(calibration_data, "__len__"):
        if allow_materialize and hasattr(calibration_data, "__iter__"):
            calibration_data = list(calibration_data)
        else:
            raise ValueError(
                "Calibration data must define __len__ (or enable materialization "
                "via INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE or context.run.allow_calibration_materialize)."
            )

    total_available = (
        len(calibration_data) if hasattr(calibration_data, "__len__") else 0
    )
    if total_available == 0:
        raise ValueError("Calibration data is empty; cannot compute metrics.")

    resolved_preview_n = max(
        int(preview_n if preview_n is not None else max(total_available // 2, 1)),
        0,
    )
    resolved_final_n = max(
        int(final_n if final_n is not None else resolved_preview_n),
        0,
    )
    max_single_arm = max(resolved_preview_n, resolved_final_n)
    if max_single_arm <= 0:
        raise ValueError("preview_n and final_n cannot both be zero.")

    if max_single_arm > total_available:
        runner._log_event(
            "eval",
            "data_scaled",
            LogLevel.WARNING,
            {
                "requested_preview": resolved_preview_n,
                "requested_final": resolved_final_n,
                "available": total_available,
            },
        )
        resolved_preview_n = min(resolved_preview_n, total_available)
        resolved_final_n = min(resolved_final_n, total_available)

    requested_preview = resolved_preview_n
    requested_final = resolved_final_n
    if resolved_preview_n + resolved_final_n > total_available:
        runner._log_event(
            "eval",
            "window_shortage",
            LogLevel.WARNING,
            {
                "requested_preview": resolved_preview_n,
                "requested_final": resolved_final_n,
                "available": total_available,
            },
        )

    resolved_preview_n = min(resolved_preview_n, total_available)
    final_start = resolved_preview_n
    remaining = max(total_available - resolved_preview_n, 0)
    if resolved_final_n > remaining:
        runner._log_event(
            "eval",
            "final_window_shortage",
            LogLevel.WARNING,
            {
                "requested_final": resolved_final_n,
                "available_after_preview": remaining,
                "requested_preview": requested_preview,
                "requested_final_original": requested_final,
            },
        )
        resolved_final_n = remaining

    calibration_source = calibration_data
    preview_data, calibration_data = slice_calibration(
        calibration_source,
        start=0,
        count=resolved_preview_n,
        allow_materialize=allow_materialize,
    )
    final_data, calibration_data = slice_calibration(
        calibration_data,
        start=final_start,
        count=resolved_final_n,
        allow_materialize=allow_materialize,
    )
    if not final_data and requested_final > 0 and total_available > 0:
        fallback_final_n = min(requested_final, total_available)
        final_data, calibration_data = slice_calibration(
            calibration_source,
            start=0,
            count=fallback_final_n,
            allow_materialize=allow_materialize,
        )
        resolved_final_n = fallback_final_n

    return _EvalWindowSelection(
        preview_data=preview_data,
        final_data=final_data,
        preview_n=resolved_preview_n,
        final_n=resolved_final_n,
        requested_preview=requested_preview,
        requested_final=requested_final,
        total_available=total_available,
    )


def _has_multimodal_batches(*batch_groups: Any) -> bool:
    for batch_group in batch_groups:
        for batch in list(batch_group or []):
            if _is_multimodal_batch(batch):
                return True
    return False


def _resolve_eval_runtime_context(
    runner: Any,
    model: Any,
    calibration_data: Any,
    adapter: Any,
    *,
    preview_n: int | None,
    final_n: int | None,
    config: Any | None,
    torch_mod: Any,
) -> _EvalRuntimeContext | tuple[dict[str, Any], dict[str, Any]]:
    psutil_module: Any = importlib.import_module("psutil")
    model.eval()

    debug_trace_enabled = bool(os.environ.get("INVARLOCK_DEBUG_TRACE"))
    if (
        not debug_trace_enabled
        and config
        and isinstance(getattr(config, "context", None), dict)
    ):
        debug_trace_enabled = bool(config.context.get("debug_trace", False))
    if debug_trace_enabled:
        runner._log_event(
            "eval",
            "real_metrics_snapshot",
            LogLevel.DEBUG,
            {
                "preview_n": preview_n,
                "final_n": final_n,
                "calibration_len": (
                    len(calibration_data)
                    if hasattr(calibration_data, "__len__")
                    else "n/a"
                ),
            },
        )

    device = _resolve_eval_device(model, config, torch_mod)
    process = psutil_module.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024

    policy_flags = runner._resolve_policy_flags(config)
    window_selection = _select_eval_windows(
        runner,
        calibration_data,
        preview_n=preview_n,
        final_n=final_n,
        allow_materialize=policy_flags["allow_calibration_materialize"],
    )
    preview_data = window_selection.preview_data
    final_data = window_selection.final_data

    eval_context: dict[str, Any] = {}
    if config and isinstance(getattr(config, "context", None), dict):
        eval_context = config.context.get("eval", {}) or {}

    loss_cfg = eval_context.get("loss", {}) if isinstance(eval_context, dict) else {}
    resolved_loss_mode = str(
        loss_cfg.get("resolved_type") or loss_cfg.get("type") or ""
    ).lower()

    if _has_multimodal_batches(preview_data, final_data):
        return _build_multimodal_eval_result(
            model,
            list(preview_data),
            list(final_data),
            adapter=adapter,
            device=device,
            config=config,
            process=process,
            initial_memory=initial_memory,
        )

    bootstrap_cfg = eval_context.get("bootstrap", {}) or {}
    bootstrap_enabled = bool(bootstrap_cfg.get("enabled", True))
    bootstrap_method = str(bootstrap_cfg.get("method", "bca_paired_delta_log")).lower()
    bootstrap_replicates = int(
        bootstrap_cfg.get("replicates", bootstrap_cfg.get("n", 1000) or 1000)
    )
    bootstrap_alpha = float(bootstrap_cfg.get("alpha", 0.05) or 0.05)
    bootstrap_seed_cfg = bootstrap_cfg.get("seed")
    ci_band = float(bootstrap_cfg.get("ci_band", 0.10) or 0.10)

    if bootstrap_method == "percentile":
        single_method = delta_method = "percentile"
    elif bootstrap_method == "bca_paired_delta_log":
        single_method = delta_method = "bca"
    else:
        single_method = bootstrap_method
        delta_method = bootstrap_method

    dataset_seed = None
    profile_label = ""
    pairing_context: dict[str, Any] = {}
    if config and isinstance(config.context, dict):
        dataset_cfg = config.context.get("dataset", {})
        if isinstance(dataset_cfg, dict):
            dataset_seed = dataset_cfg.get("seed")
        profile_label = str(config.context.get("profile", "")).lower()
        pairing_context = config.context.get("pairing_baseline", {}) or {}

    bootstrap_seed = (
        bootstrap_seed_cfg if bootstrap_seed_cfg is not None else dataset_seed
    )
    try:
        bootstrap_seed = int(bootstrap_seed) if bootstrap_seed is not None else 0
    except (TypeError, ValueError):
        bootstrap_seed = 0
    if bootstrap_replicates <= 0:
        bootstrap_enabled = False
    if not (0.0 < bootstrap_alpha < 1.0):
        bootstrap_alpha = 0.05

    return _EvalRuntimeContext(
        device=device,
        process=process,
        initial_memory=float(initial_memory),
        preview_data=preview_data,
        final_data=final_data,
        preview_n=window_selection.preview_n,
        final_n=window_selection.final_n,
        resolved_loss_mode=resolved_loss_mode,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_method=bootstrap_method,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_alpha=bootstrap_alpha,
        bootstrap_seed=int(bootstrap_seed),
        ci_band=ci_band,
        single_method=single_method,
        delta_method=delta_method,
        profile_label=profile_label,
        pairing_context=pairing_context,
    )


def _compute_slice_metrics(
    runner: Any,
    model: Any,
    runtime: _EvalRuntimeContext,
) -> _SliceMetricsResult:
    preview_limit = resolve_limit(runtime.preview_data, runtime.preview_n)
    final_limit = resolve_limit(runtime.final_data, runtime.final_n)

    preview_summary, preview_error = compute_slice_summary(
        runner,
        model,
        runtime.preview_data,
        max_batches=preview_limit,
        start_idx=0,
        device=runtime.device,
        resolved_loss_mode=runtime.resolved_loss_mode,
    )
    final_summary, final_error = compute_slice_summary(
        runner,
        model,
        runtime.final_data,
        max_batches=final_limit,
        start_idx=preview_summary["num_batches"],
        device=runtime.device,
        resolved_loss_mode=runtime.resolved_loss_mode,
    )
    eval_error = preview_error or final_error

    preview_raw_losses = preview_summary["log_losses"]
    final_raw_losses = final_summary["log_losses"]
    paired_windows_attempted = min(len(preview_raw_losses), len(final_raw_losses))

    preview_log_losses = [
        float(loss) for loss in preview_raw_losses if math.isfinite(loss)
    ]
    final_log_losses = [float(loss) for loss in final_raw_losses if math.isfinite(loss)]
    if len(preview_log_losses) != len(preview_raw_losses):
        runner._log_event(
            "eval",
            "non_finite_preview_losses_filtered",
            LogLevel.WARNING,
            {
                "total": len(preview_raw_losses),
                "filtered": len(preview_raw_losses) - len(preview_log_losses),
            },
        )
    if len(final_log_losses) != len(final_raw_losses):
        runner._log_event(
            "eval",
            "non_finite_final_losses_filtered",
            LogLevel.WARNING,
            {
                "total": len(final_raw_losses),
                "filtered": len(final_raw_losses) - len(final_log_losses),
            },
        )

    preview_tokens_ct = preview_summary["total_tokens"]
    final_tokens_ct = final_summary["total_tokens"]
    preview_batches_ct = preview_summary["num_batches"]
    final_batches_ct = final_summary["num_batches"]
    preview_window_ids = list(preview_summary["window_ids"])
    final_window_ids = list(final_summary["window_ids"])
    preview_tokens = list(preview_summary["tokens"])
    final_tokens = list(final_summary["tokens"])
    preview_token_counts = list(preview_summary.get("window_token_counts", []))
    final_token_counts = list(final_summary.get("window_token_counts", []))
    preview_attention_masks = list(preview_summary.get("attention_masks", []))
    final_attention_masks = list(final_summary.get("attention_masks", []))
    preview_mask_counts = list(preview_summary.get("masked_token_counts", []))
    final_mask_counts = list(final_summary.get("masked_token_counts", []))
    preview_labels = list(preview_summary.get("labels", []))
    final_labels = list(final_summary.get("labels", []))
    preview_actual_token_counts = list(preview_summary.get("actual_token_counts", []))
    final_actual_token_counts = list(final_summary.get("actual_token_counts", []))
    preview_actual_tokens_ct = int(
        preview_summary.get("actual_total_tokens", preview_tokens_ct)
    )
    final_actual_tokens_ct = int(
        final_summary.get("actual_total_tokens", final_tokens_ct)
    )
    preview_masked_total = (
        sum(preview_mask_counts) if preview_mask_counts else int(preview_tokens_ct)
    )
    final_masked_total = (
        sum(final_mask_counts) if final_mask_counts else int(final_tokens_ct)
    )
    preview_weighted_loss = float(preview_summary.get("weighted_log_loss", 0.0))
    final_weighted_loss = float(final_summary.get("weighted_log_loss", 0.0))

    if preview_tokens_ct > 0:
        preview_mean_log = float(preview_weighted_loss / preview_tokens_ct)
        pm_preview = math.exp(preview_mean_log)
    elif preview_log_losses:
        preview_mean_log = float(np.mean(preview_log_losses))
        pm_preview = math.exp(preview_mean_log)
    else:
        pm_preview = preview_summary["ppl"]
        if not math.isfinite(pm_preview) or pm_preview <= 0:
            pm_preview = float("nan")
        preview_mean_log = math.log(pm_preview)

    if final_tokens_ct > 0:
        final_mean_log = float(final_weighted_loss / final_tokens_ct)
        pm_final = math.exp(final_mean_log)
    elif final_log_losses:
        final_mean_log = float(np.mean(final_log_losses))
        pm_final = math.exp(final_mean_log)
    else:
        pm_final = final_summary["ppl"]
        if not math.isfinite(pm_final) or pm_final <= 0:
            pm_final = float("nan")
        final_mean_log = math.log(pm_final)

    delta_mean_log = final_mean_log - preview_mean_log
    ppl_ratio = math.exp(delta_mean_log)
    pm_invalid = False
    try:
        if not (math.isfinite(delta_mean_log) and math.isfinite(ppl_ratio)):
            raise RuntimeError("non_finite_primary_metric")
        expected_ratio = math.exp(delta_mean_log)
        if abs(ppl_ratio - expected_ratio) > 1e-6:
            raise RuntimeError("primary_metric_ratio_mismatch")
    except (ArithmeticError, TypeError, ValueError, RuntimeError) as exc:
        pm_invalid = True
        runner._log_event(
            "eval",
            "primary_metric_invalid",
            LogLevel.WARNING,
            {
                "pm_preview": float(pm_preview),
                "pm_final": float(pm_final),
                "delta_mean_log": float(delta_mean_log),
                "pm_ratio": float(ppl_ratio),
                "error": str(exc),
            },
        )

    return _SliceMetricsResult(
        eval_error=eval_error,
        preview_log_losses=preview_log_losses,
        final_log_losses=final_log_losses,
        preview_tokens_ct=int(preview_tokens_ct),
        final_tokens_ct=int(final_tokens_ct),
        preview_batches_ct=int(preview_batches_ct),
        final_batches_ct=int(final_batches_ct),
        preview_window_ids=preview_window_ids,
        final_window_ids=final_window_ids,
        preview_tokens=preview_tokens,
        final_tokens=final_tokens,
        preview_token_counts=preview_token_counts,
        final_token_counts=final_token_counts,
        preview_attention_masks=preview_attention_masks,
        final_attention_masks=final_attention_masks,
        preview_mask_counts=preview_mask_counts,
        final_mask_counts=final_mask_counts,
        preview_labels=preview_labels,
        final_labels=final_labels,
        preview_actual_token_counts=preview_actual_token_counts,
        final_actual_token_counts=final_actual_token_counts,
        preview_actual_tokens_ct=int(preview_actual_tokens_ct),
        final_actual_tokens_ct=int(final_actual_tokens_ct),
        preview_masked_total=int(preview_masked_total),
        final_masked_total=int(final_masked_total),
        preview_mean_log=float(preview_mean_log),
        final_mean_log=float(final_mean_log),
        delta_mean_log=float(delta_mean_log),
        pm_preview=float(pm_preview),
        pm_final=float(pm_final),
        ppl_ratio=float(ppl_ratio),
        paired_windows_attempted=int(paired_windows_attempted),
        pm_invalid=bool(pm_invalid),
    )


def _build_real_metrics_payload(
    model: Any,
    runtime: _EvalRuntimeContext,
    slices: _SliceMetricsResult,
    bootstrap: _BootstrapDeltaResult,
    pairing: _PairingCoverageResult,
) -> tuple[dict[str, Any], dict[str, Any]]:
    latency_ms_per_tok = measure_latency(
        model,
        runtime.preview_data[:1] if runtime.preview_data else runtime.final_data[:1],
        runtime.device,
    )
    current_memory = runtime.process.memory_info().rss / 1024 / 1024
    peak_memory = max(runtime.initial_memory, current_memory)

    eval_samples = int(slices.preview_batches_ct) + int(slices.final_batches_ct)
    total_tokens = int(slices.preview_actual_tokens_ct) + int(
        slices.final_actual_tokens_ct
    )
    masked_total_tokens = int(slices.preview_masked_total) + int(
        slices.final_masked_total
    )

    paired_windows_count = (
        slices.paired_windows_attempted
        if slices.paired_windows_attempted
        else len(bootstrap.delta_samples)
    )
    unweighted_delta_mean = (
        float(np.mean(bootstrap.delta_samples))
        if bootstrap.delta_samples
        else float(slices.delta_mean_log)
    )
    preview_weighted_delta_mean: float | None = None
    if bootstrap.delta_weights:
        total_weight = float(sum(bootstrap.delta_weights))
        if total_weight > 0.0:
            preview_weighted_delta_mean = float(
                np.dot(bootstrap.delta_samples, bootstrap.delta_weights) / total_weight
            )
    paired_delta_mean = float(slices.delta_mean_log)
    paired_delta_std = (
        float(np.std(bootstrap.delta_samples, ddof=1))
        if len(bootstrap.delta_samples) > 1
        else 0.0
    )
    paired_delta_min = (
        float(min(bootstrap.delta_samples)) if bootstrap.delta_samples else None
    )
    paired_delta_max = (
        float(max(bootstrap.delta_samples)) if bootstrap.delta_samples else None
    )

    pm_kind = "ppl_causal"
    if runtime.resolved_loss_mode == "mlm":
        pm_kind = "ppl_mlm"
    elif runtime.resolved_loss_mode in {"seq2seq", "s2s", "t5"}:
        pm_kind = "ppl_seq2seq"

    metrics = {
        "primary_metric": {
            "kind": pm_kind,
            "preview": (
                float(slices.pm_preview) if math.isfinite(slices.pm_preview) else None
            ),
            "final": float(slices.pm_final) if math.isfinite(slices.pm_final) else None,
            "invalid": bool(bootstrap.pm_invalid),
            "degraded": bool(bootstrap.pm_invalid or bootstrap.degraded_reason),
            "degraded_reason": bootstrap.degraded_reason,
        },
        "logloss_preview": float(slices.preview_mean_log),
        "logloss_final": float(slices.final_mean_log),
        "logloss_delta": float(slices.delta_mean_log),
        "logloss_preview_ci": tuple(map(float, bootstrap.preview_log_ci)),
        "logloss_final_ci": tuple(map(float, bootstrap.final_log_ci)),
        "logloss_delta_ci": tuple(map(float, bootstrap.delta_log_ci)),
        "latency_ms_per_tok": latency_ms_per_tok,
        "memory_mb_peak": peak_memory,
        "eval_samples": eval_samples,
        "total_tokens": total_tokens,
        "preview_total_tokens": int(slices.preview_actual_tokens_ct),
        "final_total_tokens": int(slices.final_actual_tokens_ct),
        "masked_tokens_total": masked_total_tokens,
        "masked_tokens_preview": int(slices.preview_masked_total),
        "masked_tokens_final": int(slices.final_masked_total),
        "reduction": {
            "mode": "token_mean",
            "implementation": "huggingface_cross_entropy",
        },
        "window_overlap_fraction": float(pairing.window_overlap_fraction),
        "window_match_fraction": float(pairing.window_match_fraction),
        "window_pairing_reason": pairing.pairing_reason,
        "window_pairing_preview": {
            "matched": pairing.preview_pair_stats["matched"],
            "expected": pairing.preview_pair_stats["expected"],
            "reason": pairing.preview_pair_stats.get("reason"),
        },
        "window_pairing_final": {
            "matched": pairing.final_pair_stats["matched"],
            "expected": pairing.final_pair_stats["expected"],
            "reason": pairing.final_pair_stats.get("reason"),
        },
        "bootstrap": pairing.bootstrap_info,
        "paired_windows": paired_windows_count,
        "paired_delta_summary": {
            "mean": paired_delta_mean,
            "mean_unweighted": unweighted_delta_mean,
            "mean_preview_weighted": (
                preview_weighted_delta_mean
                if preview_weighted_delta_mean is not None
                else unweighted_delta_mean
            ),
            "std": paired_delta_std,
            "min": paired_delta_min,
            "max": paired_delta_max,
            "degenerate": bootstrap.degenerate_delta,
            "degenerate_reason": bootstrap.degenerate_reason,
        },
    }
    if slices.eval_error:
        metrics["eval_error"] = slices.eval_error

    eval_windows = {
        "preview": {
            "window_ids": slices.preview_window_ids[: runtime.preview_n],
            "logloss": list(slices.preview_log_losses),
            "input_ids": slices.preview_tokens,
            "attention_masks": slices.preview_attention_masks,
            "token_counts": slices.preview_token_counts,
            "masked_token_counts": slices.preview_mask_counts,
            "actual_token_counts": slices.preview_actual_token_counts,
            "labels": slices.preview_labels,
        },
        "final": {
            "window_ids": slices.final_window_ids[: runtime.final_n],
            "logloss": list(slices.final_log_losses),
            "input_ids": slices.final_tokens,
            "attention_masks": slices.final_attention_masks,
            "token_counts": slices.final_token_counts,
            "masked_token_counts": slices.final_mask_counts,
            "actual_token_counts": slices.final_actual_token_counts,
            "labels": slices.final_labels,
        },
    }
    return metrics, eval_windows


def compute_real_metrics(
    runner: Any,
    model: Any,
    calibration_data: Any,
    adapter: Any,
    preview_n: int | None = None,
    final_n: int | None = None,
    config: Any | None = None,
    *,
    compute_paired_delta_log_ci_fn: Any = compute_paired_delta_log_ci,
    logspace_to_ratio_ci_fn: Any = logspace_to_ratio_ci,
    coverage_requirements: Any = BOOTSTRAP_COVERAGE_REQUIREMENTS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute evaluation metrics from calibration data."""
    import torch

    runtime = _resolve_eval_runtime_context(
        runner,
        model,
        calibration_data,
        adapter,
        preview_n=preview_n,
        final_n=final_n,
        config=config,
        torch_mod=torch,
    )
    if not isinstance(runtime, _EvalRuntimeContext):
        return runtime

    slices = _compute_slice_metrics(runner, model, runtime)
    bootstrap = _compute_bootstrap_delta_stats(
        runner,
        runtime,
        slices,
        compute_paired_delta_log_ci_fn=compute_paired_delta_log_ci_fn,
        logspace_to_ratio_ci_fn=logspace_to_ratio_ci_fn,
    )
    try:
        pairing = _evaluate_pairing_and_coverage(
            runner,
            runtime,
            slices,
            config=config,
            coverage_requirements=coverage_requirements,
            compute_window_pairing_metrics_fn=compute_window_pairing_metrics,
            assess_bootstrap_coverage_fn=assess_bootstrap_coverage,
        )
    except InvarlockError as exc:
        slices = replace(
            slices,
            eval_error={"type": type(exc).__name__, "message": str(exc)},
        )
        pairing = _pairing_error_result(runtime)
    except RuntimeError as exc:
        message = str(exc)
        if not message.startswith(
            (
                "Window pairing mismatch detected",
                "Window overlap detected",
                "Window count mismatch detected",
            )
        ):
            raise
        slices = replace(
            slices,
            eval_error={"type": type(exc).__name__, "message": message},
        )
        pairing = _pairing_error_result(runtime)
    return _build_real_metrics_payload(model, runtime, slices, bootstrap, pairing)


__all__ = [
    "compute_real_metrics",
    "measure_latency",
    "samples_to_dataloader",
    "_raise_latency_error",
    "_evaluate_vision_text_arm",
    "_has_multimodal_batches",
    "_is_multimodal_batch",
    "_model_kwargs",
    "_normalize_answer_text",
    "_normalize_reference_answers",
    "_prediction_answer_text",
    "_resolve_adapter_hook",
    "_resolve_metric_kind",
]
