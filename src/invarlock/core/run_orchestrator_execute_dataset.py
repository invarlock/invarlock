"""Dataset preparation helpers for run orchestration execution."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

from invarlock.core.run_orchestrator_execute_helpers import (
    _RunDatasetState,
)
from invarlock.core.run_orchestrator_types import (
    RunCalibrationBatchSizesDebugEvent,
    RunDatasetLoadingEvent,
    RunMaskedTokensDebugEvent,
    RunPreviewLabelsDebugEvent,
)


def _emit_dataset_debug_trace(
    *,
    emit: Any,
    calibration_data: list[dict[str, Any]],
    preview_count: int,
    final_count: int,
    use_mlm: bool,
) -> None:
    if not os.environ.get("INVARLOCK_DEBUG_TRACE"):
        return
    emit(
        RunCalibrationBatchSizesDebugEvent(
            preview_count=int(preview_count),
            final_count=int(final_count),
            total_count=len(calibration_data),
        )
    )
    if not (use_mlm and calibration_data):
        return
    masked_preview = sum(
        entry.get("mlm_masked", 0) for entry in calibration_data[:preview_count]
    )
    masked_final = sum(
        entry.get("mlm_masked", 0) for entry in calibration_data[preview_count:]
    )
    emit(
        RunMaskedTokensDebugEvent(
            preview_masked=int(masked_preview),
            final_masked=int(masked_final),
        )
    )
    emit(RunPreviewLabelsDebugEvent(labels=tuple(calibration_data[0]["labels"][:10])))


def _load_dataset_state(
    *,
    cfg: Any,
    model_profile: Any,
    resolved_device: Any,
    profile: str | None,
    profile_normalized: str,
    requested_preview: int,
    requested_final: int,
    effective_preview: int,
    effective_final: int,
    use_mlm: bool,
    mask_prob: float,
    mask_seed: int,
    random_token_prob: float,
    original_token_prob: float,
    resolved_loss_type: str,
    tier: str | None,
    baseline_report_data: dict[str, Any] | None,
    tokenizer: Any | None,
    tokenizer_hash: str | None,
    resolved_split: str,
    pairing_schedule: dict[str, Any] | None,
    collect_timings: bool,
    timings: dict[str, float],
    run_context: dict[str, Any],
    materialize_run_dataset_fn: Any,
    validate_and_harvest_baseline_schedule_fn: Any,
    materialize_baseline_pairing_schedule_fn: Any,
    resolve_tokenizer_fn: Any,
    build_provider_dataset_plan_fn: Any,
    emit: Any,
    emit_transition: Any,
    fail_run: Any,
) -> _RunDatasetState:
    calibration_data: list[dict[str, Any]] = []
    dataset_meta: dict[str, Any] = {}
    window_plan: dict[str, Any] | None = None
    preview_records: list[dict[str, Any]] = []
    final_records: list[dict[str, Any]] = []
    preview_mask_counts: list[int] = []
    final_mask_counts: list[int] = []
    preview_count = effective_preview
    final_count = effective_final
    dataset_timing_start: float | None = perf_counter() if collect_timings else None
    used_fallback_split = False
    if pairing_schedule or cfg.dataset.provider:
        if not pairing_schedule:
            emit(RunDatasetLoadingEvent(provider=str(cfg.dataset.provider)))
        try:
            dataset_result = materialize_run_dataset_fn(
                pairing_schedule=pairing_schedule,
                cfg=cfg,
                model_profile=model_profile,
                resolved_device=resolved_device,
                profile=profile,
                profile_normalized=profile_normalized,
                requested_preview=requested_preview,
                requested_final=requested_final,
                effective_preview=effective_preview,
                effective_final=effective_final,
                use_mlm=use_mlm,
                mask_prob=mask_prob,
                mask_seed=mask_seed,
                random_token_prob=random_token_prob,
                original_token_prob=original_token_prob,
                resolved_loss_type=resolved_loss_type,
                tier=tier,
                baseline_report_data=baseline_report_data,
                tokenizer=tokenizer,
                tokenizer_hash=tokenizer_hash,
                resolved_split=resolved_split,
                validate_and_harvest_baseline_schedule_fn=(
                    validate_and_harvest_baseline_schedule_fn
                ),
                materialize_baseline_pairing_schedule_fn=(
                    materialize_baseline_pairing_schedule_fn
                ),
                resolve_tokenizer_fn=resolve_tokenizer_fn,
                build_provider_dataset_plan_fn=build_provider_dataset_plan_fn,
            )
        except ValueError as exc:
            fail_run(str(exc), error=exc)
        except (
            ImportError,
            ModuleNotFoundError,
            AttributeError,
            RuntimeError,
            TypeError,
        ) as exc:
            fail_run(str(exc), error=exc)
        for diagnostic in dataset_result.diagnostics:
            emit_transition("dataset", diagnostic)
        resolved_split = dataset_result.resolved_split
        used_fallback_split = dataset_result.used_fallback_split
        tokenizer = dataset_result.tokenizer
        tokenizer_hash = dataset_result.tokenizer_hash
        calibration_data = dataset_result.calibration_data
        dataset_meta = dataset_result.dataset_meta
        window_plan = dataset_result.window_plan
        preview_count = dataset_result.preview_count
        final_count = dataset_result.final_count
        effective_preview = dataset_result.effective_preview
        effective_final = dataset_result.effective_final
        preview_mask_counts = dataset_result.preview_mask_counts
        final_mask_counts = dataset_result.final_mask_counts
        preview_records = dataset_result.preview_records
        final_records = dataset_result.final_records
    try:
        run_context["dataset"]["preview_n"] = preview_count
        run_context["dataset"]["final_n"] = final_count
    except (KeyError, TypeError):
        pass
    run_context["dataset_meta"] = dataset_meta
    if window_plan:
        run_context["window_plan"] = window_plan
    if dataset_timing_start is not None:
        timings["load_dataset"] = max(0.0, float(perf_counter() - dataset_timing_start))
    _emit_dataset_debug_trace(
        emit=emit,
        calibration_data=calibration_data,
        preview_count=preview_count,
        final_count=final_count,
        use_mlm=use_mlm,
    )
    return _RunDatasetState(
        tokenizer=tokenizer,
        tokenizer_hash=tokenizer_hash,
        calibration_data=calibration_data,
        dataset_meta=dataset_meta,
        window_plan=window_plan,
        preview_count=preview_count,
        final_count=final_count,
        effective_preview=effective_preview,
        effective_final=effective_final,
        preview_mask_counts=preview_mask_counts,
        final_mask_counts=final_mask_counts,
        preview_records=preview_records,
        final_records=final_records,
        resolved_split=resolved_split,
        used_fallback_split=used_fallback_split,
    )
