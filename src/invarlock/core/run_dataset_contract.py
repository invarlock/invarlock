from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from invarlock.eval.data_support import DatasetDiagnostic


def _cfg_auto_tier(cfg: Any) -> str | None:
    section_fn = getattr(cfg, "section", None)
    section = None
    if callable(section_fn):
        try:
            section = section_fn("auto")
        except (AttributeError, KeyError, TypeError):
            section = None
    if isinstance(section, dict):
        tier = section.get("tier")
    else:
        try:
            tier = cfg.auto.tier
        except (AttributeError, KeyError, TypeError):
            tier = None
    return str(tier) if isinstance(tier, str) and tier else None


@dataclass(frozen=True)
class RunDatasetContractResult:
    resolved_split: str | None
    used_fallback_split: bool
    tokenizer: Any
    tokenizer_hash: str | None
    calibration_data: list[dict[str, Any]]
    dataset_meta: dict[str, Any]
    window_plan: dict[str, Any] | None
    preview_count: int
    final_count: int
    effective_preview: int
    effective_final: int
    preview_mask_counts: list[int]
    final_mask_counts: list[int]
    preview_records: list[dict[str, Any]]
    final_records: list[dict[str, Any]]
    diagnostics: tuple[DatasetDiagnostic, ...] = ()


def materialize_run_dataset(
    *,
    pairing_schedule: dict[str, Any] | None,
    cfg: Any,
    baseline_report_data: dict[str, Any] | None,
    tokenizer_hash: str | None,
    resolved_loss_type: str,
    profile: str | None,
    model_profile: Any,
    tokenizer: Any,
    use_mlm: bool,
    mask_prob: float,
    mask_seed: int,
    random_token_prob: float,
    original_token_prob: float,
    tier: str | None,
    requested_preview: int | None,
    requested_final: int | None,
    effective_preview: int,
    effective_final: int,
    resolved_device: str,
    profile_normalized: str | None,
    resolved_split: str | None,
    validate_and_harvest_baseline_schedule_fn: Any,
    materialize_baseline_pairing_schedule_fn: Any,
    resolve_tokenizer_fn: Any,
    build_provider_dataset_plan_fn: Any,
) -> RunDatasetContractResult:
    if pairing_schedule:
        harvested = validate_and_harvest_baseline_schedule_fn(
            cfg,
            pairing_schedule,
            baseline_report_data,
            tokenizer_hash=tokenizer_hash,
            resolved_loss_type=resolved_loss_type,
            profile=profile,
            typed_failures=True,
        )
        dataset_meta = harvested["dataset_meta"]
        window_plan = harvested["window_plan"]
        calibration_data = harvested["calibration_data"]
        resolved_tokenizer = tokenizer
        resolved_tokenizer_hash = tokenizer_hash

        if use_mlm and resolved_tokenizer is None:
            resolved_tokenizer, resolved_tokenizer_hash = resolve_tokenizer_fn(
                model_profile
            )

        materialized_baseline = materialize_baseline_pairing_schedule_fn(
            pairing_schedule=pairing_schedule,
            calibration_data=calibration_data,
            dataset_meta=dataset_meta,
            window_plan=window_plan,
            tokenizer=resolved_tokenizer,
            use_mlm=use_mlm,
            mask_prob=mask_prob,
            mask_seed=mask_seed,
            random_token_prob=random_token_prob,
            original_token_prob=original_token_prob,
            resolved_tier=tier or _cfg_auto_tier(cfg),
            profile=profile,
        )
        return RunDatasetContractResult(
            resolved_split=None,
            used_fallback_split=False,
            tokenizer=resolved_tokenizer,
            tokenizer_hash=resolved_tokenizer_hash,
            calibration_data=materialized_baseline.calibration_data,
            dataset_meta=materialized_baseline.dataset_meta,
            window_plan=materialized_baseline.window_plan,
            preview_count=materialized_baseline.preview_count,
            final_count=materialized_baseline.final_count,
            effective_preview=materialized_baseline.effective_preview,
            effective_final=materialized_baseline.effective_final,
            preview_mask_counts=materialized_baseline.preview_mask_counts,
            final_mask_counts=materialized_baseline.final_mask_counts,
            preview_records=[],
            final_records=[],
            diagnostics=(),
        )

    if cfg.dataset.provider:
        dataset_plan = build_provider_dataset_plan_fn(
            cfg=cfg,
            model_profile=model_profile,
            resolved_device=resolved_device,
            profile=profile,
            profile_normalized=profile_normalized,
            requested_preview=requested_preview,
            requested_final=requested_final,
            effective_preview=effective_preview,
            effective_final=effective_final,
            pairing_schedule_present=bool(pairing_schedule),
            use_mlm=use_mlm,
            mask_prob=mask_prob,
            mask_seed=mask_seed,
            random_token_prob=random_token_prob,
            original_token_prob=original_token_prob,
            resolved_loss_type=resolved_loss_type,
            tier=tier,
        )
        return RunDatasetContractResult(
            resolved_split=dataset_plan.resolved_split,
            used_fallback_split=dataset_plan.used_fallback_split,
            tokenizer=dataset_plan.tokenizer,
            tokenizer_hash=dataset_plan.tokenizer_hash,
            calibration_data=dataset_plan.calibration_data,
            dataset_meta=dataset_plan.dataset_meta,
            window_plan=dataset_plan.window_plan,
            preview_count=dataset_plan.preview_count,
            final_count=dataset_plan.final_count,
            effective_preview=dataset_plan.effective_preview,
            effective_final=dataset_plan.effective_final,
            preview_mask_counts=dataset_plan.preview_mask_counts,
            final_mask_counts=dataset_plan.final_mask_counts,
            preview_records=dataset_plan.preview_records,
            final_records=dataset_plan.final_records,
            diagnostics=tuple(dataset_plan.diagnostics),
        )

    return RunDatasetContractResult(
        resolved_split=resolved_split,
        used_fallback_split=False,
        tokenizer=tokenizer,
        tokenizer_hash=tokenizer_hash,
        calibration_data=[],
        dataset_meta={},
        window_plan=None,
        preview_count=effective_preview,
        final_count=effective_final,
        effective_preview=effective_preview,
        effective_final=effective_final,
        preview_mask_counts=[],
        final_mask_counts=[],
        preview_records=[],
        final_records=[],
        diagnostics=(),
    )


__all__ = ["RunDatasetContractResult", "materialize_run_dataset"]
