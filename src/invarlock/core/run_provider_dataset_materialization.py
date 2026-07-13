from __future__ import annotations

import hashlib
from typing import Any

from .dataset_identity import dataset_identity_from_provider
from .run_provider_dataset_plan import (
    ApplyMlmMasksFn,
    HashSequencesFn,
    ProviderDatasetPlanDiagnostic,
    ProviderDatasetPlanResult,
    ResolvePmMinTokensTargetFn,
    SafeIntFn,
    TensorOrListToIntsFn,
    TokenizerDigestFn,
)


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _provider_labels(data_provider: Any) -> tuple[Any, Any]:
    try:
        return (
            getattr(data_provider, "last_preview_labels", None),
            getattr(data_provider, "last_final_labels", None),
        )
    except (AttributeError, TypeError):
        return None, None


def _apply_provider_preview_labels(
    preview_records: list[dict[str, Any]],
    provider_labels_prev: Any,
    *,
    tensor_or_list_to_ints_fn: TensorOrListToIntsFn,
) -> None:
    for idx_local, record in enumerate(preview_records):
        if provider_labels_prev is not None and idx_local < len(provider_labels_prev):
            record["labels"] = tensor_or_list_to_ints_fn(
                provider_labels_prev[idx_local]
            )


def _ensure_window_plan(
    window_plan: dict[str, Any] | None,
    *,
    requested_preview: int,
    requested_final: int,
    preview_count: int,
    final_count: int,
    preview_total_tokens: int,
    final_total_tokens: int,
    min_tokens_target: int,
    tokens_floor_met: bool,
    profile: str | None,
    dedupe_adjustments: Any,
) -> dict[str, Any]:
    if window_plan is None:
        window_plan = {
            "profile": (profile or "").lower() or "default",
            "requested_preview": int(requested_preview),
            "requested_final": int(requested_final),
            "capacity": {},
        }
    window_plan["actual_preview"] = int(preview_count)
    window_plan["actual_final"] = int(final_count)
    window_plan["coverage_ok"] = (
        window_plan.get("coverage_ok", True) and preview_count == final_count
    )
    window_plan["preview_total_tokens"] = int(preview_total_tokens)
    window_plan["final_total_tokens"] = int(final_total_tokens)
    window_plan["min_tokens_target"] = int(min_tokens_target)
    window_plan["tokens_floor_met"] = bool(tokens_floor_met)
    if dedupe_adjustments:
        window_plan["dedupe_adjustments"] = list(dedupe_adjustments)
    return window_plan


def _apply_mlm_if_needed(
    *,
    use_mlm: bool,
    preview_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    tokenizer: Any,
    mask_prob: float,
    mask_seed: int,
    random_token_prob: float,
    original_token_prob: float,
    apply_mlm_masks_fn: ApplyMlmMasksFn,
) -> tuple[int, int, list[int], list[int]]:
    if not use_mlm:
        return 0, 0, [0] * len(preview_records), [0] * len(final_records)
    preview_mask_total, preview_mask_counts = apply_mlm_masks_fn(
        preview_records,
        tokenizer=tokenizer,
        mask_prob=mask_prob,
        seed=mask_seed,
        random_token_prob=random_token_prob,
        original_token_prob=original_token_prob,
        prefix="preview",
    )
    final_mask_total, final_mask_counts = apply_mlm_masks_fn(
        final_records,
        tokenizer=tokenizer,
        mask_prob=mask_prob,
        seed=mask_seed,
        random_token_prob=random_token_prob,
        original_token_prob=original_token_prob,
        prefix="final",
    )
    return preview_mask_total, final_mask_total, preview_mask_counts, final_mask_counts


def _calibration_entry(
    record: dict[str, Any],
    *,
    arm: str,
    index: int,
    use_mlm: bool,
    provider_labels: Any,
    tensor_or_list_to_ints_fn: TensorOrListToIntsFn,
) -> dict[str, Any]:
    entry = {
        "input_ids": record["input_ids"],
        "attention_mask": record["attention_mask"],
        "window_id": f"{arm}::{index}",
        "dataset_index": record.get("dataset_index"),
        "mlm_masked": record.get("mlm_masked", 0),
    }
    if use_mlm:
        entry["labels"] = record.get("labels", [-100] * len(record["input_ids"]))
    elif provider_labels is not None and index < len(provider_labels):
        entry["labels"] = tensor_or_list_to_ints_fn(provider_labels[index])
    return entry


def _build_calibration_data(
    *,
    preview_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    use_mlm: bool,
    provider_labels_prev: Any,
    provider_labels_fin: Any,
    tensor_or_list_to_ints_fn: TensorOrListToIntsFn,
) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
    calibration_data: list[dict[str, Any]] = []
    preview_sequences = [record["input_ids"] for record in preview_records]
    for idx, record in enumerate(preview_records):
        calibration_data.append(
            _calibration_entry(
                record,
                arm="preview",
                index=idx,
                use_mlm=use_mlm,
                provider_labels=provider_labels_prev,
                tensor_or_list_to_ints_fn=tensor_or_list_to_ints_fn,
            )
        )

    final_sequences = [record["input_ids"] for record in final_records]
    for idx, record in enumerate(final_records):
        calibration_data.append(
            _calibration_entry(
                record,
                arm="final",
                index=idx,
                use_mlm=use_mlm,
                provider_labels=provider_labels_fin,
                tensor_or_list_to_ints_fn=tensor_or_list_to_ints_fn,
            )
        )
    return calibration_data, preview_sequences, final_sequences


def _dataset_meta(
    *,
    data_provider: Any,
    tokenizer: Any,
    tokenizer_hash: str | None,
    preview_hash: str,
    final_hash: str,
    preview_total_tokens: int,
    final_total_tokens: int,
    min_tokens_target: int,
    tokens_floor_met: bool,
    resolved_loss_type: str,
    use_mlm: bool,
    preview_mask_total: int,
    final_mask_total: int,
    window_plan: dict[str, Any],
    include_window_plan: bool,
    tokenizer_digest_fn: TokenizerDigestFn,
    safe_int_fn: SafeIntFn,
) -> dict[str, Any]:
    dataset_meta = {
        "tokenizer_name": _optional_text(getattr(tokenizer, "name_or_path", None)),
        "tokenizer_hash": (
            tokenizer_hash
            if tokenizer_hash is not None
            else tokenizer_digest_fn(tokenizer)
        ),
        "vocab_size": safe_int_fn(getattr(tokenizer, "vocab_size", 0), 0),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "add_prefix_space": getattr(tokenizer, "add_prefix_space", None),
        "dataset_hash": hashlib.blake2s(
            (preview_hash + final_hash).encode("utf-8"), digest_size=16
        ).hexdigest(),
        "preview_hash": preview_hash,
        "final_hash": final_hash,
        "preview_total_tokens": int(preview_total_tokens),
        "final_total_tokens": int(final_total_tokens),
        "min_tokens_target": int(min_tokens_target),
        "tokens_floor_met": bool(tokens_floor_met),
        "loss_type": resolved_loss_type,
    }
    dataset_meta.update(dataset_identity_from_provider(data_provider))
    if include_window_plan:
        dataset_meta["window_plan"] = window_plan
        capacity_meta = window_plan.get("capacity")
        if capacity_meta:
            dataset_meta["window_capacity"] = capacity_meta
    if use_mlm:
        dataset_meta["masked_tokens_preview"] = int(preview_mask_total)
        dataset_meta["masked_tokens_final"] = int(final_mask_total)
        dataset_meta["masked_tokens_total"] = int(preview_mask_total + final_mask_total)
    strat_stats = getattr(data_provider, "stratification_stats", None)
    if strat_stats:
        dataset_meta["stratification"] = strat_stats
    scorer_profile = getattr(data_provider, "scorer_profile", None)
    if scorer_profile:
        dataset_meta["scorer_profile"] = scorer_profile
    return dataset_meta


def _materialize_text_provider_dataset_plan(
    *,
    data_provider: Any,
    resolved_split: str,
    used_fallback_split: bool,
    tokenizer: Any,
    tokenizer_hash: str | None,
    effective_windows: dict[str, Any],
    requested_preview: int,
    requested_final: int,
    effective_preview: int,
    effective_final: int,
    resolved_loss_type: str,
    window_plan: dict[str, Any] | None,
    use_mlm: bool,
    mask_prob: float,
    mask_seed: int,
    random_token_prob: float,
    original_token_prob: float,
    tier: str | None,
    profile: str | None,
    apply_mlm_masks_fn: ApplyMlmMasksFn,
    resolve_pm_min_tokens_target_fn: ResolvePmMinTokensTargetFn,
    hash_sequences_fn: HashSequencesFn,
    tokenizer_digest_fn: TokenizerDigestFn,
    safe_int_fn: SafeIntFn,
    tensor_or_list_to_ints_fn: TensorOrListToIntsFn,
    diagnostics: list[ProviderDatasetPlanDiagnostic],
) -> ProviderDatasetPlanResult:
    preview_records = list(effective_windows["preview_records"])
    final_records = list(effective_windows["final_records"])
    preview_count = int(effective_windows["actual_preview"])
    final_count = int(effective_windows["actual_final"])
    effective_preview = preview_count
    effective_final = final_count

    provider_labels_prev, provider_labels_fin = _provider_labels(data_provider)
    _apply_provider_preview_labels(
        preview_records,
        provider_labels_prev,
        tensor_or_list_to_ints_fn=tensor_or_list_to_ints_fn,
    )

    min_tokens_target = resolve_pm_min_tokens_target_fn(
        tier=tier or None,
        profile=profile,
    )
    preview_total_tokens = int(effective_windows["preview_total_tokens"])
    final_total_tokens = int(effective_windows["final_total_tokens"])
    tokens_floor_met = (preview_total_tokens + final_total_tokens) >= int(
        min_tokens_target
    )
    window_plan = _ensure_window_plan(
        window_plan,
        requested_preview=requested_preview,
        requested_final=requested_final,
        preview_count=preview_count,
        final_count=final_count,
        preview_total_tokens=preview_total_tokens,
        final_total_tokens=final_total_tokens,
        min_tokens_target=int(min_tokens_target),
        tokens_floor_met=tokens_floor_met,
        profile=profile,
        dedupe_adjustments=effective_windows["dedupe_adjustments"],
    )

    preview_mask_total, final_mask_total, preview_mask_counts, final_mask_counts = (
        _apply_mlm_if_needed(
            use_mlm=use_mlm,
            preview_records=preview_records,
            final_records=final_records,
            tokenizer=tokenizer,
            mask_prob=mask_prob,
            mask_seed=mask_seed,
            random_token_prob=random_token_prob,
            original_token_prob=original_token_prob,
            apply_mlm_masks_fn=apply_mlm_masks_fn,
        )
    )
    calibration_data, preview_sequences, final_sequences = _build_calibration_data(
        preview_records=preview_records,
        final_records=final_records,
        use_mlm=use_mlm,
        provider_labels_prev=provider_labels_prev,
        provider_labels_fin=provider_labels_fin,
        tensor_or_list_to_ints_fn=tensor_or_list_to_ints_fn,
    )

    preview_hash = hash_sequences_fn(preview_sequences)
    final_hash = hash_sequences_fn(final_sequences)
    dataset_meta = _dataset_meta(
        data_provider=data_provider,
        tokenizer=tokenizer,
        tokenizer_hash=tokenizer_hash,
        preview_hash=preview_hash,
        final_hash=final_hash,
        preview_total_tokens=preview_total_tokens,
        final_total_tokens=final_total_tokens,
        min_tokens_target=int(min_tokens_target),
        tokens_floor_met=tokens_floor_met,
        resolved_loss_type=resolved_loss_type,
        use_mlm=use_mlm,
        preview_mask_total=preview_mask_total,
        final_mask_total=final_mask_total,
        window_plan=window_plan,
        include_window_plan=bool(window_plan),
        tokenizer_digest_fn=tokenizer_digest_fn,
        safe_int_fn=safe_int_fn,
    )

    return ProviderDatasetPlanResult(
        data_provider=data_provider,
        resolved_split=resolved_split,
        used_fallback_split=used_fallback_split,
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
        diagnostics=tuple(diagnostics),
    )
