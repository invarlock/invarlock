from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderDatasetPlanNotice:
    tag: str
    message: str
    emoji: str | None = None


@dataclass(frozen=True)
class ProviderDatasetPlanResult:
    data_provider: Any
    resolved_split: str
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
    notices: tuple[ProviderDatasetPlanNotice, ...]


ResolveProviderAndSplitFn = Callable[..., tuple[Any, str, bool]]
ResolveTokenizerFn = Callable[[Any], tuple[Any, str | None]]
MaybePlanReleaseWindowsFn = Callable[..., dict[str, Any]]
ResolveEffectiveWindowsFn = Callable[..., dict[str, Any]]
ApplyMlmMasksFn = Callable[..., tuple[int, list[int]]]
ResolvePmMinTokensTargetFn = Callable[..., int]
HashSequencesFn = Callable[[Any], str]
TokenizerDigestFn = Callable[[Any], str]
SafeIntFn = Callable[[Any, int], int]
TensorOrListToIntsFn = Callable[[Any], list[int]]


def _build_provider_kwargs(cfg_dataset: Any) -> dict[str, Any]:
    provider_kwargs: dict[str, Any] = {}
    for key in (
        "dataset_name",
        "config_name",
        "text_field",
        "src_field",
        "tgt_field",
        "cache_dir",
        "max_samples",
        "file",
        "path",
        "data_files",
    ):
        try:
            value = getattr(cfg_dataset, key)
        except (AttributeError, TypeError):
            value = None
        if value is not None and value != "":
            provider_kwargs[key] = value

    provider_val = getattr(cfg_dataset, "provider", None)
    if isinstance(provider_val, dict):
        for key, value in provider_val.items():
            if key != "kind" and value is not None and value != "":
                provider_kwargs[key] = value
        return provider_kwargs

    if isinstance(provider_val, str):
        return provider_kwargs

    try:
        _ = provider_val.get("kind")  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return provider_kwargs

    try:
        items = provider_val._data.items()  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        try:
            items = provider_val.items()  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            return provider_kwargs

    for key, value in items:
        if key != "kind" and value is not None and value != "":
            provider_kwargs[key] = value
    return provider_kwargs


def build_provider_dataset_plan(
    *,
    cfg: Any,
    model_profile: Any,
    console: Any,
    resolved_device: str | None,
    profile: str | None,
    profile_normalized: str | None,
    requested_preview: int,
    requested_final: int,
    effective_preview: int,
    effective_final: int,
    pairing_schedule_present: bool,
    use_mlm: bool,
    mask_prob: float,
    mask_seed: int,
    random_token_prob: float,
    original_token_prob: float,
    resolved_loss_type: str,
    tier: str | None,
    get_provider_fn: Any,
    resolve_provider_and_split_fn: ResolveProviderAndSplitFn,
    resolve_tokenizer_fn: ResolveTokenizerFn,
    maybe_plan_release_windows_fn: MaybePlanReleaseWindowsFn,
    resolve_effective_windows_fn: ResolveEffectiveWindowsFn,
    apply_mlm_masks_fn: ApplyMlmMasksFn,
    resolve_pm_min_tokens_target_fn: ResolvePmMinTokensTargetFn,
    hash_sequences_fn: HashSequencesFn,
    tokenizer_digest_fn: TokenizerDigestFn,
    safe_int_fn: SafeIntFn,
    tensor_or_list_to_ints_fn: TensorOrListToIntsFn,
) -> ProviderDatasetPlanResult:
    notices: list[ProviderDatasetPlanNotice] = []

    def _provider_notice(tag: str, message: str, emoji: str | None = None) -> None:
        notices.append(ProviderDatasetPlanNotice(tag=tag, message=message, emoji=emoji))

    provider_kwargs = _build_provider_kwargs(cfg.dataset)
    data_provider, resolved_split, used_fallback_split = resolve_provider_and_split_fn(
        cfg,
        model_profile,
        get_provider_fn=get_provider_fn,
        provider_kwargs=provider_kwargs,
        console=console,
        resolved_device=resolved_device,
        emit=_provider_notice,
    )

    tokenizer, tokenizer_hash = resolve_tokenizer_fn(model_profile)
    dataset_stride = getattr(
        cfg.dataset, "stride", getattr(cfg.dataset, "seq_len", 0) // 2
    )
    window_plan: dict[str, Any] | None = None
    release_profile = (profile or "").lower() == "release"
    if release_profile and not pairing_schedule_present:
        estimate_fn = getattr(data_provider, "estimate_capacity", None)
        if callable(estimate_fn):
            capacity_fast = bool(getattr(cfg.eval, "capacity_fast", False))
            capacity_meta = estimate_fn(
                tokenizer=tokenizer,
                seq_len=cfg.dataset.seq_len,
                stride=dataset_stride,
                split=resolved_split,
                target_total=requested_preview + requested_final,
                fast_mode=capacity_fast,
            )
            variance_policy = getattr(cfg.guards, "variance", None)
            max_calibration = (
                getattr(variance_policy, "max_calib", 0)
                if variance_policy is not None
                else 0
            )
            window_plan = maybe_plan_release_windows_fn(
                capacity_meta,
                requested_preview=requested_preview,
                requested_final=requested_final,
                max_calibration=max_calibration,
                console=console,
            )
            actual_per_arm = int(window_plan["actual_preview"])
            effective_preview = actual_per_arm
            effective_final = actual_per_arm
            dataset_stride = getattr(
                cfg.dataset, "stride", getattr(cfg.dataset, "seq_len", 0)
            )
        else:
            notices.append(
                ProviderDatasetPlanNotice(
                    tag="WARN",
                    message=(
                        "Release profile requested but dataset provider does not expose "
                        "capacity estimation; using configured window counts."
                    ),
                    emoji="⚠️",
                )
            )

    signature_transform = None
    if use_mlm:

        def _signature_transform(
            preview_records_in: list[dict[str, Any]],
            final_records_in: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            temp_preview_records = [
                {
                    "input_ids": list(record["input_ids"]),
                    "attention_mask": list(record["attention_mask"]),
                    "dataset_index": record.get("dataset_index"),
                    "window_id": record.get("window_id"),
                }
                for record in preview_records_in
            ]
            temp_final_records = [
                {
                    "input_ids": list(record["input_ids"]),
                    "attention_mask": list(record["attention_mask"]),
                    "dataset_index": record.get("dataset_index"),
                    "window_id": record.get("window_id"),
                }
                for record in final_records_in
            ]
            apply_mlm_masks_fn(
                temp_preview_records,
                tokenizer=tokenizer,
                mask_prob=mask_prob,
                seed=mask_seed,
                random_token_prob=random_token_prob,
                original_token_prob=original_token_prob,
                prefix="preview",
            )
            apply_mlm_masks_fn(
                temp_final_records,
                tokenizer=tokenizer,
                mask_prob=mask_prob,
                seed=mask_seed,
                random_token_prob=random_token_prob,
                original_token_prob=original_token_prob,
                prefix="final",
            )
            return temp_preview_records + temp_final_records

        signature_transform = _signature_transform

    effective_windows = resolve_effective_windows_fn(
        data_provider=data_provider,
        tokenizer=tokenizer,
        seq_len=cfg.dataset.seq_len,
        stride=getattr(cfg.dataset, "stride", cfg.dataset.seq_len // 2),
        preview_n=effective_preview,
        final_n=effective_final,
        seed=getattr(cfg.dataset, "seed", 42),
        split=resolved_split,
        requested_preview=requested_preview,
        requested_final=requested_final,
        profile=profile_normalized,
        signature_transform=signature_transform,
        event_fn=lambda message: notices.append(
            ProviderDatasetPlanNotice(tag="WARN", message=message, emoji="⚠️")
        ),
    )

    preview_records = list(effective_windows["preview_records"])
    final_records = list(effective_windows["final_records"])
    preview_count = int(effective_windows["actual_preview"])
    final_count = int(effective_windows["actual_final"])
    effective_preview = preview_count
    effective_final = final_count

    try:
        provider_labels_prev = getattr(data_provider, "last_preview_labels", None)
        provider_labels_fin = getattr(data_provider, "last_final_labels", None)
    except (AttributeError, TypeError):
        provider_labels_prev = None
        provider_labels_fin = None

    for idx_local, record in enumerate(preview_records):
        if provider_labels_prev is not None and idx_local < len(provider_labels_prev):
            record["labels"] = tensor_or_list_to_ints_fn(
                provider_labels_prev[idx_local]
            )

    min_tokens_target = resolve_pm_min_tokens_target_fn(
        tier=tier or getattr(getattr(cfg, "auto", None), "tier", None),
        profile=profile,
    )
    tokens_floor_met = (
        int(effective_windows["preview_total_tokens"])
        + int(effective_windows["final_total_tokens"])
    ) >= int(min_tokens_target)

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
    window_plan["preview_total_tokens"] = int(effective_windows["preview_total_tokens"])
    window_plan["final_total_tokens"] = int(effective_windows["final_total_tokens"])
    window_plan["min_tokens_target"] = int(min_tokens_target)
    window_plan["tokens_floor_met"] = bool(tokens_floor_met)
    if effective_windows["dedupe_adjustments"]:
        window_plan["dedupe_adjustments"] = list(
            effective_windows["dedupe_adjustments"]
        )

    calibration_data: list[dict[str, Any]] = []
    preview_mask_total = 0
    final_mask_total = 0
    preview_mask_counts: list[int] = []
    final_mask_counts: list[int] = []
    if use_mlm:
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
    else:
        preview_mask_counts = [0] * len(preview_records)
        final_mask_counts = [0] * len(final_records)

    preview_sequences = [record["input_ids"] for record in preview_records]
    for idx, record in enumerate(preview_records):
        entry = {
            "input_ids": record["input_ids"],
            "attention_mask": record["attention_mask"],
            "window_id": f"preview::{idx}",
            "dataset_index": record.get("dataset_index"),
            "mlm_masked": record.get("mlm_masked", 0),
        }
        if use_mlm:
            entry["labels"] = record.get("labels", [-100] * len(record["input_ids"]))
        calibration_data.append(entry)

    final_sequences = [record["input_ids"] for record in final_records]
    for idx, record in enumerate(final_records):
        entry = {
            "input_ids": record["input_ids"],
            "attention_mask": record["attention_mask"],
            "window_id": f"final::{idx}",
            "dataset_index": record.get("dataset_index"),
            "mlm_masked": record.get("mlm_masked", 0),
        }
        if use_mlm:
            entry["labels"] = record.get("labels", [-100] * len(record["input_ids"]))
        elif provider_labels_fin is not None and idx < len(provider_labels_fin):
            entry["labels"] = tensor_or_list_to_ints_fn(provider_labels_fin[idx])
        calibration_data.append(entry)

    masked_tokens_total = preview_mask_total + final_mask_total
    preview_hash = hash_sequences_fn(preview_sequences)
    final_hash = hash_sequences_fn(final_sequences)
    dataset_meta = {
        "tokenizer_name": getattr(tokenizer, "name_or_path", "unknown"),
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
        "preview_total_tokens": int(effective_windows["preview_total_tokens"]),
        "final_total_tokens": int(effective_windows["final_total_tokens"]),
        "min_tokens_target": int(min_tokens_target),
        "tokens_floor_met": bool(tokens_floor_met),
        "loss_type": resolved_loss_type,
    }
    if use_mlm:
        dataset_meta["masked_tokens_preview"] = int(preview_mask_total)
        dataset_meta["masked_tokens_final"] = int(final_mask_total)
        dataset_meta["masked_tokens_total"] = int(masked_tokens_total)
    if window_plan:
        dataset_meta["window_plan"] = window_plan
        capacity_meta = window_plan.get("capacity")
        if capacity_meta:
            dataset_meta["window_capacity"] = capacity_meta
    strat_stats = getattr(data_provider, "stratification_stats", None)
    if strat_stats:
        dataset_meta["stratification"] = strat_stats
    scorer_profile = getattr(data_provider, "scorer_profile", None)
    if scorer_profile:
        dataset_meta["scorer_profile"] = scorer_profile

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
        notices=tuple(notices),
    )
