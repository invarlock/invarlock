from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from invarlock.eval.data_support import DatasetDiagnostic

from .metric_provider_resolution import resolve_provider_kind_and_kwargs


@dataclass(frozen=True)
class ProviderDatasetPlanDiagnostic:
    code: str
    summary: str
    level: str = "info"
    context: dict[str, Any] = field(default_factory=dict)


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
    diagnostics: tuple[ProviderDatasetPlanDiagnostic, ...]


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


def _section_value(section: Any, key: str) -> Any:
    get_value = getattr(section, "get", None)
    if callable(get_value):
        try:
            return get_value(key)
        except (AttributeError, KeyError, TypeError):
            pass
    try:
        return getattr(section, key)
    except (AttributeError, KeyError, TypeError):
        return None


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
        value = _section_value(cfg_dataset, key)
        if value is not None and value != "":
            provider_kwargs[key] = value

    _provider_kind, explicit_provider_kwargs = resolve_provider_kind_and_kwargs(
        getattr(cfg_dataset, "provider", None)
    )
    provider_kwargs.update(explicit_provider_kwargs)
    return provider_kwargs


def _section_dict(cfg: Any, name: str) -> dict[str, Any]:
    section_fn = getattr(cfg, "section", None)
    if callable(section_fn):
        try:
            section = section_fn(name)
        except (AttributeError, KeyError, TypeError):
            section = None
        if isinstance(section, dict):
            return section
    try:
        value = getattr(cfg, name)
    except (AttributeError, KeyError, TypeError):
        value = None
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            key: item for key, item in vars(value).items() if not key.startswith("_")
        }
    return {}


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _hash_texts(values: Sequence[str]) -> str:
    return hashlib.blake2s(
        "||".join(str(value) for value in values).encode("utf-8"),
        digest_size=16,
    ).hexdigest()


def _split_vision_text_counts(
    *,
    available: int,
    requested_preview: int,
    requested_final: int,
    effective_preview: int,
    effective_final: int,
) -> tuple[int, int]:
    if available <= 0:
        return 0, 0
    desired_preview = max(int(effective_preview), 0)
    desired_final = max(int(effective_final), 0)
    if desired_preview == 0 and desired_final == 0:
        return 0, 0
    if desired_preview + desired_final <= available:
        return desired_preview, desired_final
    if desired_preview <= 0:
        return 0, min(desired_final, available)
    if desired_final <= 0:
        return min(desired_preview, available), 0
    if available == 1:
        return (0, 1) if requested_final > 0 else (1, 0)

    desired_total = desired_preview + desired_final
    preview_count = int(round(available * (desired_preview / desired_total)))
    preview_count = max(1, min(preview_count, available - 1, desired_preview))
    final_count = min(desired_final, available - preview_count)
    return preview_count, final_count


def _vision_text_dataset_plan(
    *,
    data_provider: Any,
    resolved_split: str,
    used_fallback_split: bool,
    cfg_dataset: Any,
    requested_preview: int,
    requested_final: int,
    effective_preview: int,
    effective_final: int,
    resolved_loss_type: str,
    diagnostics: list[ProviderDatasetPlanDiagnostic],
) -> ProviderDatasetPlanResult:
    examples_fn = getattr(data_provider, "examples", None)
    if not callable(examples_fn):
        raise TypeError("vision_text provider must expose examples()")

    raw_examples = list(examples_fn(split=resolved_split))
    preview_count, final_count = _split_vision_text_counts(
        available=len(raw_examples),
        requested_preview=requested_preview,
        requested_final=requested_final,
        effective_preview=effective_preview,
        effective_final=effective_final,
    )

    preview_records = [dict(item) for item in raw_examples[:preview_count]]
    final_records = [
        dict(item) for item in raw_examples[preview_count : preview_count + final_count]
    ]

    seq_len = int(getattr(cfg_dataset, "seq_len", 0) or 0)
    calibration_data: list[dict[str, Any]] = []
    for arm, records in (("preview", preview_records), ("final", final_records)):
        for index, record in enumerate(records):
            record["seq_len"] = seq_len
            record["example_id"] = str(
                record.get("id") or record.get("example_id") or ""
            )
            entry = dict(record)
            entry["window_id"] = f"{arm}::{index}"
            calibration_data.append(entry)

    preview_ids = [str(record["example_id"]) for record in preview_records]
    final_ids = [str(record["example_id"]) for record in final_records]
    preview_hash = _hash_texts(preview_ids)
    final_hash = _hash_texts(final_ids)
    provider_digest = (
        data_provider.digest()
        if callable(getattr(data_provider, "digest", None))
        else {}
    )
    window_plan = {
        "profile": "vision_text",
        "requested_preview": int(requested_preview),
        "requested_final": int(requested_final),
        "capacity": {"available_examples": int(len(raw_examples))},
        "actual_preview": int(len(preview_records)),
        "actual_final": int(len(final_records)),
        "coverage_ok": bool(final_records or requested_final == 0),
        "preview_total_tokens": 0,
        "final_total_tokens": 0,
        "min_tokens_target": 0,
        "tokens_floor_met": True,
    }
    dataset_meta = {
        "provider_kind": "vision_text",
        "provider_digest": dict(provider_digest)
        if isinstance(provider_digest, Mapping)
        else {},
        "dataset_hash": hashlib.blake2s(
            (preview_hash + final_hash).encode("utf-8"), digest_size=16
        ).hexdigest(),
        "preview_hash": preview_hash,
        "final_hash": final_hash,
        "preview_example_ids": preview_ids,
        "final_example_ids": final_ids,
        "preview_total_tokens": 0,
        "final_total_tokens": 0,
        "min_tokens_target": 0,
        "tokens_floor_met": True,
        "loss_type": resolved_loss_type,
        "window_plan": window_plan,
    }

    return ProviderDatasetPlanResult(
        data_provider=data_provider,
        resolved_split=resolved_split,
        used_fallback_split=used_fallback_split,
        tokenizer=None,
        tokenizer_hash=None,
        calibration_data=calibration_data,
        dataset_meta=dataset_meta,
        window_plan=window_plan,
        preview_count=len(preview_records),
        final_count=len(final_records),
        effective_preview=len(preview_records),
        effective_final=len(final_records),
        preview_mask_counts=[0] * len(preview_records),
        final_mask_counts=[0] * len(final_records),
        preview_records=preview_records,
        final_records=final_records,
        diagnostics=tuple(diagnostics),
    )


def _resolve_release_window_plan(
    *,
    data_provider: Any,
    eval_section: dict[str, Any],
    guards_section: dict[str, Any],
    cfg_dataset: Any,
    resolved_split: str,
    tokenizer: Any,
    requested_preview: int,
    requested_final: int,
    profile: str | None,
    pairing_schedule_present: bool,
    maybe_plan_release_windows_fn: MaybePlanReleaseWindowsFn,
    diagnostics: list[ProviderDatasetPlanDiagnostic],
) -> tuple[dict[str, Any] | None, int, int]:
    effective_preview = int(requested_preview)
    effective_final = int(requested_final)
    window_plan: dict[str, Any] | None = None
    release_profile = (profile or "").lower() == "release"
    if not release_profile or pairing_schedule_present:
        return window_plan, effective_preview, effective_final
    estimate_fn = getattr(data_provider, "estimate_capacity", None)
    if not callable(estimate_fn):
        diagnostics.append(
            ProviderDatasetPlanDiagnostic(
                code="provider.capacity_missing",
                summary=(
                    "Release profile requested but dataset provider does not expose "
                    "capacity estimation; using configured window counts."
                ),
                level="warning",
            )
        )
        return window_plan, effective_preview, effective_final
    capacity_fast = bool(eval_section.get("capacity_fast", False))
    dataset_stride = getattr(
        cfg_dataset, "stride", getattr(cfg_dataset, "seq_len", 0) // 2
    )
    capacity_meta = estimate_fn(
        tokenizer=tokenizer,
        seq_len=cfg_dataset.seq_len,
        stride=dataset_stride,
        split=resolved_split,
        target_total=requested_preview + requested_final,
        fast_mode=capacity_fast,
    )
    variance_policy = (
        guards_section.get("variance")
        if isinstance(guards_section.get("variance"), dict)
        else None
    )
    max_calibration = (
        int(variance_policy.get("max_calib", 0)) if variance_policy is not None else 0
    )
    window_plan = maybe_plan_release_windows_fn(
        capacity_meta,
        requested_preview=requested_preview,
        requested_final=requested_final,
        max_calibration=max_calibration,
    )
    actual_per_arm = int(window_plan["actual_preview"])
    return window_plan, actual_per_arm, actual_per_arm


def _build_signature_transform(
    *,
    use_mlm: bool,
    tokenizer: Any,
    mask_prob: float,
    mask_seed: int,
    random_token_prob: float,
    original_token_prob: float,
    apply_mlm_masks_fn: ApplyMlmMasksFn,
) -> (
    Callable[[list[dict[str, Any]], list[dict[str, Any]]], list[dict[str, Any]]] | None
):
    if not use_mlm:
        return None

    def _signature_transform(
        preview_records_in: list[dict[str, Any]],
        final_records_in: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        temp_preview_records: list[dict[str, Any]] = [
            {
                "input_ids": list(record["input_ids"]),
                "attention_mask": list(record["attention_mask"]),
                "dataset_index": record.get("dataset_index"),
                "window_id": record.get("window_id"),
            }
            for record in preview_records_in
        ]
        temp_final_records: list[dict[str, Any]] = [
            {
                "input_ids": list(record["input_ids"]),
                "attention_mask": list(record["attention_mask"]),
                "dataset_index": record.get("dataset_index"),
                "window_id": record.get("window_id"),
            }
            for record in final_records_in
        ]
        preview_raw_inputs = [
            list(record["input_ids"]) for record in temp_preview_records
        ]
        final_raw_inputs = [list(record["input_ids"]) for record in temp_final_records]
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
        for record, raw_input_ids in zip(
            temp_preview_records, preview_raw_inputs, strict=False
        ):
            record["input_ids"] = raw_input_ids
        for record, raw_input_ids in zip(
            temp_final_records, final_raw_inputs, strict=False
        ):
            record["input_ids"] = raw_input_ids
        return temp_preview_records + temp_final_records

    return _signature_transform


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
    from .run_provider_dataset_materialization import (
        _materialize_text_provider_dataset_plan as _materialize,
    )

    return _materialize(
        data_provider=data_provider,
        resolved_split=resolved_split,
        used_fallback_split=used_fallback_split,
        tokenizer=tokenizer,
        tokenizer_hash=tokenizer_hash,
        effective_windows=effective_windows,
        requested_preview=requested_preview,
        requested_final=requested_final,
        effective_preview=effective_preview,
        effective_final=effective_final,
        resolved_loss_type=resolved_loss_type,
        window_plan=window_plan,
        use_mlm=use_mlm,
        mask_prob=mask_prob,
        mask_seed=mask_seed,
        random_token_prob=random_token_prob,
        original_token_prob=original_token_prob,
        tier=tier,
        profile=profile,
        apply_mlm_masks_fn=apply_mlm_masks_fn,
        resolve_pm_min_tokens_target_fn=resolve_pm_min_tokens_target_fn,
        hash_sequences_fn=hash_sequences_fn,
        tokenizer_digest_fn=tokenizer_digest_fn,
        safe_int_fn=safe_int_fn,
        tensor_or_list_to_ints_fn=tensor_or_list_to_ints_fn,
        diagnostics=diagnostics,
    )


def build_provider_dataset_plan(
    *,
    cfg: Any,
    model_profile: Any,
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
    diagnostics: list[ProviderDatasetPlanDiagnostic] = []
    eval_section = _section_dict(cfg, "eval")
    guards_section = _section_dict(cfg, "guards")
    auto_section = _section_dict(cfg, "auto")

    def _collect_diagnostic(diagnostic: DatasetDiagnostic) -> None:
        diagnostics.append(
            ProviderDatasetPlanDiagnostic(
                code=diagnostic.code or diagnostic.kind,
                summary=diagnostic.message,
                level=diagnostic.severity,
                context=dict(diagnostic.metadata),
            )
        )

    provider_kwargs = _build_provider_kwargs(cfg.dataset)
    data_provider, resolved_split, used_fallback_split = resolve_provider_and_split_fn(
        cfg,
        model_profile,
        get_provider_fn=get_provider_fn,
        provider_kwargs=provider_kwargs,
        resolved_device=resolved_device,
    )
    diagnostics.append(
        ProviderDatasetPlanDiagnostic(
            code="provider.resolved",
            summary="provider resolved",
            level="info",
            context={
                "provider": getattr(
                    data_provider, "name", type(data_provider).__name__
                ),
                "split": resolved_split,
                "used_fallback_split": bool(used_fallback_split),
            },
        )
    )

    provider_name = str(getattr(data_provider, "name", "") or "")
    if provider_name == "vision_text":
        return _vision_text_dataset_plan(
            data_provider=data_provider,
            resolved_split=resolved_split,
            used_fallback_split=used_fallback_split,
            cfg_dataset=cfg.dataset,
            requested_preview=int(requested_preview),
            requested_final=int(requested_final),
            effective_preview=int(effective_preview),
            effective_final=int(effective_final),
            resolved_loss_type=resolved_loss_type,
            diagnostics=diagnostics,
        )

    tokenizer, tokenizer_hash = resolve_tokenizer_fn(model_profile)
    window_plan, effective_preview, effective_final = _resolve_release_window_plan(
        data_provider=data_provider,
        eval_section=eval_section,
        guards_section=guards_section,
        cfg_dataset=cfg.dataset,
        resolved_split=resolved_split,
        tokenizer=tokenizer,
        requested_preview=requested_preview,
        requested_final=requested_final,
        profile=profile,
        pairing_schedule_present=pairing_schedule_present,
        maybe_plan_release_windows_fn=maybe_plan_release_windows_fn,
        diagnostics=diagnostics,
    )
    signature_transform = _build_signature_transform(
        use_mlm=use_mlm,
        tokenizer=tokenizer,
        mask_prob=mask_prob,
        mask_seed=mask_seed,
        random_token_prob=random_token_prob,
        original_token_prob=original_token_prob,
        apply_mlm_masks_fn=apply_mlm_masks_fn,
    )

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
        diagnostic_fn=_collect_diagnostic,
    )

    return _materialize_text_provider_dataset_plan(
        data_provider=data_provider,
        resolved_split=resolved_split,
        used_fallback_split=used_fallback_split,
        tokenizer=tokenizer,
        tokenizer_hash=tokenizer_hash,
        effective_windows=effective_windows,
        requested_preview=requested_preview,
        requested_final=requested_final,
        effective_preview=effective_preview,
        effective_final=effective_final,
        resolved_loss_type=resolved_loss_type,
        window_plan=window_plan,
        use_mlm=use_mlm,
        mask_prob=mask_prob,
        mask_seed=mask_seed,
        random_token_prob=random_token_prob,
        original_token_prob=original_token_prob,
        tier=tier or auto_section.get("tier"),
        profile=profile,
        apply_mlm_masks_fn=apply_mlm_masks_fn,
        resolve_pm_min_tokens_target_fn=resolve_pm_min_tokens_target_fn,
        hash_sequences_fn=hash_sequences_fn,
        tokenizer_digest_fn=tokenizer_digest_fn,
        safe_int_fn=safe_int_fn,
        tensor_or_list_to_ints_fn=tensor_or_list_to_ints_fn,
        diagnostics=diagnostics,
    )


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
            preview_records=list(
                getattr(materialized_baseline, "preview_records", []) or []
            ),
            final_records=list(
                getattr(materialized_baseline, "final_records", []) or []
            ),
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
