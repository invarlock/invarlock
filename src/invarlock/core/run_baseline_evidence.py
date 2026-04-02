from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ExtractPairingScheduleFn = Callable[[dict[str, Any] | None], dict[str, Any] | None]
ApplyMlmMasksFn = Callable[..., tuple[int, list[int]]]
ResolvePmMinTokensTargetFn = Callable[..., int]
HashSequencesFn = Callable[[Any], str]
TensorOrListToIntsFn = Callable[[Any], list[int]]


@dataclass(frozen=True)
class BaselineEvidenceLoadResult:
    report_data: dict[str, Any] | None
    pairing_schedule: dict[str, Any] | None
    tokenizer_hash: str | None
    status: str
    message: str | None = None


@dataclass(frozen=True)
class BaselinePairingMaterializationResult:
    calibration_data: list[dict[str, Any]]
    dataset_meta: dict[str, Any]
    window_plan: dict[str, Any] | None
    preview_count: int
    final_count: int
    effective_preview: int
    effective_final: int
    preview_mask_counts: list[int]
    final_mask_counts: list[int]
    preview_mask_total: int
    final_mask_total: int


def _merge_pairing_schedule(
    report_data: dict[str, Any], pairing_schedule: dict[str, Any]
) -> None:
    evaluation_windows = report_data.get("evaluation_windows")
    if not isinstance(evaluation_windows, dict):
        evaluation_windows = {}
        report_data["evaluation_windows"] = evaluation_windows

    for arm in ("preview", "final"):
        source = pairing_schedule.get(arm)
        if not isinstance(source, dict):
            continue
        target = evaluation_windows.get(arm)
        if not isinstance(target, dict):
            evaluation_windows[arm] = dict(source)
            continue
        for key, value in source.items():
            target[key] = value


def _harvest_tokenizer_hash(
    report_data: dict[str, Any], tokenizer_hash: str | None
) -> str | None:
    if tokenizer_hash:
        return tokenizer_hash
    meta_obj = report_data.get("meta")
    meta = meta_obj if isinstance(meta_obj, dict) else {}
    data_obj = report_data.get("data")
    data = data_obj if isinstance(data_obj, dict) else {}
    candidate = meta.get("tokenizer_hash") or data.get("tokenizer_hash")
    if isinstance(candidate, str) and candidate:
        return candidate
    return tokenizer_hash


def load_baseline_pairing_evidence(
    *,
    baseline_path: Path,
    tokenizer_hash: str | None,
    extract_pairing_schedule_fn: ExtractPairingScheduleFn,
) -> BaselineEvidenceLoadResult:
    path_str = str(baseline_path)
    missing_path_message = (
        f"PAIRING-EVIDENCE-MISSING: baseline report path does not exist ({path_str})"
    )
    if not baseline_path.exists():
        return BaselineEvidenceLoadResult(
            report_data=None,
            pairing_schedule=None,
            tokenizer_hash=tokenizer_hash,
            status="missing_path",
            message=missing_path_message,
        )

    try:
        loaded = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        return BaselineEvidenceLoadResult(
            report_data=None,
            pairing_schedule=None,
            tokenizer_hash=tokenizer_hash,
            status="parse_failed",
            message=f"PAIRING-EVIDENCE-MISSING: baseline report JSON parse failed ({exc})",
        )

    if not isinstance(loaded, dict):
        return BaselineEvidenceLoadResult(
            report_data=None,
            pairing_schedule=None,
            tokenizer_hash=tokenizer_hash,
            status="invalid_report",
            message=(
                "PAIRING-EVIDENCE-MISSING: baseline report missing or invalid "
                f"evaluation_windows ({path_str})"
            ),
        )

    pairing_schedule = extract_pairing_schedule_fn(loaded)
    if not pairing_schedule:
        return BaselineEvidenceLoadResult(
            report_data=None,
            pairing_schedule=None,
            tokenizer_hash=tokenizer_hash,
            status="missing_schedule",
            message=(
                "PAIRING-EVIDENCE-MISSING: baseline report missing or invalid "
                f"evaluation_windows ({path_str})"
            ),
        )

    _merge_pairing_schedule(loaded, pairing_schedule)
    resolved_tokenizer_hash = _harvest_tokenizer_hash(loaded, tokenizer_hash)
    return BaselineEvidenceLoadResult(
        report_data=loaded,
        pairing_schedule=pairing_schedule,
        tokenizer_hash=resolved_tokenizer_hash,
        status="loaded",
        message=None,
    )


def _needs_masks(entries: list[dict[str, Any]]) -> tuple[bool, list[int]]:
    missing_any = False
    counts: list[int] = []
    for entry in entries:
        labels_val = entry.get("labels")
        has_label_masks = bool(
            isinstance(labels_val, list) and any(token != -100 for token in labels_val)
        )
        existing_count = int(entry.get("mlm_masked", 0))
        if not has_label_masks and existing_count <= 0:
            missing_any = True
        counts.append(int(entry.get("mlm_masked", 0)))
    return missing_any, counts


def materialize_baseline_pairing_schedule(
    *,
    pairing_schedule: dict[str, Any],
    calibration_data: list[dict[str, Any]] | None,
    dataset_meta: dict[str, Any],
    window_plan: dict[str, Any] | None,
    tokenizer: Any,
    use_mlm: bool,
    mask_prob: float,
    mask_seed: int,
    random_token_prob: float,
    original_token_prob: float,
    resolved_tier: str | None,
    profile: str | None,
    apply_mlm_masks_fn: ApplyMlmMasksFn,
    resolve_pm_min_tokens_target_fn: ResolvePmMinTokensTargetFn,
    hash_sequences_fn: HashSequencesFn,
    tensor_or_list_to_ints_fn: TensorOrListToIntsFn,
) -> BaselinePairingMaterializationResult:
    materialized = list(calibration_data or [])

    preview_window_ids = pairing_schedule["preview"].get("window_ids")
    preview_labels = pairing_schedule["preview"].get("labels")
    preview_masked_token_counts = pairing_schedule["preview"].get("masked_token_counts")
    for idx, (input_ids, attention_mask) in enumerate(
        zip(
            pairing_schedule["preview"]["input_ids"],
            pairing_schedule["preview"]["attention_masks"],
            strict=False,
        )
    ):
        window_id = (
            preview_window_ids[idx]
            if preview_window_ids and idx < len(preview_window_ids)
            else idx
        )
        entry = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "window_id": f"preview::{window_id}",
        }
        if use_mlm:
            preview_labels_list: list[int] = []
            if isinstance(preview_labels, list) and idx < len(preview_labels):
                preview_labels_list = tensor_or_list_to_ints_fn(preview_labels[idx])
            if preview_labels_list and any(
                token != -100 for token in preview_labels_list
            ):
                entry["labels"] = preview_labels_list
                entry["mlm_masked"] = sum(
                    1 for token in preview_labels_list if token != -100
                )
            else:
                entry["labels"] = []
                entry["mlm_masked"] = 0
            if isinstance(preview_masked_token_counts, list) and idx < len(
                preview_masked_token_counts
            ):
                try:
                    entry["mlm_masked"] = int(preview_masked_token_counts[idx])
                except (TypeError, ValueError, OverflowError):
                    pass
        materialized.append(entry)

    final_window_ids = pairing_schedule["final"].get("window_ids")
    final_labels = pairing_schedule["final"].get("labels")
    final_masked_token_counts = pairing_schedule["final"].get("masked_token_counts")
    for idx, (input_ids, attention_mask) in enumerate(
        zip(
            pairing_schedule["final"]["input_ids"],
            pairing_schedule["final"]["attention_masks"],
            strict=False,
        )
    ):
        window_id = (
            final_window_ids[idx]
            if final_window_ids and idx < len(final_window_ids)
            else idx
        )
        entry = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "window_id": f"final::{window_id}",
        }
        if use_mlm:
            final_labels_list: list[int] = []
            if isinstance(final_labels, list) and idx < len(final_labels):
                final_labels_list = tensor_or_list_to_ints_fn(final_labels[idx])
            if final_labels_list and any(token != -100 for token in final_labels_list):
                entry["labels"] = final_labels_list
                entry["mlm_masked"] = sum(
                    1 for token in final_labels_list if token != -100
                )
            else:
                entry["labels"] = []
                entry["mlm_masked"] = 0
            if isinstance(final_masked_token_counts, list) and idx < len(
                final_masked_token_counts
            ):
                try:
                    entry["mlm_masked"] = int(final_masked_token_counts[idx])
                except (TypeError, ValueError, OverflowError):
                    pass
        materialized.append(entry)

    preview_count = len(pairing_schedule["preview"]["input_ids"])
    final_count = len(pairing_schedule["final"]["input_ids"])
    effective_preview = int(preview_count)
    effective_final = int(final_count)
    preview_mask_total = 0
    final_mask_total = 0
    preview_mask_counts: list[int] = []
    final_mask_counts: list[int] = []
    if use_mlm:
        preview_entries = materialized[:preview_count]
        final_entries = materialized[preview_count:]
        preview_missing, preview_counts_existing = _needs_masks(preview_entries)
        final_missing, final_counts_existing = _needs_masks(final_entries)

        if preview_missing:
            preview_mask_total, preview_mask_counts = apply_mlm_masks_fn(
                preview_entries,
                tokenizer=tokenizer,
                mask_prob=mask_prob,
                seed=mask_seed,
                random_token_prob=random_token_prob,
                original_token_prob=original_token_prob,
                prefix="preview",
            )
        else:
            preview_mask_counts = preview_counts_existing
            preview_mask_total = sum(preview_mask_counts)

        if final_missing:
            final_mask_total, final_mask_counts = apply_mlm_masks_fn(
                final_entries,
                tokenizer=tokenizer,
                mask_prob=mask_prob,
                seed=mask_seed,
                random_token_prob=random_token_prob,
                original_token_prob=original_token_prob,
                prefix="final",
            )
        else:
            final_mask_counts = final_counts_existing
            final_mask_total = sum(final_mask_counts)

        if preview_mask_counts:
            for entry, count in zip(preview_entries, preview_mask_counts, strict=False):
                entry["mlm_masked"] = int(count)
        if final_mask_counts:
            for entry, count in zip(final_entries, final_mask_counts, strict=False):
                entry["mlm_masked"] = int(count)

        if preview_count > 0 and preview_mask_total <= 0:
            raise ValueError(
                "Baseline pairing schedule provided no masked tokens for preview windows; "
                "ensure MLM labels are present in the baseline report."
            )
        if final_count > 0 and final_mask_total <= 0:
            raise ValueError(
                "Baseline pairing schedule provided no masked tokens for final windows; "
                "ensure MLM labels are present in the baseline report."
            )

        dataset_meta["masked_tokens_preview"] = int(preview_mask_total)
        dataset_meta["masked_tokens_final"] = int(final_mask_total)
        dataset_meta["masked_tokens_total"] = int(preview_mask_total + final_mask_total)

    if "preview_total_tokens" not in dataset_meta:
        dataset_meta["preview_total_tokens"] = sum(
            len(tensor_or_list_to_ints_fn(seq))
            for seq in pairing_schedule["preview"]["input_ids"]
        )
    if "final_total_tokens" not in dataset_meta:
        dataset_meta["final_total_tokens"] = sum(
            len(tensor_or_list_to_ints_fn(seq))
            for seq in pairing_schedule["final"]["input_ids"]
        )
    if "preview_hash" not in dataset_meta:
        preview_hash = hash_sequences_fn(
            tensor_or_list_to_ints_fn(seq)
            for seq in pairing_schedule["preview"]["input_ids"]
        )
        dataset_meta["preview_hash"] = preview_hash
    else:
        preview_hash = dataset_meta["preview_hash"]
    if "final_hash" not in dataset_meta:
        final_hash = hash_sequences_fn(
            tensor_or_list_to_ints_fn(seq)
            for seq in pairing_schedule["final"]["input_ids"]
        )
        dataset_meta["final_hash"] = final_hash
    else:
        final_hash = dataset_meta["final_hash"]
    if "dataset_hash" not in dataset_meta:
        dataset_meta["dataset_hash"] = hashlib.blake2s(
            (str(preview_hash) + str(final_hash)).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    if not window_plan:
        window_plan = {
            "profile": (profile or "").lower() or "baseline",
            "requested_preview": int(preview_count),
            "requested_final": int(final_count),
            "actual_preview": int(preview_count),
            "actual_final": int(final_count),
            "coverage_ok": True,
            "capacity": {},
        }

    if isinstance(window_plan, dict):
        preview_masks = pairing_schedule["preview"].get("attention_masks") or []
        final_masks = pairing_schedule["final"].get("attention_masks") or []
        preview_total_tokens = sum(
            sum(tensor_or_list_to_ints_fn(mask)) for mask in preview_masks
        ) or int(dataset_meta.get("preview_total_tokens", 0) or 0)
        final_total_tokens = sum(
            sum(tensor_or_list_to_ints_fn(mask)) for mask in final_masks
        ) or int(dataset_meta.get("final_total_tokens", 0) or 0)
        min_tokens_target = resolve_pm_min_tokens_target_fn(
            tier=resolved_tier,
            profile=profile,
        )
        window_plan["preview_total_tokens"] = int(preview_total_tokens)
        window_plan["final_total_tokens"] = int(final_total_tokens)
        window_plan["min_tokens_target"] = int(min_tokens_target)
        window_plan["tokens_floor_met"] = (
            int(preview_total_tokens) + int(final_total_tokens)
        ) >= int(min_tokens_target)
        dataset_meta["min_tokens_target"] = int(min_tokens_target)
        dataset_meta["tokens_floor_met"] = bool(window_plan["tokens_floor_met"])
        dataset_meta.setdefault("window_plan", window_plan)
        capacity_meta = window_plan.get("capacity")
        if capacity_meta and "window_capacity" not in dataset_meta:
            dataset_meta["window_capacity"] = capacity_meta

    return BaselinePairingMaterializationResult(
        calibration_data=materialized,
        dataset_meta=dataset_meta,
        window_plan=window_plan,
        preview_count=preview_count,
        final_count=final_count,
        effective_preview=effective_preview,
        effective_final=effective_final,
        preview_mask_counts=preview_mask_counts,
        final_mask_counts=final_mask_counts,
        preview_mask_total=preview_mask_total,
        final_mask_total=final_mask_total,
    )
