"""Helpers for resolving effective evaluation windows after dedupe retries."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypedDict

from .data import EvaluationWindow
from .data_support import DatasetDiagnostic

_WINDOW_COERCION_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)


class WindowRecord(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    dataset_index: int | None


type WindowSelectionStatus = Literal["selected", "no_candidate"]
type WindowCandidateOutcome = Literal[
    "selected",
    "insufficient_tokens",
    "resolution_failed",
]


class EffectiveWindowPlanResult(TypedDict):
    preview_window: Any
    final_window: Any
    preview_records: list[WindowRecord]
    final_records: list[WindowRecord]
    requested_preview: int
    requested_final: int
    actual_preview: int
    actual_final: int
    coverage_ok: bool
    preview_total_tokens: int
    final_total_tokens: int
    dedupe_adjustments: list[dict[str, int]]


class CandidateEvaluation(TypedDict):
    seq_len: int
    stride: int
    requested_preview: int
    requested_final: int
    actual_preview: int
    actual_final: int
    coverage_ok: bool
    preview_total_tokens: int
    final_total_tokens: int
    total_tokens: int
    min_tokens_target: int
    headroom_ratio: float
    effective_min_tokens: int
    tokens_floor_met: bool
    outcome: WindowCandidateOutcome
    reason: str
    reason_detail: str | None


class CandidateSelectionResult(TypedDict):
    status: str
    selection_status: WindowSelectionStatus
    min_tokens_target: int
    headroom_ratio: float
    effective_min_tokens: int
    selected: CandidateEvaluation | None
    candidates: list[CandidateEvaluation]


SignatureTransform = Callable[
    [list[WindowRecord], list[WindowRecord]], list[WindowRecord]
]
DiagnosticSink = Callable[[DatasetDiagnostic], None]


def _tensor_or_list_to_ints(value: Any) -> list[int]:
    candidate = value
    if hasattr(candidate, "detach"):
        try:
            candidate = candidate.detach()
        except _WINDOW_COERCION_ERRORS:
            pass
    if hasattr(candidate, "cpu"):
        try:
            candidate = candidate.cpu()
        except _WINDOW_COERCION_ERRORS:
            pass
    if hasattr(candidate, "tolist"):
        try:
            candidate = candidate.tolist()
        except _WINDOW_COERCION_ERRORS:
            pass
    if isinstance(candidate, tuple):
        candidate = list(candidate)
    if not isinstance(candidate, list):
        return []
    result: list[int] = []
    for item in candidate:
        if isinstance(item, bool):
            return []
        try:
            result.append(int(item))
        except _WINDOW_COERCION_ERRORS:
            return []
    return result


def _coerce_indices(indices: Any) -> list[Any]:
    if isinstance(indices, list):
        return indices
    try:
        return list(indices)
    except TypeError:
        return []


def _window_records(window: Any) -> tuple[list[WindowRecord], int]:
    records: list[WindowRecord] = []
    total_tokens = 0
    indices = _coerce_indices(getattr(window, "indices", []))
    for idx_local, (input_ids, attention_mask) in enumerate(
        zip(
            getattr(window, "input_ids", []),
            getattr(window, "attention_masks", []),
            strict=False,
        )
    ):
        input_ids_list = _tensor_or_list_to_ints(input_ids)
        attention_mask_list = (
            _tensor_or_list_to_ints(attention_mask)
            if attention_mask is not None
            else [1] * len(input_ids_list)
        )
        total_tokens += sum(attention_mask_list)
        dataset_index = indices[idx_local] if idx_local < len(indices) else idx_local
        try:
            dataset_index = int(dataset_index)
        except (TypeError, ValueError):
            dataset_index = None
        records.append(
            {
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
                "dataset_index": dataset_index,
            }
        )
    return records, total_tokens


def _masked_signature(record: WindowRecord) -> tuple[int, ...]:
    return tuple(
        tok
        for tok, mask in zip(
            record.get("input_ids", []),
            record.get("attention_mask", []),
            strict=False,
        )
        if mask
    )


def _non_release_min_per_arm_floor(
    requested_preview: int,
    requested_final: int,
    release_min_windows_per_arm: int,
) -> int:
    floor_source = min(
        int(requested_preview or 0) or release_min_windows_per_arm,
        int(requested_final or 0) or release_min_windows_per_arm,
    )
    return max(10, floor_source // 2)


def resolve_effective_windows(
    *,
    data_provider: Any,
    tokenizer: Any,
    seq_len: int,
    stride: int,
    preview_n: int,
    final_n: int,
    seed: int,
    split: str,
    requested_preview: int | None = None,
    requested_final: int | None = None,
    profile: str | None = None,
    release_min_windows_per_arm: int = 200,
    signature_transform: SignatureTransform | None = None,
    diagnostic_fn: DiagnosticSink | None = None,
) -> EffectiveWindowPlanResult:
    requested_preview_n = int(requested_preview or preview_n)
    requested_final_n = int(requested_final or final_n)
    effective_preview = int(preview_n)
    effective_final = int(final_n)
    profile_normalized = (profile or "").lower() or "default"
    dedupe_adjustments: list[dict[str, int]] = []

    while True:
        preview_window, final_window = data_provider.windows(
            tokenizer=tokenizer,
            seq_len=seq_len,
            stride=stride,
            preview_n=effective_preview,
            final_n=effective_final,
            seed=seed,
            split=split,
        )

        preview_count = len(getattr(preview_window, "input_ids", []))
        final_count = len(getattr(final_window, "input_ids", []))
        if isinstance(preview_window, EvaluationWindow) and isinstance(
            final_window, EvaluationWindow
        ):
            if preview_count != effective_preview or final_count != effective_final:
                raise RuntimeError(
                    "Dataset provider returned mismatched preview/final counts "
                    f"({preview_count}/{final_count}) "
                    f"expected ({effective_preview}/{effective_final}). "
                    "CI/Release profiles require exact parity."
                )
        else:
            preview_count = effective_preview
            final_count = effective_final

        preview_records, preview_total_tokens = _window_records(preview_window)
        final_records, final_total_tokens = _window_records(final_window)
        records_for_signatures = (
            signature_transform(preview_records, final_records)
            if signature_transform is not None
            else preview_records + final_records
        )

        unique_sequences = len(
            {_masked_signature(record) for record in records_for_signatures}
        )
        combined_total = len(records_for_signatures)
        if unique_sequences == combined_total:
            return {
                "preview_window": preview_window,
                "final_window": final_window,
                "preview_records": preview_records,
                "final_records": final_records,
                "requested_preview": requested_preview_n,
                "requested_final": requested_final_n,
                "actual_preview": int(preview_count),
                "actual_final": int(final_count),
                "coverage_ok": bool(preview_count == final_count),
                "preview_total_tokens": int(preview_total_tokens),
                "final_total_tokens": int(final_total_tokens),
                "dedupe_adjustments": list(dedupe_adjustments),
            }

        deficit = combined_total - unique_sequences
        reduction = max(5, int(deficit) if deficit > 0 else 1)
        proposed_per_arm = preview_count - reduction
        if proposed_per_arm >= preview_count:
            proposed_per_arm = preview_count - 1
        min_per_arm_floor = (
            release_min_windows_per_arm
            if profile_normalized == "release"
            else _non_release_min_per_arm_floor(
                requested_preview_n,
                requested_final_n,
                release_min_windows_per_arm,
            )
        )
        if proposed_per_arm < min_per_arm_floor:
            raise RuntimeError(
                "Unable to construct non-overlapping windows within minimum window floor."
            )
        if diagnostic_fn is not None:
            diagnostic_fn(
                DatasetDiagnostic(
                    kind="window.dedupe_adjustment",
                    severity="warning",
                    message="Duplicate windows detected; reducing per-arm windows.",
                    metadata={
                        "deficit": int(deficit),
                        "proposed_per_arm": int(proposed_per_arm),
                    },
                    category="window",
                    code="window.dedupe_adjustment",
                )
            )
        dedupe_adjustments.append(
            {
                "deficit": int(deficit),
                "proposed_per_arm": int(proposed_per_arm),
            }
        )
        effective_preview = proposed_per_arm
        effective_final = proposed_per_arm


def choose_first_token_sufficient_candidate(
    *,
    data_provider: Any,
    tokenizer: Any,
    split: str,
    seed: int,
    candidates: Sequence[dict[str, int]],
    min_tokens_target: int,
    headroom_ratio: float = 1.05,
    profile: str | None = None,
    release_min_windows_per_arm: int = 200,
    signature_transform: SignatureTransform | None = None,
    diagnostic_fn: DiagnosticSink | None = None,
) -> CandidateSelectionResult:
    effective_min_tokens = int(
        math.ceil(float(min_tokens_target) * float(headroom_ratio))
    )
    evaluations: list[CandidateEvaluation] = []

    for candidate in candidates:
        seq_len = int(candidate["seq_len"])
        stride = int(candidate["stride"])
        preview_n = int(candidate["preview_n"])
        final_n = int(candidate["final_n"])
        try:
            planned = resolve_effective_windows(
                data_provider=data_provider,
                tokenizer=tokenizer,
                seq_len=seq_len,
                stride=stride,
                preview_n=preview_n,
                final_n=final_n,
                seed=seed,
                split=split,
                requested_preview=preview_n,
                requested_final=final_n,
                profile=profile,
                release_min_windows_per_arm=release_min_windows_per_arm,
                signature_transform=signature_transform,
                diagnostic_fn=diagnostic_fn,
            )
            total_tokens = int(planned["preview_total_tokens"]) + int(
                planned["final_total_tokens"]
            )
            tokens_floor_met = total_tokens >= effective_min_tokens
            evaluation: CandidateEvaluation = {
                "seq_len": seq_len,
                "stride": stride,
                "requested_preview": preview_n,
                "requested_final": final_n,
                "actual_preview": int(planned["actual_preview"]),
                "actual_final": int(planned["actual_final"]),
                "coverage_ok": bool(planned["coverage_ok"]),
                "preview_total_tokens": int(planned["preview_total_tokens"]),
                "final_total_tokens": int(planned["final_total_tokens"]),
                "total_tokens": total_tokens,
                "min_tokens_target": int(min_tokens_target),
                "headroom_ratio": float(headroom_ratio),
                "effective_min_tokens": int(effective_min_tokens),
                "tokens_floor_met": bool(tokens_floor_met),
                "outcome": "selected" if tokens_floor_met else "insufficient_tokens",
                "reason": "selected" if tokens_floor_met else "below_token_floor",
                "reason_detail": None,
            }
            evaluations.append(evaluation)
            if tokens_floor_met and planned["coverage_ok"]:
                return {
                    "status": "selected",
                    "selection_status": "selected",
                    "min_tokens_target": int(min_tokens_target),
                    "headroom_ratio": float(headroom_ratio),
                    "effective_min_tokens": int(effective_min_tokens),
                    "selected": evaluation,
                    "candidates": evaluations,
                }
        except RuntimeError as exc:
            evaluations.append(
                {
                    "seq_len": seq_len,
                    "stride": stride,
                    "requested_preview": preview_n,
                    "requested_final": final_n,
                    "actual_preview": 0,
                    "actual_final": 0,
                    "coverage_ok": False,
                    "preview_total_tokens": 0,
                    "final_total_tokens": 0,
                    "total_tokens": 0,
                    "min_tokens_target": int(min_tokens_target),
                    "headroom_ratio": float(headroom_ratio),
                    "effective_min_tokens": int(effective_min_tokens),
                    "tokens_floor_met": False,
                    "outcome": "resolution_failed",
                    "reason": str(exc),
                    "reason_detail": str(exc),
                }
            )

    return {
        "status": "no_candidate",
        "selection_status": "no_candidate",
        "min_tokens_target": int(min_tokens_target),
        "headroom_ratio": float(headroom_ratio),
        "effective_min_tokens": int(effective_min_tokens),
        "selected": None,
        "candidates": evaluations,
    }


__all__ = [
    "CandidateEvaluation",
    "CandidateSelectionResult",
    "EffectiveWindowPlanResult",
    "resolve_effective_windows",
    "choose_first_token_sufficient_candidate",
]
