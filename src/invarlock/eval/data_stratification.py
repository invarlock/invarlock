from __future__ import annotations

from typing import Any

import numpy as np

from invarlock.core.exceptions import DataError as _DataErr
from invarlock.core.exceptions import ValidationError as _ValErr

from .data_support import EvaluationWindow


def stratify_wikitext_candidates(
    candidates: list[dict[str, Any]],
    *,
    preview_n: int,
    final_n: int,
    reserve: int,
    batch_size_used: int,
) -> tuple[EvaluationWindow, EvaluationWindow, dict[str, Any]]:
    total_required = int(preview_n) + int(final_n)
    if total_required <= 0:
        raise _ValErr(
            code="E302",
            message="VALIDATION-FAILED: preview/final must be positive",
        )
    if len(candidates) < total_required:
        raise _DataErr(
            code="E305",
            message="STRATIFY-FAILED: candidate pool insufficient",
        )

    def _mean_difficulty(items: list[dict[str, Any]]) -> float:
        if not items:
            return 0.0
        return float(sum(item["difficulty"] for item in items) / len(items))

    sorted_candidates = sorted(
        candidates, key=lambda item: (item["difficulty"], item["dataset_index"])
    )
    total_candidates = len(sorted_candidates)
    selection_count = total_required
    selected_positions: list[int] = []
    used_positions: set[int] = set()

    for k in range(selection_count):
        target_position = (k + 0.5) * total_candidates / selection_count
        base_idx = int(round(target_position))
        offset = 0
        chosen: int | None = None
        while offset < total_candidates:
            for candidate_idx in (base_idx + offset, base_idx - offset):
                if (
                    0 <= candidate_idx < total_candidates
                    and candidate_idx not in used_positions
                ):
                    chosen = candidate_idx
                    break
            if chosen is not None:
                break
            offset += 1
        if chosen is not None:
            used_positions.add(chosen)
            selected_positions.append(chosen)

    if len(selected_positions) < selection_count:
        for candidate_idx in range(total_candidates):
            if candidate_idx not in used_positions:
                used_positions.add(candidate_idx)
                selected_positions.append(candidate_idx)
            if len(selected_positions) == selection_count:
                break
    if len(selected_positions) < selection_count:
        raise _DataErr(
            code="E305", message="STRATIFY-FAILED: candidate pool insufficient"
        )

    selected_candidates = [sorted_candidates[idx] for idx in selected_positions]
    selected_candidates.sort(
        key=lambda item: (item["difficulty"], item["dataset_index"])
    )
    preview_candidates: list[dict[str, Any]] = []
    final_candidates: list[dict[str, Any]] = []

    def assign_candidate(
        candidate: dict[str, Any],
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
        primary_capacity: int,
        secondary_capacity: int,
    ) -> None:
        if len(primary) < primary_capacity:
            primary.append(candidate)
        elif len(secondary) < secondary_capacity:
            secondary.append(candidate)

    for pair_start in range(0, len(selected_candidates), 2):
        pair = selected_candidates[pair_start : pair_start + 2]
        if not pair:
            continue
        if len(pair) == 2:
            easy, hard = pair
            pair_index = pair_start // 2
            if pair_index % 2 == 0:
                assign_candidate(
                    easy, preview_candidates, final_candidates, preview_n, final_n
                )
                assign_candidate(
                    hard, final_candidates, preview_candidates, final_n, preview_n
                )
            else:
                assign_candidate(
                    easy, final_candidates, preview_candidates, final_n, preview_n
                )
                assign_candidate(
                    hard, preview_candidates, final_candidates, preview_n, final_n
                )
        else:
            assign_candidate(
                pair[0],
                preview_candidates,
                final_candidates,
                preview_n,
                final_n,
            )

    assigned_ids = {
        id(candidate) for candidate in preview_candidates + final_candidates
    }
    remaining = [
        candidate
        for candidate in selected_candidates
        if id(candidate) not in assigned_ids
    ]
    for candidate in remaining:
        if len(preview_candidates) < preview_n:
            preview_candidates.append(candidate)
        elif len(final_candidates) < final_n:
            final_candidates.append(candidate)

    for _ in range(100):
        if not preview_candidates or not final_candidates:
            break
        diff = _mean_difficulty(preview_candidates) - _mean_difficulty(final_candidates)
        if abs(diff) <= 1e-4:
            break
        if diff < 0:
            preview_candidate = min(preview_candidates, key=lambda c: c["difficulty"])
            final_candidate = max(final_candidates, key=lambda c: c["difficulty"])
        else:
            preview_candidate = max(preview_candidates, key=lambda c: c["difficulty"])
            final_candidate = min(final_candidates, key=lambda c: c["difficulty"])
        if preview_candidate is final_candidate:
            break
        preview_candidates.remove(preview_candidate)
        final_candidates.remove(final_candidate)
        preview_candidates.append(final_candidate)
        final_candidates.append(preview_candidate)
        new_diff = _mean_difficulty(preview_candidates) - _mean_difficulty(
            final_candidates
        )
        if abs(new_diff) >= abs(diff) - 1e-6:
            preview_candidates.remove(final_candidate)
            final_candidates.remove(preview_candidate)
            preview_candidates.append(preview_candidate)
            final_candidates.append(final_candidate)
            break

    if len(preview_candidates) != preview_n or len(final_candidates) != final_n:
        raise _DataErr(
            code="E305",
            message=(
                "STRATIFY-FAILED: failed to allocate preview/final windows with equal counts"
            ),
            details={
                "preview_target": int(preview_n),
                "final_target": int(final_n),
                "preview_got": int(len(preview_candidates)),
                "final_got": int(len(final_candidates)),
            },
        )

    preview_candidates.sort(
        key=lambda item: (item["difficulty"], item["dataset_index"])
    )
    final_candidates.sort(key=lambda item: (item["difficulty"], item["dataset_index"]))
    preview_window = EvaluationWindow(
        input_ids=[c["input_ids"] for c in preview_candidates],
        attention_masks=[c["attention_mask"] for c in preview_candidates],
        indices=[c["dataset_index"] for c in preview_candidates],
    )
    final_window = EvaluationWindow(
        input_ids=[c["input_ids"] for c in final_candidates],
        attention_masks=[c["attention_mask"] for c in final_candidates],
        indices=[c["dataset_index"] for c in final_candidates],
    )
    if len(preview_window) != preview_n or len(final_window) != final_n:
        raise _DataErr(
            code="E305",
            message="STRATIFY-FAILED: window stratification mismatch",
            details={
                "preview_target": int(preview_n),
                "final_target": int(final_n),
                "preview_got": int(len(preview_window)),
                "final_got": int(len(final_window)),
            },
        )

    preview_difficulties = [c["difficulty"] for c in preview_candidates]
    final_difficulties = [c["difficulty"] for c in final_candidates]
    stats = {
        "pool_size": len(selected_candidates),
        "reserve": reserve,
        "batch_size_used": int(batch_size_used),
        "preview_mean_difficulty": float(np.mean(preview_difficulties))
        if preview_difficulties
        else 0.0,
        "final_mean_difficulty": float(np.mean(final_difficulties))
        if final_difficulties
        else 0.0,
        "preview_std_difficulty": float(np.std(preview_difficulties))
        if preview_difficulties
        else 0.0,
        "final_std_difficulty": float(np.std(final_difficulties))
        if final_difficulties
        else 0.0,
        "difficulty_gap": float(
            (np.mean(final_difficulties) - np.mean(preview_difficulties))
            if (preview_difficulties and final_difficulties)
            else 0.0
        ),
    }
    return preview_window, final_window, stats


__all__ = ["stratify_wikitext_candidates"]
