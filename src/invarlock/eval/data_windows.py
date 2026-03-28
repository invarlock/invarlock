"""
Evaluation window structures and deterministic hashing helpers.
"""

from __future__ import annotations

import hashlib
from typing import Any, NamedTuple


class EvaluationWindow(NamedTuple):
    """A window of tokenized samples for evaluation."""

    input_ids: list[list[int]]
    attention_masks: list[list[int]]
    indices: list[int]

    def __len__(self) -> int:
        return len(self.input_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_masks": self.attention_masks,
            "indices": self.indices,
            "length": len(self.input_ids),
        }


def split_window_by_index(
    window: EvaluationWindow, *, split_index: int
) -> tuple[EvaluationWindow, EvaluationWindow]:
    preview_input_ids: list[list[int]] = []
    preview_attention_masks: list[list[int]] = []
    preview_indices: list[int] = []
    final_input_ids: list[list[int]] = []
    final_attention_masks: list[list[int]] = []
    final_indices: list[int] = []

    for input_ids, attention_mask, index in zip(
        window.input_ids,
        window.attention_masks,
        window.indices,
        strict=False,
    ):
        if index < split_index:
            preview_input_ids.append(input_ids)
            preview_attention_masks.append(attention_mask)
            preview_indices.append(index)
        else:
            final_input_ids.append(input_ids)
            final_attention_masks.append(attention_mask)
            final_indices.append(index)

    return (
        EvaluationWindow(preview_input_ids, preview_attention_masks, preview_indices),
        EvaluationWindow(final_input_ids, final_attention_masks, final_indices),
    )


def split_labels_by_index(
    labels: list[list[int]],
    indices: list[int],
    *,
    split_index: int,
) -> tuple[list[list[int]], list[list[int]]]:
    preview_labels: list[list[int]] = []
    final_labels: list[list[int]] = []
    for index, label in zip(indices, labels, strict=False):
        if index < split_index:
            preview_labels.append(label)
        else:
            final_labels.append(label)
    return preview_labels, final_labels


def compute_window_hash(window: EvaluationWindow, include_data: bool = False) -> str:
    """
    Compute a deterministic hash of an evaluation window.

    Args:
        window: EvaluationWindow to hash
        include_data: Whether to include actual token data in hash

    Returns:
        Hex digest string of the window hash
    """
    hasher = hashlib.sha256()

    hasher.update(str(len(window)).encode())
    hasher.update(str(sorted(window.indices)).encode())

    if include_data:
        for input_ids, attention_mask in zip(
            window.input_ids, window.attention_masks, strict=False
        ):
            hasher.update(str(input_ids).encode())
            hasher.update(str(attention_mask).encode())

    return hasher.hexdigest()


__all__ = [
    "EvaluationWindow",
    "compute_window_hash",
    "split_labels_by_index",
    "split_window_by_index",
]
