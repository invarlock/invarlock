"""Embedding vocabulary-size consistency checks shared by guard and reporting."""

from __future__ import annotations

from collections import Counter
from typing import Any

_VOCAB_SIZE_ERRORS = (
    AttributeError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def coerce_vocab_counts(vocab_sizes: Any) -> Counter[int]:
    """Count valid integer vocabulary sizes while ignoring malformed entries."""

    counts: Counter[int] = Counter()
    if not isinstance(vocab_sizes, dict):
        return counts
    for value in vocab_sizes.values():
        try:
            counts[int(value)] += 1
        except _VOCAB_SIZE_ERRORS:
            continue
    return counts


def embedding_vocab_size_matches(
    baseline_vocab_sizes: Any,
    current_vocab_sizes: Any,
    module_name: str,
    baseline_size: Any,
) -> tuple[bool, int | None]:
    """Match an embedding by module name, then by vocabulary-size multiplicity."""

    try:
        expected = int(baseline_size)
    except _VOCAB_SIZE_ERRORS:
        # guard-fallback-ok: malformed baseline sizes fail closed as a mismatch.
        return False, None
    current_size = None
    if isinstance(current_vocab_sizes, dict):
        current_size = current_vocab_sizes.get(module_name)
    if current_size is not None:
        try:
            current_int = int(current_size)
        except _VOCAB_SIZE_ERRORS:
            # guard-fallback-ok: malformed current sizes fail closed as a mismatch.
            return False, None
        return current_int == expected, current_int

    baseline_counts = coerce_vocab_counts(baseline_vocab_sizes)
    current_counts = coerce_vocab_counts(current_vocab_sizes)
    if baseline_counts and current_counts.get(expected, 0) >= baseline_counts[expected]:
        return True, expected
    return False, None


__all__ = ["coerce_vocab_counts", "embedding_vocab_size_matches"]
