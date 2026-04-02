from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any


def score_candidates_byte_ngram(
    candidates: list[dict[str, Any]],
    *,
    order: int,
    pad_token: int,
    alpha: float,
) -> dict[str, Any] | None:
    if not candidates:
        return None

    order = max(1, int(order))
    pad_token = int(pad_token)
    alpha = float(alpha)
    vocab_size = pad_token + 1
    context_width = max(order - 1, 0)
    context_modulus = vocab_size ** max(context_width - 1, 0)

    def initial_context_key() -> int:
        key = 0
        for _ in range(context_width):
            key = (key * vocab_size) + pad_token
        return key

    def next_context_key(current_key: int, token: int) -> int:
        if context_width <= 0:
            return 0
        if context_width == 1:
            return int(token)
        return int((current_key % context_modulus) * vocab_size + int(token))

    context_counts: Counter[int] = Counter()
    ngram_counts: Counter[int] = Counter()
    sequences: list[bytes] = []
    start_time = time.perf_counter()

    for candidate in candidates:
        text = candidate.get("text")
        if not isinstance(text, str):
            text = ""
        byte_values = text.encode("utf-8", errors="replace")
        sequences.append(byte_values)
        context_key = initial_context_key()
        for token in byte_values:
            context_counts[context_key] += 1
            ngram_counts[(context_key * vocab_size) + int(token)] += 1
            context_key = next_context_key(context_key, int(token))

    total_tokens = 0
    for candidate, byte_values in zip(candidates, sequences, strict=False):
        loss_sum = 0.0
        token_count = 0
        context_key = initial_context_key()
        for token in byte_values:
            context_count = context_counts.get(context_key, 0)
            ngram_key = (context_key * vocab_size) + int(token)
            ngram_count = ngram_counts.get(ngram_key, 0)
            prob = (ngram_count + alpha) / (context_count + alpha * vocab_size)
            loss_sum += -math.log(prob)
            token_count += 1
            context_key = next_context_key(context_key, int(token))
        candidate["difficulty"] = loss_sum / max(token_count, 1)
        total_tokens += token_count

    elapsed = max(time.perf_counter() - start_time, 1e-9)
    tokens_per_sec = total_tokens / elapsed if total_tokens else 0.0
    return {
        "mode": "byte_ngram",
        "order": order,
        "vocab_size": vocab_size,
        "tokens_processed": total_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_per_sec,
    }


__all__ = ["score_candidates_byte_ngram"]
