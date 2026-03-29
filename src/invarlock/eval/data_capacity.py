from __future__ import annotations

import os
from typing import Any, Callable


def estimate_wikitext2_capacity(
    *,
    load_fn: Callable[..., list[str]],
    collect_tokenized_samples_fn: Callable[
        [list[str], list[int], Any, int], list[tuple[int, list[int], list[int], int]]
    ],
    tokenizer: Any,
    seq_len: int,
    stride: int,
    split: str = "validation",
    target_total: int | None = None,
    fast_mode: bool = False,
) -> dict[str, Any]:
    texts = load_fn(split=split, max_samples=2000)
    if not texts:
        return {
            "total_tokens": 0,
            "available_nonoverlap": 0,
            "available_unique": 0,
            "dedupe_rate": 0.0,
            "stride": stride,
            "seq_len": seq_len,
            "candidate_unique": 0,
            "candidate_limit": 0,
        }

    env_fast = os.environ.get("INVARLOCK_CAPACITY_FAST", "")
    env_fast_flag = isinstance(env_fast, str) and env_fast.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    use_fast = bool(fast_mode) or env_fast_flag
    if use_fast:
        base_available = len(texts)
        requested_total = int(target_total or 0)
        approx_available = (
            max(base_available, requested_total)
            if requested_total > 0
            else base_available
        )
        total_tokens = int(max(approx_available, 0) * seq_len)
        approx_available = int(max(approx_available, 0))
        return {
            "total_tokens": total_tokens,
            "available_nonoverlap": approx_available,
            "available_unique": approx_available,
            "dedupe_rate": 0.0,
            "stride": int(stride),
            "seq_len": int(seq_len),
            "candidate_unique": approx_available,
            "candidate_limit": approx_available,
        }

    tokenized = collect_tokenized_samples_fn(
        texts, list(range(len(texts))), tokenizer, seq_len
    )
    total_tokens = sum(item[3] for item in tokenized)
    available_nonoverlap = len(tokenized)
    unique_sequences = {
        tuple(
            int(tok_id)
            for tok_id, mask in zip(input_ids, attention_mask, strict=False)
            if mask
        )
        for _, input_ids, attention_mask, _ in tokenized
    }
    available_unique = len(unique_sequences)
    dedupe_rate = (
        0.0
        if available_nonoverlap == 0
        else max(
            0.0,
            1.0 - (available_unique / float(max(available_nonoverlap, 1))),
        )
    )

    candidate_unique = None
    candidate_limit = None
    if target_total is not None and target_total > 0:
        reserve_buffer = max(int(target_total * 0.2), 64)
        candidate_limit = min(len(texts), target_total + reserve_buffer)
        tokenized_subset = collect_tokenized_samples_fn(
            texts, list(range(candidate_limit)), tokenizer, seq_len
        )
        subset_signatures = {
            tuple(
                int(tok)
                for tok, mask in zip(entry[1], entry[2], strict=False)
                if mask
            )
            for entry in tokenized_subset
        }
        candidate_unique = len(subset_signatures)

    result: dict[str, Any] = {
        "total_tokens": int(total_tokens),
        "available_nonoverlap": int(available_nonoverlap),
        "available_unique": int(available_unique),
        "dedupe_rate": float(dedupe_rate),
        "stride": int(stride),
        "seq_len": int(seq_len),
    }
    if candidate_unique is not None:
        result["candidate_unique"] = int(candidate_unique)
        result["candidate_limit"] = int(candidate_limit or 0)
    return result


__all__ = ["estimate_wikitext2_capacity"]
