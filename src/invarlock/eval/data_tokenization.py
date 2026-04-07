"""
Tokenizer and window materialization helpers for eval datasets.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, cast

from invarlock.core.exceptions import DataError as _DataErr

from .data_windows import EvaluationWindow

_TOKENIZATION_ERRORS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def call_tokenizer(tokenizer: Any, /, *args: Any, **kwargs: Any) -> Any:
    return cast(Any, tokenizer)(*args, **kwargs)


def to_python_token_rows(value: Any, *, batch_size: int) -> list[list[int]]:
    candidate = value
    if batch_size == 1 and hasattr(candidate, "squeeze"):
        try:
            candidate = candidate.squeeze(0)
        except _TOKENIZATION_ERRORS:
            pass
    if hasattr(candidate, "detach"):
        try:
            candidate = candidate.detach()
        except _TOKENIZATION_ERRORS:
            pass
    if hasattr(candidate, "cpu"):
        try:
            candidate = candidate.cpu()
        except _TOKENIZATION_ERRORS:
            pass
    if hasattr(candidate, "tolist"):
        try:
            candidate = candidate.tolist()
        except _TOKENIZATION_ERRORS:
            pass
    if batch_size == 1:
        if (
            isinstance(candidate, list)
            and candidate
            and isinstance(candidate[0], (list, tuple))
        ):
            rows = candidate[:1]
        else:
            rows = [candidate]
    else:
        if not isinstance(candidate, list) or (
            candidate and not isinstance(candidate[0], (list, tuple))
        ):
            raise TypeError("Tokenizer did not return batched rows")
        rows = candidate[:batch_size]
    return [[int(token) for token in row] for row in rows]


def pad_token_ids_and_mask(
    token_ids: Sequence[int],
    *,
    seq_len: int,
    pad_id: int,
) -> tuple[list[int], list[int]]:
    raw_ids = [int(token) for token in token_ids[:seq_len]]
    real_tokens = len(raw_ids)
    if real_tokens < seq_len:
        raw_ids.extend([pad_id] * (seq_len - real_tokens))
    attention_mask = [1] * real_tokens
    if real_tokens < seq_len:
        attention_mask.extend([0] * (seq_len - real_tokens))
    return raw_ids, attention_mask


def extract_padded_token_rows(
    tokens: Any,
    *,
    batch_size: int,
    seq_len: int,
    pad_id: int,
) -> tuple[list[list[int]], list[list[int]]]:
    token_rows = to_python_token_rows(tokens["input_ids"], batch_size=batch_size)
    if len(token_rows) != batch_size:
        raise ValueError("Tokenizer returned unexpected row count")

    attention_value = tokens.get("attention_mask")
    attention_rows = (
        to_python_token_rows(attention_value, batch_size=batch_size)
        if attention_value is not None
        else []
    )

    input_ids_list: list[list[int]] = []
    attention_masks_list: list[list[int]] = []
    for index, token_row in enumerate(token_rows):
        padded_ids, inferred_mask = pad_token_ids_and_mask(
            token_row, seq_len=seq_len, pad_id=pad_id
        )
        if attention_rows:
            mask_row = [int(mask) for mask in attention_rows[index][:seq_len]]
            if len(mask_row) < seq_len:
                mask_row.extend([0] * (seq_len - len(mask_row)))
        elif len(token_row) < seq_len:
            mask_row = inferred_mask
        else:
            mask_row = [1 if token != pad_id else 0 for token in padded_ids]
        input_ids_list.append(padded_ids)
        attention_masks_list.append(mask_row)
    return input_ids_list, attention_masks_list


def encode_text(tokenizer: Any, text: str, seq_len: int) -> list[int]:
    try:
        encoded = tokenizer.encode(
            text,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
    except TypeError:
        encoded = tokenizer.encode(
            text,
            truncation=True,
            max_length=seq_len,
        )
    return [int(token) for token in encoded]


def materialize_token_row(
    tokenizer: Any,
    text: str,
    *,
    seq_len: int,
    pad_id: int,
) -> tuple[list[int], list[int]]:
    if callable(tokenizer):
        tokens = call_tokenizer(
            tokenizer,
            text,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_attention_mask=True,
        )
        token_rows, mask_rows = extract_padded_token_rows(
            tokens,
            batch_size=1,
            seq_len=seq_len,
            pad_id=pad_id,
        )
        return token_rows[0], mask_rows[0]

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        input_ids, attention_mask = pad_token_ids_and_mask(
            encode_text(tokenizer, text, seq_len),
            seq_len=seq_len,
            pad_id=pad_id,
        )
        return input_ids, attention_mask

    raise TypeError(
        "Tokenizer must be callable or expose encode(text, truncation=True, "
        "max_length=..., padding='max_length')."
    )


def tokenize_texts_padded(
    texts: Sequence[str],
    tokenizer: Any,
    seq_len: int,
    *,
    positions: Sequence[int] | None = None,
    warn_on_failure: bool = False,
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    if positions is None:
        positions = list(range(len(texts)))
    if len(texts) != len(positions):
        raise ValueError("texts and positions must have matching lengths")

    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    input_ids_list: list[list[int]] = []
    attention_masks_list: list[list[int]] = []
    kept_positions: list[int] = []
    failures: list[dict[str, Any]] = []
    for text, position in zip(texts, positions, strict=False):
        try:
            input_ids, attention_mask = materialize_token_row(
                tokenizer,
                text,
                seq_len=seq_len,
                pad_id=pad_id,
            )
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
            kept_positions.append(int(position))
        except _TOKENIZATION_ERRORS as exc:
            failures.append({"position": int(position), "error": str(exc)})
            if warn_on_failure:
                warnings.warn(
                    f"Failed to tokenize sample {position}: {exc}",
                    stacklevel=2,
                )
    if failures and not warn_on_failure:
        raise _DataErr(
            code="E304",
            message=(
                "TOKENIZE-INSUFFICIENT: failed to tokenize one or more evaluation "
                "samples"
            ),
            details={
                "requested": int(len(texts)),
                "succeeded": int(len(kept_positions)),
                "failed_positions": [item["position"] for item in failures],
                "errors": [item["error"] for item in failures],
            },
        )

    return input_ids_list, attention_masks_list, kept_positions


def tokenize_combined_pairs(
    pairs: Sequence[tuple[str, str]],
    *,
    tokenizer: Any,
    seq_len: int,
    positions: Sequence[int],
) -> tuple[EvaluationWindow, list[list[int]]]:
    source_texts = [src for src, _ in pairs]
    target_texts = [tgt for _, tgt in pairs]
    src_ids, src_masks, src_positions = tokenize_texts_padded(
        source_texts,
        tokenizer,
        seq_len,
        positions=positions,
    )
    tgt_ids, tgt_masks, tgt_positions = tokenize_texts_padded(
        target_texts,
        tokenizer,
        seq_len,
        positions=positions,
    )
    src_map = {
        position: (input_ids, attention_mask)
        for position, input_ids, attention_mask in zip(
            src_positions, src_ids, src_masks, strict=False
        )
    }
    tgt_map = {
        position: [
            int(token) if int(mask) else -100
            for token, mask in zip(target_ids, target_mask, strict=False)
        ]
        for position, target_ids, target_mask in zip(
            tgt_positions, tgt_ids, tgt_masks, strict=False
        )
    }
    kept_positions = [
        int(position)
        for position in positions
        if position in src_map and position in tgt_map
    ]
    window = EvaluationWindow(
        [src_map[position][0] for position in kept_positions],
        [src_map[position][1] for position in kept_positions],
        kept_positions,
    )
    labels = [tgt_map[position] for position in kept_positions]
    return window, labels


__all__ = [
    "encode_text",
    "extract_padded_token_rows",
    "materialize_token_row",
    "pad_token_ids_and_mask",
    "tokenize_combined_pairs",
    "tokenize_texts_padded",
]
