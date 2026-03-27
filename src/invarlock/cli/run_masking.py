"""Masked-LM helper utilities extracted from the run command shell."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from invarlock.cli.run_pairing_helpers import (
    _safe_int,
    _tensor_or_list_to_ints,
    _to_int_list,
)


def _derive_mlm_seed(base_seed: int, window_id: str | int, position: int) -> int:
    payload = f"{base_seed}:{window_id}:{position}".encode()
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _apply_mlm_masks(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    mask_prob: float,
    seed: int,
    random_token_prob: float,
    original_token_prob: float,
    prefix: str,
) -> tuple[int, list[int]]:
    """Apply basic BERT-style MLM masking to tokenized records in-place."""
    if mask_prob <= 0.0:
        zeroed: list[int] = []
        for record in records:
            length = len(record["input_ids"])
            record["labels"] = [-100] * length
            record["mlm_masked"] = 0
            zeroed.append(0)
        return 0, zeroed

    vocab_size = _safe_int(getattr(tokenizer, "vocab_size", 0))
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        raise RuntimeError(
            "Tokenizer does not define mask_token_id; required for MLM evaluation."
        )
    try:
        mask_token_id = int(mask_token_id)
    except (TypeError, ValueError, OverflowError):
        mask_token_id = _safe_int(mask_token_id, 0)

    special_ids = set()
    for attr in (
        "cls_token_id",
        "sep_token_id",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    ):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            try:
                special_ids.add(int(value))
            except Exception:
                pass
    try:
        special_ids.update(
            int(token) for token in getattr(tokenizer, "all_special_ids", []) or []
        )
    except Exception:
        pass

    masked_total = 0
    masked_counts: list[int] = []
    for idx_record, record in enumerate(records):
        window_id = record.get("window_id", f"{prefix}:{idx_record}")
        input_ids = _tensor_or_list_to_ints(record.get("input_ids", []))
        attention = _tensor_or_list_to_ints(record.get("attention_mask", []))
        labels = [-100] * len(input_ids)

        masked = 0
        for pos, (token, attn) in enumerate(zip(input_ids, attention, strict=False)):
            if not attn:
                continue
            if int(token) in special_ids:
                continue
            if random.random() < mask_prob:
                rng = random.Random(_derive_mlm_seed(seed, window_id, pos))
                labels[pos] = int(token)
                choice = rng.random()
                if choice < 1.0 - (random_token_prob + original_token_prob):
                    input_ids[pos] = mask_token_id
                elif choice < 1.0 - original_token_prob and vocab_size > 0:
                    rng2 = random.Random(_derive_mlm_seed(seed + 17, window_id, pos))
                    input_ids[pos] = rng2.randint(0, max(1, vocab_size - 1))
                masked += 1

        if masked == 0:
            candidate_positions = [
                pos
                for pos, (token, attn) in enumerate(zip(input_ids, attention, strict=False))
                if attn and int(token) not in special_ids
            ]
            if candidate_positions:
                pos = candidate_positions[len(candidate_positions) // 2]
                rng = random.Random(_derive_mlm_seed(seed + 17, window_id, pos))
                labels[pos] = int(input_ids[pos])
                masked = 1
                choice = rng.random()
                if choice < 1.0 - (random_token_prob + original_token_prob):
                    input_ids[pos] = mask_token_id
                elif choice < 1.0 - original_token_prob and vocab_size > 0:
                    input_ids[pos] = rng.randrange(vocab_size)

        record["input_ids"] = _to_int_list(input_ids)
        record["attention_mask"] = _to_int_list(attention)
        record["labels"] = _to_int_list(labels)
        record["mlm_masked"] = masked
        masked_total += masked
        masked_counts.append(masked)

    return masked_total, masked_counts


def _tokenizer_digest(tokenizer: Any) -> str:
    """Compute a stable digest for a tokenizer config."""
    try:
        if hasattr(tokenizer, "get_vocab"):
            try:
                items = getattr(tokenizer.get_vocab(), "items", None)
                if callable(items):
                    pairs = list(items())
                    pairs = [
                        (str(key), int(value))
                        for key, value in pairs
                        if isinstance(key, str | int)
                    ]
                    payload = json.dumps(sorted(pairs), separators=(",", ":")).encode()
                    return hashlib.sha256(payload).hexdigest()
            except Exception:
                pass
        vocab = getattr(tokenizer, "vocab", None)
        if isinstance(vocab, list):
            try:
                payload = json.dumps(
                    [(str(key), int(value)) for key, value in vocab],
                    separators=(",", ":"),
                ).encode()
                return hashlib.sha256(payload).hexdigest()
            except Exception:
                pass
        attrs = {
            "name": getattr(tokenizer, "name_or_path", None),
            "eos": getattr(tokenizer, "eos_token", None),
            "pad": getattr(tokenizer, "pad_token", None),
            "size": _safe_int(getattr(tokenizer, "vocab_size", 0)),
        }
        return hashlib.sha256(json.dumps(attrs, sort_keys=True).encode()).hexdigest()
    except Exception:
        return "unknown-tokenizer"

