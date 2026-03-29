from __future__ import annotations

import os
import random
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from invarlock.core.exceptions import DataError as _DataErr
from invarlock.core.exceptions import ValidationError as _ValErr

from .data_capacity import estimate_wikitext2_capacity as _estimate_wikitext2_capacity
from .data_difficulty import (
    score_candidates_byte_ngram as _score_candidates_byte_ngram_impl,
)
from .data_stratification import (
    stratify_wikitext_candidates as _stratify_wikitext_candidates,
)
from .data_support import EventEmitter, _require_load_dataset
from .data_tokenization import tokenize_combined_pairs, tokenize_texts_padded
from .data_windows import EvaluationWindow, split_labels_by_index, split_window_by_index
from .providers.local_jsonl import LocalJSONLProvider
from .providers.local_jsonl_pairs import LocalJSONLPairsProvider


class DatasetProvider(Protocol):
    name: str

    @abstractmethod
    def load(self, split: str = "validation", **kwargs: Any) -> list[str]: ...

    @abstractmethod
    def windows(
        self,
        tokenizer: Any,
        *,
        seq_len: int = 128,
        stride: int = 64,
        preview_n: int = 100,
        final_n: int = 100,
        seed: int = 42,
        split: str = "validation",
    ) -> tuple[EvaluationWindow, EvaluationWindow]: ...

    def estimate_capacity(
        self,
        tokenizer: Any,
        *,
        seq_len: int,
        stride: int,
        split: str = "validation",
        target_total: int | None = None,
        fast_mode: bool = False,
    ) -> dict[str, Any]: ...

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "type": "dataset_provider"}


class WikiText2Provider:
    name = "wikitext2"
    _BYTE_NGRAM_ORDER = 4
    _BYTE_NGRAM_PAD = 256
    _BYTE_NGRAM_ALPHA = 1.0

    def __init__(
        self,
        cache_dir: Path | None = None,
        device_hint: str | None = None,
        emit: EventEmitter | None = None,
        **_: Any,
    ):
        self.cache_dir = cache_dir
        self._emit_event = emit
        self._validate_dependencies()
        self._last_stratification_stats: dict[str, Any] | None = None
        self._last_batch_size_used: int = 0
        self._last_scorer_profile: dict[str, Any] | None = None
        self._texts_cache: dict[str, list[str]] = {}
        normalized_hint = (device_hint or "").strip().lower()
        self._device_hint: str | None = normalized_hint or None

    def _event(self, tag: str, message: str, *, emoji: str | None = None) -> None:
        if self._emit_event is None:
            return
        self._emit_event(tag, message, emoji)

    def _validate_dependencies(self) -> None:
        _require_load_dataset(
            "DEPENDENCY-MISSING: datasets library required for WikiText-2 loading"
        )

    def estimate_capacity(
        self,
        tokenizer: Any,
        *,
        seq_len: int,
        stride: int,
        split: str = "validation",
        target_total: int | None = None,
        fast_mode: bool = False,
    ) -> dict[str, Any]:
        return _estimate_wikitext2_capacity(
            load_fn=self.load,
            collect_tokenized_samples_fn=self._collect_tokenized_samples,
            tokenizer=tokenizer,
            seq_len=seq_len,
            stride=stride,
            split=split,
            target_total=target_total,
            fast_mode=fast_mode,
        )

    def load(
        self, split: str = "validation", max_samples: int = 2000, **kwargs: Any
    ) -> list[str]:
        self._event("DATA", f"WikiText-2 {split}: loading split...", emoji="📚")
        cached = self._texts_cache.get(split)
        if cached is not None and len(cached) >= max_samples:
            return cached[:max_samples]

        dataset_slice = f"{split}[:{max_samples}]" if max_samples > 0 else split
        load_dataset = _require_load_dataset(
            "DEPENDENCY-MISSING: datasets library required for WikiText-2 loading"
        )
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=dataset_slice,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )

        valid_texts: list[str] = []
        for item in dataset:
            text = str(item.get("text", "")).strip()
            if len(text) >= 20 and any(c.isalpha() for c in text):
                valid_texts.append(text)

        if str(os.environ.get("INVARLOCK_DEDUP_TEXTS", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            seen: set[str] = set()
            deduped: list[str] = []
            for text in valid_texts:
                if text not in seen:
                    seen.add(text)
                    deduped.append(text)
            valid_texts = deduped

        prev = self._texts_cache.get(split)
        if prev is None or len(valid_texts) > len(prev):
            self._texts_cache[split] = list(valid_texts)

        self._event("DATA", f"Loaded {len(valid_texts)}/{len(dataset)} valid samples")
        return valid_texts

    def windows(
        self,
        tokenizer: Any,
        *,
        seq_len: int = 128,
        stride: int = 64,
        preview_n: int = 100,
        final_n: int = 100,
        seed: int = 42,
        split: str = "validation",
    ) -> tuple[EvaluationWindow, EvaluationWindow]:
        total_required = preview_n + final_n
        if total_required <= 0:
            raise _ValErr(
                code="E302",
                message="VALIDATION-FAILED: preview/final must be positive",
            )

        extra_pool = max(500, int(0.5 * total_required))
        max_samples = max(total_required + extra_pool, 2000)
        texts = self.load(split=split, max_samples=max_samples)

        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(len(texts)).tolist()
        reserve = max(16, int(0.1 * total_required))
        target_pool = min(len(texts), total_required + reserve * 2)
        if target_pool < total_required:
            raise _DataErr(
                code="E303",
                message=(
                    "CAPACITY-INSUFFICIENT: not enough valid samples for requested preview/final"
                ),
                details={
                    "have": int(len(texts)),
                    "preview": int(preview_n),
                    "final": int(final_n),
                },
            )

        candidates: list[dict[str, Any]] = []
        used_indices: set[int] = set()
        cursor = 0
        chunk_size = max(64, min(256, target_pool))
        self._event("DATA", "Creating evaluation windows:", emoji="📊")
        self._event("DATA", f"Requested preview/final: {preview_n}/{final_n}")
        self._event("DATA", f"Sampling pool target: {target_pool} (reserve {reserve})")

        while len(candidates) < total_required + reserve and cursor < len(
            shuffled_indices
        ):
            batch = shuffled_indices[cursor : cursor + chunk_size]
            cursor += chunk_size
            tokenized_batch = self._collect_tokenized_samples(
                texts, batch, tokenizer, seq_len
            )
            for idx, input_ids, attention_mask, real_tokens in tokenized_batch:
                if idx in used_indices:
                    continue
                used_indices.add(idx)
                candidates.append(
                    {
                        "dataset_index": idx,
                        "text": texts[idx],
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_count": real_tokens,
                        "seq_len": len(input_ids),
                    }
                )
            if cursor >= len(shuffled_indices) and len(candidates) < total_required:
                break

        if len(candidates) < total_required:
            raise _DataErr(
                code="E304",
                message=(
                    "TOKENIZE-INSUFFICIENT: failed to gather enough tokenized samples"
                ),
                details={"needed": int(total_required), "got": int(len(candidates))},
            )

        self._score_candidates_byte_ngram(candidates)
        preview_window, final_window, stratification_stats = (
            _stratify_wikitext_candidates(
                candidates,
                preview_n=preview_n,
                final_n=final_n,
                reserve=reserve,
                batch_size_used=self._last_batch_size_used,
            )
        )
        self._last_stratification_stats = stratification_stats

        self._event("DATA", f"Seed: {seed}, Seq length: {seq_len}")
        self._event("DATA", f"Preview: {len(preview_window)} samples")
        self._event("DATA", f"Final: {len(final_window)} samples")
        return preview_window, final_window

    def _collect_tokenized_samples(
        self,
        texts: Sequence[str],
        indices: Sequence[int],
        tokenizer: Any,
        seq_len: int,
    ) -> list[tuple[int, list[int], list[int], int]]:
        batch_texts: list[str] = []
        batch_indices: list[int] = []
        for idx in indices:
            if idx >= len(texts):
                continue
            batch_indices.append(int(idx))
            batch_texts.append(texts[idx])

        input_ids_list, attention_masks_list, valid_indices = tokenize_texts_padded(
            batch_texts,
            tokenizer,
            seq_len,
            positions=batch_indices,
            warn_on_failure=True,
        )

        results: list[tuple[int, list[int], list[int], int]] = []
        for idx, input_ids, attention_mask in zip(
            valid_indices,
            input_ids_list,
            attention_masks_list,
            strict=False,
        ):
            real_tokens = int(sum(attention_mask))
            if real_tokens > 1:
                results.append((idx, input_ids, attention_mask, real_tokens))
        return results

    def _score_candidates_byte_ngram(self, candidates: list[dict[str, Any]]) -> bool:
        profile = _score_candidates_byte_ngram_impl(
            candidates,
            order=self._BYTE_NGRAM_ORDER,
            pad_token=self._BYTE_NGRAM_PAD,
            alpha=self._BYTE_NGRAM_ALPHA,
        )
        self._last_batch_size_used = len(candidates) if profile is not None else 0
        self._last_scorer_profile = profile
        return profile is not None

    def _tokenize_samples(
        self,
        texts: list[str],
        indices: list[int],
        tokenizer: Any,
        seq_len: int,
        window_name: str,
    ) -> EvaluationWindow:
        collected = self._collect_tokenized_samples(texts, indices, tokenizer, seq_len)
        input_ids_list = [entry[1] for entry in collected]
        attention_masks_list = [entry[2] for entry in collected]
        valid_indices = [entry[0] for entry in collected]
        self._event(
            "DATA",
            f"{window_name}: {len(valid_indices)}/{len(indices)} samples tokenized",
        )
        return EvaluationWindow(
            input_ids=input_ids_list,
            attention_masks=attention_masks_list,
            indices=valid_indices,
        )

    @property
    def stratification_stats(self) -> dict[str, Any] | None:
        return self._last_stratification_stats

    @property
    def scorer_profile(self) -> dict[str, Any] | None:
        return self._last_scorer_profile

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "dataset_provider",
            "dataset": "wikitext-2-raw-v1",
            "source": "huggingface/datasets",
            "deterministic": True,
            "default_split": "validation",
            "requires": ["datasets"],
        }


class SyntheticProvider:
    name = "synthetic"

    def __init__(
        self,
        base_samples: list[str] | None = None,
        emit: EventEmitter | None = None,
    ):
        self.base_samples = base_samples or self._default_samples()
        self._load_cache: dict[int, list[str]] = {}

    def _default_samples(self) -> list[str]:
        return [
            "The weather today is quite pleasant with clear skies and gentle winds.",
            "Scientists have discovered a new species in the Amazon rainforest region.",
            "The stock market showed significant gains during this quarter's trading.",
            "Technology companies are investing heavily in artificial intelligence research.",
            "The new restaurant downtown serves excellent Mediterranean cuisine daily.",
            "Climate change continues to affect global weather patterns significantly.",
            "The university announced new programs in data science and engineering.",
            "Renewable energy sources are becoming more cost-effective than fossil fuels.",
            "The museum exhibition features artwork from the Renaissance period.",
            "Public transportation systems are being upgraded in major cities worldwide.",
            "Medical researchers published breakthrough findings about genetic therapy.",
            "The concert hall will host a performance by the symphony orchestra.",
            "Local farmers are adopting sustainable agricultural practices this season.",
            "The new software update includes enhanced security features and performance.",
            "International trade agreements are being renegotiated between countries.",
        ]

    def estimate_capacity(
        self,
        tokenizer: Any,
        *,
        seq_len: int,
        stride: int,
        split: str = "validation",
        target_total: int | None = None,
        fast_mode: bool = False,
    ) -> dict[str, Any]:
        available = len(self.base_samples)
        return {
            "total_tokens": int(available * seq_len),
            "available_nonoverlap": int(available),
            "available_unique": int(available),
            "dedupe_rate": 0.0,
            "stride": int(stride),
            "seq_len": int(seq_len),
            "candidate_unique": int(available),
            "candidate_limit": int(available),
        }

    def load(
        self, split: str = "validation", max_samples: int = 500, **kwargs: Any
    ) -> list[str]:
        cached = self._load_cache.get(int(max_samples))
        if cached is not None:
            return cached

        expanded_samples: list[str] = []
        variations = [
            lambda s: s,
            lambda s: f"Recently, {s.lower()}",
            lambda s: f"According to reports, {s.lower()}",
            lambda s: f"It is notable that {s.lower()}",
            lambda s: f"Furthermore, {s.lower()}",
            lambda s: f"In addition, {s.lower()}",
        ]
        for variation in variations:
            for base_text in self.base_samples:
                expanded_samples.append(variation(base_text))
                if len(expanded_samples) >= max_samples:
                    self._load_cache[int(max_samples)] = expanded_samples
                    return expanded_samples

        idx = 0
        while len(expanded_samples) < max_samples:
            base_text = self.base_samples[idx % len(self.base_samples)]
            variation = variations[(idx // len(self.base_samples)) % len(variations)]
            expanded_samples.append(
                f"{variation(base_text)} [synthetic #{len(expanded_samples)}]"
            )
            idx += 1

        self._load_cache[int(max_samples)] = expanded_samples
        return expanded_samples

    def windows(
        self,
        tokenizer: Any,
        *,
        seq_len: int = 128,
        stride: int = 64,
        preview_n: int = 100,
        final_n: int = 100,
        seed: int = 42,
        split: str = "validation",
    ) -> tuple[EvaluationWindow, EvaluationWindow]:
        texts = self.load(split=split, max_samples=preview_n + final_n)
        total = min(len(texts), int(preview_n) + int(final_n))
        combined_window = self._simple_tokenize(
            texts[:total], tokenizer, seq_len, list(range(total))
        )
        return split_window_by_index(combined_window, split_index=preview_n)

    def _simple_tokenize(
        self, texts: list[str], tokenizer: Any, seq_len: int, indices: list[int]
    ) -> EvaluationWindow:
        input_ids_list, attention_masks_list, valid_indices = tokenize_texts_padded(
            texts,
            tokenizer,
            seq_len,
            positions=indices,
        )
        if not input_ids_list:
            raise _DataErr(
                code="E304",
                message="TOKENIZE-INSUFFICIENT: failed to tokenize synthetic samples",
                details={"requested": int(len(texts)), "got": int(len(valid_indices))},
            )
        return EvaluationWindow(
            input_ids=input_ids_list,
            attention_masks=attention_masks_list,
            indices=valid_indices,
        )

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "dataset_provider",
            "dataset": "synthetic",
            "source": "generated",
            "deterministic": True,
            "base_samples": len(self.base_samples),
        }


class HFTextProvider:
    name = "hf_text"

    def __init__(
        self,
        dataset_name: str | None = None,
        config_name: str | None = None,
        text_field: str = "text",
        cache_dir: str | None = None,
        trust_remote_code: bool = False,
        max_samples: int = 2000,
        emit: EventEmitter | None = None,
    ):
        _require_load_dataset(
            "DEPENDENCY-MISSING: datasets library required for hf_text provider"
        )
        self.dataset_name = dataset_name or "wikitext"
        self.config_name = config_name or None
        self.text_field = text_field
        self.cache_dir = cache_dir
        self.trust_remote_code = bool(trust_remote_code)
        self.max_samples = int(max_samples)
        self._texts_cache: dict[str, list[str]] = {}

    def load(self, split: str = "validation", **kwargs: Any) -> list[str]:
        cached = self._texts_cache.get(split)
        if cached is not None:
            return cached
        load_dataset = _require_load_dataset(
            "DEPENDENCY-MISSING: datasets library required for hf_text provider"
        )
        ds = load_dataset(
            path=self.dataset_name,
            name=self.config_name,
            split=split,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        texts: list[str] = []
        count = 0
        for row in ds:
            if self.text_field not in row:
                continue
            value = row[self.text_field]
            if isinstance(value, str) and value.strip():
                texts.append(value)
                count += 1
                if count >= self.max_samples:
                    break
        self._texts_cache[split] = texts
        return texts

    def _simple_tokenize(
        self, texts: list[str], tokenizer: Any, seq_len: int, indices: list[int]
    ) -> EvaluationWindow:
        input_ids_list, attention_masks_list, valid_indices = tokenize_texts_padded(
            texts,
            tokenizer,
            seq_len,
            positions=indices,
        )
        return EvaluationWindow(
            input_ids_list,
            attention_masks_list,
            valid_indices,
        )

    def _token_signature(
        self, input_ids: Sequence[int], attention_mask: Sequence[int]
    ) -> tuple[int, ...]:
        return tuple(
            int(token_id)
            for token_id, mask in zip(input_ids, attention_mask, strict=False)
            if int(mask)
        )

    def _collect_unique_window_samples(
        self,
        texts: Sequence[str],
        tokenizer: Any,
        *,
        seq_len: int,
        positions: Sequence[int],
        target_total: int,
    ) -> list[tuple[int, list[int], list[int]]]:
        unique_samples: list[tuple[int, list[int], list[int]]] = []
        seen_signatures: set[tuple[int, ...]] = set()
        chunk_size = max(64, min(256, int(target_total or 1)))
        for start in range(0, len(positions), chunk_size):
            batch_positions = [
                int(pos) for pos in positions[start : start + chunk_size]
            ]
            batch_texts = [texts[pos] for pos in batch_positions]
            tokenized_window = self._simple_tokenize(
                batch_texts,
                tokenizer,
                seq_len,
                batch_positions,
            )
            for idx, input_ids, attention_mask in zip(
                tokenized_window.indices,
                tokenized_window.input_ids,
                tokenized_window.attention_masks,
                strict=False,
            ):
                signature = self._token_signature(input_ids, attention_mask)
                if len(signature) <= 1 or signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                unique_samples.append(
                    (
                        int(idx),
                        [int(token) for token in input_ids],
                        [int(mask) for mask in attention_mask],
                    )
                )
                if len(unique_samples) >= target_total:
                    return unique_samples
        return unique_samples

    def windows(
        self,
        tokenizer: Any,
        *,
        seq_len: int = 128,
        stride: int = 64,
        preview_n: int = 100,
        final_n: int = 100,
        seed: int = 42,
        split: str = "validation",
    ) -> tuple[EvaluationWindow, EvaluationWindow]:
        texts = self.load(split=split)
        if not texts:
            raise _DataErr(
                code="E306",
                message=(
                    "NO-SAMPLES: hf_text produced no samples; check dataset_name/config_name/text_field"
                ),
            )
        total_required = int(preview_n) + int(final_n)
        if total_required <= 0:
            raise _ValErr(
                code="E302",
                message="VALIDATION-FAILED: preview/final must be positive",
            )
        selected_positions = list(range(len(texts)))
        random.Random(int(seed)).shuffle(selected_positions)
        unique_samples = self._collect_unique_window_samples(
            texts,
            tokenizer,
            seq_len=seq_len,
            positions=selected_positions,
            target_total=total_required,
        )
        if len(unique_samples) < total_required:
            raise _DataErr(
                code="E304",
                message=(
                    "TOKENIZE-INSUFFICIENT: failed to gather enough unique tokenized samples"
                ),
                details={
                    "needed": int(total_required),
                    "got": int(len(unique_samples)),
                },
            )
        preview_samples = unique_samples[: int(preview_n)]
        final_samples = unique_samples[int(preview_n) : total_required]
        return (
            EvaluationWindow(
                input_ids=[sample[1] for sample in preview_samples],
                attention_masks=[sample[2] for sample in preview_samples],
                indices=[sample[0] for sample in preview_samples],
            ),
            EvaluationWindow(
                input_ids=[sample[1] for sample in final_samples],
                attention_masks=[sample[2] for sample in final_samples],
                indices=[sample[0] for sample in final_samples],
            ),
        )

    def estimate_capacity(
        self,
        tokenizer: Any,
        *,
        seq_len: int,
        stride: int,
        split: str = "validation",
        target_total: int | None = None,
        fast_mode: bool = False,
    ) -> dict[str, Any]:
        texts = self.load(split=split)
        return {
            "total_tokens": 0,
            "available_nonoverlap": len(texts),
            "available_unique": len(texts),
            "dedupe_rate": 0.0,
            "stride": stride,
            "seq_len": seq_len,
            "candidate_unique": len(texts),
            "candidate_limit": min(len(texts), self.max_samples),
        }


class HFSeq2SeqProvider:
    name = "hf_seq2seq"

    def __init__(
        self,
        dataset_name: str,
        config_name: str | None = None,
        src_field: str = "source",
        tgt_field: str = "target",
        cache_dir: str | None = None,
        max_samples: int = 2000,
        emit: EventEmitter | None = None,
    ) -> None:
        _require_load_dataset(
            "DEPENDENCY-MISSING: datasets library required for hf_seq2seq provider"
        )
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.cache_dir = cache_dir
        self.max_samples = int(max_samples)
        self.last_preview_labels: list[list[int]] | None = None
        self.last_final_labels: list[list[int]] | None = None
        self._pairs_cache: dict[str, list[tuple[str, str]]] = {}

    def _load_pairs(self, split: str) -> list[tuple[str, str]]:
        cached = self._pairs_cache.get(split)
        if cached is not None:
            return cached
        load_dataset = _require_load_dataset(
            "DEPENDENCY-MISSING: datasets library required for hf_seq2seq provider"
        )
        ds = load_dataset(
            path=self.dataset_name,
            name=self.config_name,
            split=split,
            cache_dir=self.cache_dir,
        )
        out: list[tuple[str, str]] = []
        count = 0
        for row in ds:
            src = row.get(self.src_field)
            tgt = row.get(self.tgt_field)
            if (
                isinstance(src, str)
                and src.strip()
                and isinstance(tgt, str)
                and tgt.strip()
            ):
                out.append((src, tgt))
                count += 1
                if count >= self.max_samples:
                    break
        self._pairs_cache[split] = out
        return out

    def windows(
        self,
        tokenizer: Any,
        *,
        seq_len: int = 128,
        stride: int = 64,
        preview_n: int = 100,
        final_n: int = 100,
        seed: int = 42,
        split: str = "validation",
    ) -> tuple[EvaluationWindow, EvaluationWindow]:
        pairs = self._load_pairs(split)
        if not pairs:
            raise _DataErr(
                code="E307",
                message=(
                    "NO-PAIRS: hf_seq2seq produced no pairs; check src_field/tgt_field"
                ),
            )
        prev_pairs = pairs[:preview_n]
        fin_pairs = pairs[preview_n : preview_n + final_n]
        combined_pairs = prev_pairs + fin_pairs
        combined_positions = list(range(len(prev_pairs))) + list(
            range(preview_n, preview_n + len(fin_pairs))
        )
        combined_window, combined_labels = tokenize_combined_pairs(
            combined_pairs,
            tokenizer=tokenizer,
            seq_len=seq_len,
            positions=combined_positions,
        )
        preview_window, final_window = split_window_by_index(
            combined_window, split_index=preview_n
        )
        self.last_preview_labels, self.last_final_labels = split_labels_by_index(
            combined_labels,
            combined_window.indices,
            split_index=preview_n,
        )
        return preview_window, final_window

    def estimate_capacity(
        self,
        tokenizer: Any,
        *,
        seq_len: int,
        stride: int,
        split: str = "validation",
        target_total: int | None = None,
        fast_mode: bool = False,
    ) -> dict[str, Any]:
        pairs = self._load_pairs(split)
        n = len(pairs)
        return {
            "total_tokens": int(n * seq_len),
            "available_nonoverlap": n,
            "available_unique": n,
            "dedupe_rate": 0.0,
            "stride": stride,
            "seq_len": seq_len,
            "candidate_unique": n,
            "candidate_limit": n,
            "tokens_available": int(n * seq_len),
            "examples_available": n,
        }


__all__ = [
    "DatasetProvider",
    "HFSeq2SeqProvider",
    "HFTextProvider",
    "LocalJSONLProvider",
    "LocalJSONLPairsProvider",
    "SyntheticProvider",
    "WikiText2Provider",
]
