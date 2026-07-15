from __future__ import annotations

import logging
import os
import random as _random
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from invarlock.core.exceptions import DataError as _DataErr
from invarlock.core.exceptions import ValidationError as _ValErr

from .data_hf_providers import HFSeq2SeqProvider, HFTextProvider
from .data_stratification import (
    stratify_wikitext_candidates as _stratify_wikitext_candidates,
)
from .data_support import (
    EvaluationWindow,
    _require_load_dataset,
    load_dataset_with_cache_fallback,
    split_window_by_index,
)
from .data_support import (
    estimate_wikitext2_capacity as _estimate_wikitext2_capacity,
)
from .data_support import (
    score_candidates_byte_ngram as _score_candidates_byte_ngram_impl,
)
from .data_tokenization import tokenize_texts_padded

LOGGER = logging.getLogger(__name__)
random = _random


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
        revision: str | None = None,
        **_: Any,
    ):
        self.cache_dir = cache_dir
        self.dataset_name = "Salesforce/wikitext"
        self.config_name = "wikitext-2-raw-v1"
        self.revision = revision or None
        self._validate_dependencies()
        self._last_stratification_stats: dict[str, Any] | None = None
        self._last_batch_size_used: int = 0
        self._last_scorer_profile: dict[str, Any] | None = None
        self._texts_cache: dict[str, list[str]] = {}
        normalized_hint = (device_hint or "").strip().lower()
        self._device_hint: str | None = normalized_hint or None

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
        LOGGER.info("WikiText-2 %s: loading split...", split)
        cached = self._texts_cache.get(split)
        if cached is not None and len(cached) >= max_samples:
            return cached[:max_samples]

        dataset_slice = f"{split}[:{max_samples}]" if max_samples > 0 else split
        dataset = load_dataset_with_cache_fallback(
            self.dataset_name,
            self.config_name,
            split=dataset_slice,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            revision=self.revision,
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

        LOGGER.info("Loaded %s/%s valid samples", len(valid_texts), len(dataset))
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
        LOGGER.info("Creating evaluation windows")
        LOGGER.info("Requested preview/final: %s/%s", preview_n, final_n)
        LOGGER.info("Sampling pool target: %s (reserve %s)", target_pool, reserve)

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

        LOGGER.info("Seed: %s, Seq length: %s", seed, seq_len)
        LOGGER.info("Preview: %s samples", len(preview_window))
        LOGGER.info("Final: %s samples", len(final_window))
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
        LOGGER.info(
            "%s: %s/%s samples tokenized",
            window_name,
            len(valid_indices),
            len(indices),
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


__all__ = [
    "DatasetProvider",
    "HFSeq2SeqProvider",
    "HFTextProvider",
    "SyntheticProvider",
    "WikiText2Provider",
]
