from __future__ import annotations

import math
import os
import random
import time
from abc import abstractmethod
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from invarlock.core.exceptions import DataError as _DataErr
from invarlock.core.exceptions import ValidationError as _ValErr

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
        texts = self.load(split=split, max_samples=2000)
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

        tokenized = self._collect_tokenized_samples(
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
            tokenized_subset = self._collect_tokenized_samples(
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

        result = {
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

        def mean_difficulty(items: list[dict[str, Any]]) -> float:
            if not items:
                return 0.0
            return float(sum(item["difficulty"] for item in items) / len(items))

        for _ in range(100):
            if not preview_candidates or not final_candidates:
                break
            diff = mean_difficulty(preview_candidates) - mean_difficulty(
                final_candidates
            )
            if abs(diff) <= 1e-4:
                break
            if diff < 0:
                preview_candidate = min(
                    preview_candidates, key=lambda c: c["difficulty"]
                )
                final_candidate = max(final_candidates, key=lambda c: c["difficulty"])
            else:
                preview_candidate = max(
                    preview_candidates, key=lambda c: c["difficulty"]
                )
                final_candidate = min(final_candidates, key=lambda c: c["difficulty"])
            if preview_candidate is final_candidate:
                break
            preview_candidates.remove(preview_candidate)
            final_candidates.remove(final_candidate)
            preview_candidates.append(final_candidate)
            final_candidates.append(preview_candidate)
            new_diff = mean_difficulty(preview_candidates) - mean_difficulty(
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
        final_candidates.sort(
            key=lambda item: (item["difficulty"], item["dataset_index"])
        )
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
        self._last_stratification_stats = {
            "pool_size": len(selected_candidates),
            "reserve": reserve,
            "batch_size_used": int(self._last_batch_size_used),
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
        if not candidates:
            self._last_batch_size_used = 0
            self._last_scorer_profile = None
            return False

        order = max(1, int(self._BYTE_NGRAM_ORDER))
        pad_token = int(self._BYTE_NGRAM_PAD)
        alpha = float(self._BYTE_NGRAM_ALPHA)
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

        self._last_batch_size_used = len(candidates)
        elapsed = max(time.perf_counter() - start_time, 1e-9)
        tokens_per_sec = total_tokens / elapsed if total_tokens else 0.0
        self._last_scorer_profile = {
            "mode": "byte_ngram",
            "order": order,
            "vocab_size": vocab_size,
            "tokens_processed": total_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_per_sec,
        }
        return True

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
