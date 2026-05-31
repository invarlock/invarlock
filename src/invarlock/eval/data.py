"""
Canonical evaluation data entrypoint.

Owns the provider registry and the public eval-data import surface.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from invarlock.core.exceptions import DataError as _DataErr
from invarlock.core.exceptions import ValidationError as _ValErr

from .data_providers import (
    DatasetProvider,
    HFSeq2SeqProvider,
    HFTextProvider,
    SyntheticProvider,
    WikiText2Provider,
)
from .data_support import (
    EvaluationProvider,
    EvaluationWindow,
    compute_window_hash,
    load_local_jsonl_pairs,
    load_local_jsonl_texts,
    local_jsonl_cache_key,
    resolve_local_jsonl_files,
    split_labels_by_index,
    split_window_by_index,
)
from .data_tokenization import tokenize_combined_pairs, tokenize_texts_padded


class LocalJSONLProvider:
    name = "local_jsonl"

    def __init__(
        self,
        file: str | None = None,
        path: str | None = None,
        data_files: str | list[str] | None = None,
        text_field: str = "text",
        max_samples: int = 2000,
    ) -> None:
        self.file = file
        self.path = path
        self.data_files = data_files
        self.text_field = text_field or "text"
        self.max_samples = int(max_samples)
        self._load_cache: tuple[tuple[Any, ...], list[str]] | None = None

    def _resolve_files(self) -> list[Path]:
        return resolve_local_jsonl_files(
            file=self.file, path=self.path, data_files=self.data_files
        )

    def load(self, split: str = "validation", **kwargs: Any) -> list[str]:
        files = self._resolve_files()
        cache_key = local_jsonl_cache_key(
            files,
            field_names=(self.text_field,),
            max_samples=self.max_samples,
        )
        if self._load_cache is not None and self._load_cache[0] == cache_key:
            return self._load_cache[1]
        texts = load_local_jsonl_texts(
            files, text_field=self.text_field, max_samples=self.max_samples
        )
        self._load_cache = (cache_key, texts)
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
                    "NO-SAMPLES: local_jsonl produced no samples; check file/path/data_files"
                ),
            )
        total = min(len(texts), int(preview_n) + int(final_n))
        combined_window = self._simple_tokenize(
            texts[:total],
            tokenizer,
            seq_len,
            list(range(total)),
        )
        return split_window_by_index(combined_window, split_index=preview_n)

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
            "candidate_limit": len(texts),
        }


class LocalJSONLPairsProvider:
    name = "local_jsonl_pairs"

    def __init__(
        self,
        file: str | None = None,
        path: str | None = None,
        data_files: str | list[str] | None = None,
        src_field: str = "source",
        tgt_field: str = "target",
        max_samples: int = 2000,
    ) -> None:
        self.file = file
        self.path = path
        self.data_files = data_files
        self.src_field = src_field or "source"
        self.tgt_field = tgt_field or "target"
        self.max_samples = int(max_samples)
        self.last_preview_labels: list[list[int]] | None = None
        self.last_final_labels: list[list[int]] | None = None
        self._pairs_cache: tuple[tuple[Any, ...], list[tuple[str, str]]] | None = None

    def _resolve_files(self) -> list[Path]:
        return resolve_local_jsonl_files(
            file=self.file, path=self.path, data_files=self.data_files
        )

    def _load_pairs(self) -> list[tuple[str, str]]:
        files = self._resolve_files()
        cache_key = local_jsonl_cache_key(
            files,
            field_names=(self.src_field, self.tgt_field),
            max_samples=self.max_samples,
        )
        if self._pairs_cache is not None and self._pairs_cache[0] == cache_key:
            return self._pairs_cache[1]
        pairs = load_local_jsonl_pairs(
            files,
            src_field=self.src_field,
            tgt_field=self.tgt_field,
            max_samples=self.max_samples,
        )
        self._pairs_cache = (cache_key, pairs)
        return pairs

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
        pairs = self._load_pairs()
        if not pairs:
            raise ValueError(
                "local_jsonl_pairs produced no pairs; check src_field/tgt_field and files"
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
        pairs = self._load_pairs()
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


class TextLMProvider(EvaluationProvider):
    """Deterministic synthetic text LM provider for tests and smokes."""

    def __init__(self, **kwargs: Any) -> None:
        self._task = str(kwargs.get("task", "causal")).lower()
        self._n = int(kwargs.get("n", 16))
        self._seq_len = int(kwargs.get("seq_len", 8))
        self._mask_prob = float(kwargs.get("mask_prob", 0.15))
        self._pad_id = int(kwargs.get("pad_id", 0))
        self._eos_id = int(kwargs.get("eos_id", 2))
        self._ids: list[str] = []
        if self._task not in {"causal", "mlm"}:
            raise ValueError("task must be 'causal' or 'mlm'")
        if self._n < 0:
            raise ValueError("n must be non-negative")
        if self._seq_len < 3:
            raise ValueError("seq_len must be at least 3")
        if not math.isfinite(self._mask_prob) or not 0.0 <= self._mask_prob <= 1.0:
            raise ValueError("mask_prob must be a finite value in [0, 1]")

    def pairing_schedule(self) -> list[str]:
        return (
            sorted(self._ids) if self._ids else [f"ex{i:04d}" for i in range(self._n)]
        )

    def digest(self) -> dict[str, Any]:
        return {"provider": "text_lm", "version": 1, "task": self._task}

    def _gen_example(self, idx: int, *, seed: int) -> dict[str, Any]:
        import random

        rng = random.Random((seed + 31) ^ (idx * 131))
        real_len = max(3, self._seq_len - (idx % 3))
        ids = [1 + (rng.randint(0, 19)) for _ in range(real_len - 1)] + [self._eos_id]
        if real_len < self._seq_len:
            ids = ids + [self._pad_id] * (self._seq_len - real_len)
        attn = [1 if t != self._pad_id else 0 for t in ids]
        ex_id = f"ex{idx:04d}"

        labels: list[int] | None = None
        weights = sum(attn)
        if self._task == "mlm":
            labels = [-100] * len(ids)
            masked = 0
            for pos, (tok, m) in enumerate(zip(ids, attn, strict=False)):
                if not m or tok in (self._pad_id, self._eos_id):
                    continue
                rng2 = random.Random((seed + idx * 17 + pos * 13) & 0x7FFFFFFF)
                if rng2.random() < self._mask_prob:
                    labels[pos] = tok
                    masked += 1
            if masked == 0:
                for pos, (tok, m) in enumerate(zip(ids, attn, strict=False)):
                    if m and tok not in (self._pad_id, self._eos_id):
                        labels[pos] = tok
                        masked = 1
                        break
            weights = masked

        return {
            "ids": ex_id,
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels if labels is not None else [],
            "weights": int(weights),
        }

    def batches(self, *, seed: int, batch_size: int) -> Iterable[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch: dict[str, Any] = {
            "ids": [],
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "weights": [],
        }
        self._ids = []
        for i in range(self._n):
            ex = self._gen_example(i, seed=seed)
            self._ids.append(ex["ids"])
            for k in ("ids", "input_ids", "attention_mask", "labels", "weights"):
                batch[k].append(ex[k])
            if len(batch["ids"]) >= batch_size:
                yield batch
                batch = {
                    "ids": [],
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "weights": [],
                }
        if batch["ids"]:
            yield batch


class Seq2SeqProvider(EvaluationProvider):
    """Deterministic synthetic seq2seq provider for tests and smokes."""

    name = "seq2seq"

    def __init__(self, **kwargs: Any) -> None:
        self._n = int(kwargs.get("n", 12))
        self._src_len = int(kwargs.get("src_len", 6))
        self._tgt_len = int(kwargs.get("tgt_len", 7))
        self._pad_id = int(kwargs.get("pad_id", 0))
        self._bos_id = int(kwargs.get("bos_id", 1))
        self._eos_id = int(kwargs.get("eos_id", 2))
        self._ids: list[str] = []
        self.last_preview_labels: list[list[int]] | None = None
        self.last_final_labels: list[list[int]] | None = None

    def pairing_schedule(self) -> list[str]:
        return (
            sorted(self._ids) if self._ids else [f"ex{i:04d}" for i in range(self._n)]
        )

    def digest(self) -> dict[str, Any]:
        return {
            "provider": "seq2seq",
            "version": 1,
            "pad_id": self._pad_id,
            "eos_id": self._eos_id,
            "bos_id": self._bos_id,
        }

    def _gen_example(self, idx: int, *, seed: int) -> dict[str, Any]:
        import random

        rng = random.Random((seed + 17) ^ (idx * 97))
        src_real = max(3, self._src_len - (idx % 2))
        src_ids = (
            [self._bos_id]
            + [1 + rng.randint(0, 19) for _ in range(src_real - 2)]
            + [self._eos_id]
        )
        if src_real < self._src_len:
            src_ids += [self._pad_id] * (self._src_len - src_real)
        src_mask = [1 if t != self._pad_id else 0 for t in src_ids]

        tgt_real = max(3, self._tgt_len - (idx % 3))
        tgt_ids = [1 + rng.randint(0, 19) for _ in range(tgt_real - 1)] + [self._eos_id]
        if tgt_real < self._tgt_len:
            tgt_ids += [self._pad_id] * (self._tgt_len - tgt_real)
        tgt_mask = [1 if t != self._pad_id else 0 for t in tgt_ids]

        ex_id = f"ex{idx:04d}"
        weights = sum(1 for _t, m in zip(tgt_ids, tgt_mask, strict=False) if m)
        return {
            "ids": ex_id,
            "src_ids": src_ids,
            "src_mask": src_mask,
            "tgt_ids": tgt_ids,
            "tgt_mask": tgt_mask,
            "weights": int(weights),
        }

    def batches(self, *, seed: int, batch_size: int) -> Iterable[dict[str, Any]]:
        assert batch_size > 0
        batch = {
            "ids": [],
            "src_ids": [],
            "src_mask": [],
            "tgt_ids": [],
            "tgt_mask": [],
            "weights": [],
        }
        self._ids = []
        for i in range(self._n):
            ex = self._gen_example(i, seed=seed)
            self._ids.append(ex["ids"])
            for k in ("ids", "src_ids", "src_mask", "tgt_ids", "tgt_mask", "weights"):
                batch[k].append(ex[k])
            if len(batch["ids"]) >= batch_size:
                yield batch
                batch = {
                    "ids": [],
                    "src_ids": [],
                    "src_mask": [],
                    "tgt_ids": [],
                    "tgt_mask": [],
                    "weights": [],
                }
        if batch["ids"]:
            yield batch

    def load(
        self, split: str = "validation", **kwargs: Any
    ) -> list[str]:  # pragma: no cover - seq2seq does not expose raw-text windows
        _ = split, kwargs
        return []

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
        _ = stride, split
        total = max(1, int(preview_n) + int(final_n))
        if self._n < total:
            self._n = total
        batches = list(self.batches(seed=seed, batch_size=total))
        if not batches:
            raise ValueError("seq2seq provider produced no examples")
        batch = batches[0]
        src_ids_list = [list(x) for x in batch.get("src_ids", [])][:total]
        src_mask_list = [list(x) for x in batch.get("src_mask", [])][:total]
        tgt_ids_list = [list(x) for x in batch.get("tgt_ids", [])][:total]
        pad_id = getattr(tokenizer, "pad_token_id", self._pad_id)

        def _pad(seq: list[int], *, fill: int) -> list[int]:
            if len(seq) < seq_len:
                return (seq + [fill] * (seq_len - len(seq)))[:seq_len]
            return seq[:seq_len]

        input_ids = [_pad(seq, fill=pad_id) for seq in src_ids_list]
        attention_masks: list[list[int]] = []
        for index, seq in enumerate(input_ids):
            if index < len(src_mask_list) and len(src_mask_list[index]) == len(
                src_ids_list[index]
            ):
                attention_masks.append(
                    [int(v) for v in _pad(list(src_mask_list[index]), fill=0)]
                )
            else:
                attention_masks.append([1 if token != pad_id else 0 for token in seq])

        preview_window, final_window = split_window_by_index(
            EvaluationWindow(
                input_ids=input_ids,
                attention_masks=attention_masks,
                indices=list(range(len(input_ids))),
            ),
            split_index=preview_n,
        )
        self.last_preview_labels = [
            _pad(list(seq), fill=-100) for seq in tgt_ids_list[:preview_n]
        ]
        self.last_final_labels = [
            _pad(list(seq), fill=-100)
            for seq in tgt_ids_list[preview_n : preview_n + final_n]
        ]
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
        _ = tokenizer, split, target_total, fast_mode
        return {
            "total_tokens": int(self._n * seq_len),
            "available_nonoverlap": int(self._n),
            "available_unique": int(self._n),
            "dedupe_rate": 0.0,
            "stride": int(stride),
            "seq_len": int(seq_len),
            "candidate_unique": int(self._n),
            "candidate_limit": int(self._n),
            "tokens_available": int(self._n * seq_len),
            "examples_available": int(self._n),
        }


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_answers(obj: dict[str, Any]) -> list[str]:
    answers = obj.get("answers")
    if isinstance(answers, list):
        values = [str(answer).strip() for answer in answers if str(answer).strip()]
        if values:
            return values
    answer = obj.get("answer")
    if isinstance(answer, str) and answer.strip():
        return [answer.strip()]
    raise _DataErr(
        code="E306",
        message="NO-SAMPLES: vision_text record is missing answer/answers",
    )


def _resolve_image_path(image_path: str, *, base_dir: Path) -> Path:
    candidate = Path(image_path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    if not candidate.exists() or not candidate.is_file():
        raise _DataErr(
            code="E306",
            message=f"NO-SAMPLES: vision_text image file is missing ({candidate})",
        )
    return candidate


class VisionTextProvider(EvaluationProvider):
    name = "vision_text"

    def __init__(
        self,
        *,
        file: str | None = None,
        path: str | None = None,
        data_files: str | list[str] | None = None,
        max_samples: int = 0,
        items: list[dict[str, Any]] | None = None,
        transform_pipeline: str = "",
        seed: int | None = None,
    ) -> None:
        self.file = file
        self.path = path
        self.data_files = data_files
        self.max_samples = int(max_samples or 0)
        self._transform_pipeline = str(transform_pipeline or "")
        self._seed = int(seed) if seed is not None else None
        self._items_override = list(items or [])
        self._examples_cache: list[dict[str, Any]] | None = None

    def available_splits(self) -> list[str]:
        return ["validation"]

    def _resolve_files(self) -> list[Path]:
        if self._items_override:
            return []
        return resolve_local_jsonl_files(
            file=self.file,
            path=self.path,
            data_files=self.data_files,
        )

    def _load_examples(self) -> list[dict[str, Any]]:
        if self._examples_cache is not None:
            return list(self._examples_cache)

        examples: list[dict[str, Any]] = []
        if self._items_override:
            for index, raw in enumerate(self._items_override, start=1):
                if not isinstance(raw, dict):
                    continue
                rec_id = str(raw.get("id") or f"memory:{index}")
                prompt = str(raw.get("prompt") or "")
                answers = _normalize_answers(raw)
                image_bytes = raw.get("image_bytes")
                if isinstance(image_bytes, bytearray):
                    image_bytes = bytes(image_bytes)
                if not isinstance(image_bytes, bytes):
                    image_bytes = b""
                examples.append(
                    {
                        "id": rec_id,
                        "image_path": str(raw.get("image_path") or ""),
                        "prompt": prompt,
                        "answer": answers[0],
                        "answers": answers,
                        "image_sha256": _sha256_hex(image_bytes),
                        "prompt_sha256": _sha256_hex(prompt.encode("utf-8")),
                        "answer_sha256": _sha256_hex(
                            json.dumps(answers, ensure_ascii=True).encode("utf-8")
                        ),
                    }
                )
            self._examples_cache = examples
            return list(examples)

        files = self._resolve_files()
        if not files:
            raise _DataErr(
                code="E306",
                message=(
                    "NO-SAMPLES: vision_text produced no samples; check "
                    "file/path/data_files"
                ),
            )

        for file_path in files:
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as exc:
                raise _DataErr(
                    code="E306",
                    message=f"NO-SAMPLES: failed to read vision_text manifest ({exc})",
                ) from exc
            for line_no, raw_line in enumerate(lines, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise _DataErr(
                        code="E306",
                        message=(
                            "NO-SAMPLES: failed to parse vision_text manifest "
                            f"{file_path}:{line_no} ({exc})"
                        ),
                    ) from exc
                if not isinstance(obj, dict):
                    continue
                prompt = obj.get("prompt")
                image_path = obj.get("image_path")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise _DataErr(
                        code="E306",
                        message=(
                            "NO-SAMPLES: vision_text record is missing prompt "
                            f"({file_path}:{line_no})"
                        ),
                    )
                if not isinstance(image_path, str) or not image_path.strip():
                    raise _DataErr(
                        code="E306",
                        message=(
                            "NO-SAMPLES: vision_text record is missing image_path "
                            f"({file_path}:{line_no})"
                        ),
                    )
                answers = _normalize_answers(obj)
                resolved_image = _resolve_image_path(
                    image_path,
                    base_dir=file_path.parent,
                )
                image_bytes = resolved_image.read_bytes()
                rec_id = str(
                    obj.get("id") or f"{file_path.name}:{line_no}:{resolved_image.name}"
                )
                examples.append(
                    {
                        "id": rec_id,
                        "image_path": str(resolved_image),
                        "prompt": prompt.strip(),
                        "answer": answers[0],
                        "answers": answers,
                        "image_sha256": _sha256_hex(image_bytes),
                        "prompt_sha256": _sha256_hex(prompt.strip().encode("utf-8")),
                        "answer_sha256": _sha256_hex(
                            json.dumps(answers, ensure_ascii=True).encode("utf-8")
                        ),
                        "source_file": str(file_path),
                        "source_line": line_no,
                    }
                )
                if self.max_samples > 0 and len(examples) >= self.max_samples:
                    self._examples_cache = examples
                    return list(examples)

        if not examples:
            raise _DataErr(
                code="E306",
                message=(
                    "NO-SAMPLES: vision_text produced no samples; check "
                    "file/path/data_files"
                ),
            )

        self._examples_cache = examples
        return list(examples)

    def examples(
        self, split: str = "validation", **kwargs: Any
    ) -> list[dict[str, Any]]:
        del split, kwargs
        return self._load_examples()

    def pairing_schedule(self) -> list[str]:
        return sorted(str(item["id"]) for item in self._load_examples())

    def digest(self) -> dict[str, Any]:
        examples = self._load_examples()
        by_id = sorted(examples, key=lambda item: str(item["id"]))
        ids_sha256 = _sha256_hex("".join(item["id"] for item in by_id).encode("utf-8"))
        images_sha256 = _sha256_hex(
            "".join(item["image_sha256"] for item in by_id).encode("utf-8")
        )
        prompts_sha256 = _sha256_hex(
            "".join(item["prompt_sha256"] for item in by_id).encode("utf-8")
        )
        answers_sha256 = _sha256_hex(
            "".join(item["answer_sha256"] for item in by_id).encode("utf-8")
        )
        digest: dict[str, Any] = {
            "provider": "vision_text",
            "version": 2,
            "ids_sha256": ids_sha256,
            "images_sha256": images_sha256,
            "prompts_sha256": prompts_sha256,
            "answers_sha256": answers_sha256,
            "transform_pipeline": self._transform_pipeline,
        }
        if self._seed is not None:
            digest["seed"] = int(self._seed)
        return digest

    def batches(self, *, seed: int, batch_size: int) -> Iterable[dict[str, Any]]:
        del seed
        size = max(int(batch_size or 1), 1)
        examples = self._load_examples()
        for index in range(0, len(examples), size):
            chunk = examples[index : index + size]
            if len(chunk) == 1:
                yield dict(chunk[0])
            else:
                yield {"records": [dict(item) for item in chunk]}


_PROVIDERS: dict[str, type[DatasetProvider]] = {
    "wikitext2": WikiText2Provider,
    "synthetic": SyntheticProvider,
    "hf_text": HFTextProvider,
    "local_jsonl": LocalJSONLProvider,
    "seq2seq": Seq2SeqProvider,
    "hf_seq2seq": HFSeq2SeqProvider,
    "local_jsonl_pairs": LocalJSONLPairsProvider,
    "vision_text": VisionTextProvider,
}


def get_provider(name: str, **kwargs: Any) -> DatasetProvider:
    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys())
        raise _ValErr(
            code="E308",
            message="PROVIDER-NOT-FOUND: unknown dataset provider",
            details={"provider": name, "available": available},
        )

    provider_class = _PROVIDERS[name]
    init_kwargs = dict(kwargs)
    init_kwargs.pop("emit", None)
    return cast(DatasetProvider, provider_class(**init_kwargs))


def list_providers() -> list[str]:
    return list(_PROVIDERS.keys())


__all__ = [
    "DatasetProvider",
    "EvaluationWindow",
    "HFTextProvider",
    "LocalJSONLProvider",
    "LocalJSONLPairsProvider",
    "Seq2SeqProvider",
    "SyntheticProvider",
    "TextLMProvider",
    "VisionTextProvider",
    "WikiText2Provider",
    "compute_window_hash",
    "get_provider",
    "list_providers",
]
