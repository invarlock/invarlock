"""
HuggingFace multimodal adapter (image-text to text).
====================================================

Phase 1 keeps the structural surface aligned with the causal adapter while
adding processor-aware batch preparation hooks for image-text evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from invarlock.core import INVARLOCK_CORE_ABI as CORE_ABI
from invarlock.core.error_utils import wrap_errors
from invarlock.core.exceptions import DependencyError, ModelLoadError

from .hf_causal import HF_Causal_Adapter
from .hf_loading import resolve_core_loader_strategy

INVARLOCK_CORE_ABI = CORE_ABI

_ALLOW_DIRECT_SUBMODULE = True
_PROCESSOR_DIGEST_ERRORS = (DependencyError, ModelLoadError, RuntimeError)


def _hash_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class HF_Multimodal_Adapter(HF_Causal_Adapter):
    name = "hf_multimodal"

    def __init__(self) -> None:
        super().__init__()
        self._processor: Any | None = None
        self._processor_digest: str | None = None
        self._last_model_id: str | None = None

    def load_model(self, model_id: str, device: str = "auto", **kwargs: Any) -> Any:
        self._last_model_id = str(model_id)
        try:
            with wrap_errors(
                DependencyError,
                "E203",
                "DEPENDENCY-MISSING: transformers",
                lambda e: {"dependency": "transformers"},
            ):
                strategy = resolve_core_loader_strategy(
                    task="multimodal",
                    model_id=model_id,
                    kwargs=kwargs,
                    allow_direct_submodule=_ALLOW_DIRECT_SUBMODULE,
                )
                auto_strategy = (
                    strategy
                    if strategy.strategy == "auto"
                    else resolve_core_loader_strategy(
                        task="multimodal",
                        model_id=model_id,
                        kwargs=kwargs,
                        allow_direct_submodule=False,
                    )
                )
            self._last_loader_strategy = strategy.strategy
            self._last_loader_label = strategy.loader_label

            try:
                with wrap_errors(
                    ModelLoadError,
                    "E201",
                    f"MODEL-LOAD-FAILED: {strategy.loader_label}",
                    lambda e: {"model_id": model_id},
                ):
                    model = self._load_pretrained_model(
                        strategy.loader,
                        model_id,
                        load_device=device,
                        **kwargs,
                    )
            except ModelLoadError:
                if strategy.strategy == "auto":
                    raise
                self._last_loader_strategy = auto_strategy.strategy
                self._last_loader_label = auto_strategy.loader_label
                with wrap_errors(
                    ModelLoadError,
                    "E201",
                    f"MODEL-LOAD-FAILED: {auto_strategy.loader_label}",
                    lambda e: {"model_id": model_id},
                ):
                    model = self._load_pretrained_model(
                        auto_strategy.loader,
                        model_id,
                        load_device=device,
                        **kwargs,
                    )

            return self._safe_to_device(model, device)
        except DependencyError:
            raise

    def _unwrap(self, model: Any) -> tuple[Any, Any, Any]:
        candidate = getattr(model, "language_model", None)
        if candidate is not None:
            return super()._unwrap(candidate)
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return super()._unwrap(model.model.language_model)
        return super()._unwrap(model)

    def _require_processor(self) -> Any:
        if self._processor is not None:
            return self._processor
        if not self._last_model_id:
            raise RuntimeError(
                "Processor unavailable before load_model(); explicit multimodal evaluation "
                "requires a loaded model identifier."
            )
        try:
            from transformers import AutoProcessor
        except ModuleNotFoundError as exc:
            raise DependencyError(
                code="E203",
                message="DEPENDENCY-MISSING: transformers",
                details={"dependency": "transformers"},
            ) from exc

        self._processor = AutoProcessor.from_pretrained(self._last_model_id)
        self._processor_digest = self._compute_processor_digest(self._processor)
        return self._processor

    def _compute_processor_digest(self, processor: Any) -> str:
        tokenizer = getattr(processor, "tokenizer", None)
        image_processor = getattr(processor, "image_processor", None)
        payload: dict[str, Any] = {
            "processor_class": processor.__class__.__name__,
            "processor_name": str(getattr(processor, "name_or_path", "") or ""),
        }
        if tokenizer is not None:
            payload["tokenizer"] = {
                "class": tokenizer.__class__.__name__,
                "name": str(getattr(tokenizer, "name_or_path", "") or ""),
                "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
                "eos_token": getattr(tokenizer, "eos_token", None),
                "pad_token": getattr(tokenizer, "pad_token", None),
            }
        if image_processor is not None:
            size = getattr(image_processor, "size", None)
            image_mean = getattr(image_processor, "image_mean", None)
            image_std = getattr(image_processor, "image_std", None)
            payload["image_processor"] = {
                "class": image_processor.__class__.__name__,
                "size": size,
                "image_mean": image_mean,
                "image_std": image_std,
            }
        return _hash_json(payload)

    @property
    def processor_digest(self) -> str | None:
        if self._processor_digest is None:
            try:
                if self._processor is not None:
                    self._processor_digest = self._compute_processor_digest(
                        self._processor
                    )
                else:
                    self._require_processor()
            except _PROCESSOR_DIGEST_ERRORS:
                return None
        return self._processor_digest

    def _open_image(self, batch: dict[str, Any]) -> Any:
        from PIL import Image

        image_path = batch.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("vision_text batch is missing image_path")
        path = Path(image_path).expanduser()
        with Image.open(path) as image:
            return image.convert("RGB")

    def _reference_answers(self, batch: dict[str, Any]) -> list[str]:
        answers = batch.get("answers")
        if isinstance(answers, list):
            values = [str(answer).strip() for answer in answers if str(answer).strip()]
            if values:
                return values
        answer = batch.get("answer")
        if isinstance(answer, str) and answer.strip():
            return [answer.strip()]
        return []

    def _chat_messages(
        self, *, prompt: str, answer: str | None = None
    ) -> list[dict[str, Any]]:
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if answer is not None:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            )
        return messages

    def _processor_text(
        self, processor: Any, *, prompt: str, answer: str | None = None
    ) -> str:
        if hasattr(processor, "apply_chat_template"):
            return processor.apply_chat_template(
                self._chat_messages(prompt=prompt, answer=answer),
                tokenize=False,
                add_generation_prompt=answer is None,
            )
        if answer is None:
            return prompt
        return f"{prompt}\n{answer}"

    def _should_retry_without_truncation(self, exc: ValueError) -> bool:
        message = str(exc).strip().lower()
        return "mismatch in `image` token count" in message and (
            "truncation='max_length'" in message or "max_length" in message
        )

    def _processor_call(
        self,
        processor: Any,
        *,
        text: str,
        image: Any,
        seq_len: int | None,
    ) -> tuple[dict[str, Any], bool]:
        kwargs = {
            "text": text,
            "images": image,
            "return_tensors": "pt",
            "truncation": bool(seq_len),
            "max_length": seq_len,
        }
        try:
            payload = processor(**kwargs)
            return dict(payload), bool(seq_len)
        except ValueError as exc:
            if not seq_len or not self._should_retry_without_truncation(exc):
                raise
        retry_payload = processor(
            text=text,
            images=image,
            return_tensors="pt",
            truncation=False,
            max_length=None,
        )
        return dict(retry_payload), False

    def _move_to_device(self, payload: dict[str, Any], device: Any) -> dict[str, Any]:
        prepared: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(device)
            else:
                prepared[key] = value
        return prepared

    def prepare_model_inputs(
        self, batch: dict[str, Any], device: Any, include_labels: bool
    ) -> dict[str, Any]:
        processor = self._require_processor()
        image = self._open_image(batch)
        prompt = str(batch.get("prompt", "") or "")
        answers = self._reference_answers(batch)
        answer = answers[0] if answers else ""
        seq_len = int(batch.get("seq_len", 0) or 0) or None

        prompt_text = self._processor_text(processor, prompt=prompt)
        prompt_payload, prompt_used_truncation = self._processor_call(
            processor,
            text=prompt_text,
            image=image,
            seq_len=seq_len,
        )
        prompt_ids = prompt_payload.get("input_ids")
        prompt_length = (
            int(prompt_ids.shape[-1]) if isinstance(prompt_ids, torch.Tensor) else 0
        )

        if include_labels:
            full_text = self._processor_text(processor, prompt=prompt, answer=answer)
            payload, labels_used_truncation = self._processor_call(
                processor,
                text=full_text,
                image=image,
                seq_len=seq_len,
            )
            if prompt_used_truncation and not labels_used_truncation:
                prompt_payload, _ = self._processor_call(
                    processor,
                    text=prompt_text,
                    image=image,
                    seq_len=None,
                )
                prompt_ids = prompt_payload.get("input_ids")
                prompt_length = (
                    int(prompt_ids.shape[-1])
                    if isinstance(prompt_ids, torch.Tensor)
                    else 0
                )
            labels = payload.get("input_ids")
            if isinstance(labels, torch.Tensor):
                labels = labels.clone()
                if prompt_length > 0:
                    labels[..., :prompt_length] = -100
                attention_mask = payload.get("attention_mask")
                if isinstance(attention_mask, torch.Tensor):
                    labels = labels.masked_fill(attention_mask == 0, -100)
                payload["labels"] = labels
        else:
            payload = prompt_payload

        prepared = self._move_to_device(dict(payload), device)
        prepared["_decode_prompt_length"] = prompt_length
        prepared["_example_id"] = str(batch.get("id") or batch.get("example_id") or "")
        prepared["_reference_answers"] = answers
        prepared["_processor_sha256"] = self.processor_digest
        prepared["_max_new_tokens"] = max(
            16,
            min(int(seq_len or 64), max((len(answer.split()) + 8), 16)),
        )
        if include_labels and isinstance(prepared.get("labels"), torch.Tensor):
            prepared["_answer_token_count"] = int(
                (prepared["labels"] != -100).sum().item()
            )
        else:
            prepared["_answer_token_count"] = 0
        return prepared

    def prepare_generation_inputs(
        self, batch: dict[str, Any], device: Any
    ) -> dict[str, Any]:
        return self.prepare_model_inputs(batch, device, include_labels=False)

    def decode_generated(
        self, generated_ids: Any, prepared_batch: dict[str, Any]
    ) -> list[str]:
        processor = self._require_processor()
        tokenizer = getattr(processor, "tokenizer", None)
        decoder = processor if hasattr(processor, "batch_decode") else tokenizer
        if decoder is None or not hasattr(decoder, "batch_decode"):
            raise RuntimeError("Processor does not expose batch_decode")
        prompt_length = int(prepared_batch.get("_decode_prompt_length", 0) or 0)
        if isinstance(generated_ids, torch.Tensor):
            ids = generated_ids.detach().cpu()
        else:
            ids = generated_ids
        if isinstance(ids, torch.Tensor) and ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if (
            isinstance(ids, torch.Tensor)
            and prompt_length > 0
            and ids.shape[-1] > prompt_length
        ):
            ids = ids[:, prompt_length:]
        outputs = decoder.batch_decode(ids, skip_special_tokens=True)
        return [str(output).strip() for output in outputs]
