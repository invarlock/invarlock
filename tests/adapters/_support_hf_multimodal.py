from __future__ import annotations

from pathlib import Path

import torch

from invarlock.adapters.hf_multimodal import HF_Multimodal_Adapter


def _write_ppm(path: Path) -> None:
    path.write_text("P3\n1 1\n255\n255 0 0\n", encoding="utf-8")


class _FakeTokenizer:
    name_or_path = "fake-tokenizer"
    vocab_size = 321
    eos_token = "</s>"
    pad_token = "<pad>"


class _FakeImageProcessor:
    size = {"height": 224, "width": 224}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.25, 0.25, 0.25]


class _FakeProcessor:
    def __init__(self) -> None:
        self.name_or_path = "fake-processor"
        self.tokenizer = _FakeTokenizer()
        self.image_processor = _FakeImageProcessor()
        self.decoded_inputs: list[list[int]] = []
        self.template_kwargs: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):  # noqa: ANN001
        del tokenize
        self.template_kwargs.append(dict(kwargs))
        prompt = messages[0]["content"][1]["text"]
        if len(messages) > 1:
            answer = messages[1]["content"][0]["text"]
            return f"USER:{prompt}\nASSISTANT:{answer}"
        suffix = "\nASSISTANT:" if add_generation_prompt else ""
        return f"USER:{prompt}{suffix}"

    def __call__(self, *, text, images, return_tensors, truncation, max_length):  # noqa: ANN001
        del images, return_tensors, truncation, max_length
        if "ASSISTANT:cat" in text:
            input_ids = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
        else:
            input_ids = torch.tensor([[11, 12, 13]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def batch_decode(self, ids, skip_special_tokens=True):  # noqa: ANN001
        del skip_special_tokens
        rows = ids.tolist()
        self.decoded_inputs = rows
        return [" ".join(str(token) for token in row) for row in rows]


class _RetryingProcessor(_FakeProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, bool, int | None]] = []

    def __call__(self, *, text, images, return_tensors, truncation, max_length):  # noqa: ANN001
        del images, return_tensors
        self.calls.append((text, bool(truncation), max_length))
        if truncation:
            raise ValueError(
                "Mismatch in `image` token count between text and `input_ids`. "
                "Likely due to `truncation='max_length'`."
            )
        return super().__call__(
            text=text,
            images=None,
            return_tensors="pt",
            truncation=False,
            max_length=max_length,
        )


class _TokenizerOnlyDecoder:
    def __init__(self) -> None:
        self.decoded_inputs: list[object] = []

    def batch_decode(self, ids, skip_special_tokens=True):  # noqa: ANN001
        del skip_special_tokens
        self.decoded_inputs.append(ids)
        if isinstance(ids, torch.Tensor):
            rows = ids.tolist()
        else:
            rows = ids
        return ["decoded:" + " ".join(str(token) for token in row) for row in rows]


class _ProcessorWithoutTemplate:
    def __call__(self, *, text, images, return_tensors, truncation, max_length):  # noqa: ANN001
        del text, images, return_tensors, truncation, max_length
        return {"input_ids": ["not-a-tensor"], "attention_mask": None}


class _ProcessorPromptRetry(_FakeProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, bool, int | None]] = []

    def __call__(self, *, text, images, return_tensors, truncation, max_length):  # noqa: ANN001
        del images, return_tensors
        self.calls.append((text, bool(truncation), max_length))
        if "ASSISTANT:cat" in text and truncation:
            raise ValueError(
                "Mismatch in `image` token count between text and `input_ids`. "
                "Likely due to `truncation='max_length'`."
            )
        return super().__call__(
            text=text,
            images=None,
            return_tensors="pt",
            truncation=truncation,
            max_length=max_length,
        )


class _ProcessorZeroPromptLength:
    def __init__(self) -> None:
        self.name_or_path = "zero-prompt"
        self.tokenizer = _FakeTokenizer()
        self.image_processor = _FakeImageProcessor()

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):  # noqa: ANN001
        del tokenize, add_generation_prompt, kwargs
        prompt = messages[0]["content"][1]["text"]
        if len(messages) > 1:
            answer = messages[1]["content"][0]["text"]
            return f"{prompt}\n{answer}"
        return prompt

    def __call__(self, *, text, images, return_tensors, truncation, max_length):  # noqa: ANN001
        del images, return_tensors, truncation, max_length
        if "\n" in text:
            return {"input_ids": torch.tensor([[21, 22]], dtype=torch.long)}
        return {"input_ids": torch.zeros((1, 0), dtype=torch.long)}


class _ProcessorWithoutDecoder:
    tokenizer = object()


def _build_adapter() -> tuple[HF_Multimodal_Adapter, _FakeProcessor]:
    adapter = HF_Multimodal_Adapter()
    processor = _FakeProcessor()
    adapter._processor = processor
    adapter._processor_digest = adapter._compute_processor_digest(processor)
    adapter._last_model_id = "fake/model"
    return adapter, processor
