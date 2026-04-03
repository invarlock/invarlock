from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invarlock.adapters.hf_multimodal import HF_Multimodal_Adapter
from invarlock.core.exceptions import DependencyError, ModelLoadError


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

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):  # noqa: ANN001
        del tokenize
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
        self, messages, tokenize=False, add_generation_prompt=False
    ):  # noqa: ANN001
        del tokenize, add_generation_prompt
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


def test_hf_multimodal_prepare_model_inputs_masks_prompt_tokens(tmp_path: Path) -> None:
    adapter, _processor = _build_adapter()
    image_path = tmp_path / "demo.ppm"
    _write_ppm(image_path)

    prepared = adapter.prepare_model_inputs(
        {
            "id": "ex-1",
            "image_path": str(image_path),
            "prompt": "what is shown?",
            "answers": ["cat"],
            "seq_len": 32,
        },
        device="cpu",
        include_labels=True,
    )

    assert prepared["_example_id"] == "ex-1"
    assert prepared["_reference_answers"] == ["cat"]
    assert prepared["_processor_sha256"] == adapter.processor_digest
    assert prepared["_decode_prompt_length"] == 3
    assert prepared["_answer_token_count"] == 2
    assert prepared["_max_new_tokens"] == 16
    assert prepared["labels"].tolist() == [[-100, -100, -100, 14, 15]]


def test_hf_multimodal_prepare_generation_inputs_and_decode(tmp_path: Path) -> None:
    adapter, processor = _build_adapter()
    image_path = tmp_path / "demo.ppm"
    _write_ppm(image_path)

    prepared = adapter.prepare_generation_inputs(
        {
            "example_id": "ex-2",
            "image_path": str(image_path),
            "prompt": "describe the image",
            "answer": "cat",
            "seq_len": 12,
        },
        device="cpu",
    )

    assert "labels" not in prepared
    assert prepared["_example_id"] == "ex-2"
    assert prepared["_answer_token_count"] == 0

    decoded = adapter.decode_generated(
        torch.tensor([[1, 2, 21, 22]], dtype=torch.long),
        {"_decode_prompt_length": 2},
    )

    assert processor.decoded_inputs == [[21, 22]]
    assert decoded == ["21 22"]


def test_hf_multimodal_prepare_model_inputs_retries_without_truncation(
    tmp_path: Path,
) -> None:
    adapter = HF_Multimodal_Adapter()
    processor = _RetryingProcessor()
    adapter._processor = processor
    adapter._processor_digest = adapter._compute_processor_digest(processor)
    adapter._last_model_id = "fake/model"
    image_path = tmp_path / "demo.ppm"
    _write_ppm(image_path)

    prepared = adapter.prepare_model_inputs(
        {
            "id": "ex-retry",
            "image_path": str(image_path),
            "prompt": "what is shown?",
            "answers": ["cat"],
            "seq_len": 8,
        },
        device="cpu",
        include_labels=True,
    )

    assert prepared["_decode_prompt_length"] == 3
    assert prepared["_answer_token_count"] == 2
    assert prepared["labels"].tolist() == [[-100, -100, -100, 14, 15]]
    assert processor.calls == [
        ("USER:what is shown?\nASSISTANT:", True, 8),
        ("USER:what is shown?\nASSISTANT:", False, None),
        ("USER:what is shown?\nASSISTANT:cat", True, 8),
        ("USER:what is shown?\nASSISTANT:cat", False, None),
    ]


def test_hf_multimodal_load_model_uses_resolved_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HF_Multimodal_Adapter()
    strategy = SimpleNamespace(
        strategy="direct_submodule",
        loader_label="direct-loader",
        loader="direct-loader",
    )
    auto_strategy = SimpleNamespace(
        strategy="auto",
        loader_label="auto-loader",
        loader="auto-loader",
    )

    calls: list[tuple[str, str, dict[str, object]]] = []
    monkeypatch.setattr(
        "invarlock.adapters.hf_multimodal.resolve_core_loader_strategy",
        lambda **kwargs: auto_strategy
        if kwargs.get("allow_direct_submodule") is False
        else strategy,
    )
    monkeypatch.setattr(
        adapter,
        "_load_pretrained_model",
        lambda loader, model_id, **kwargs: calls.append((loader, model_id, kwargs))
        or {"loader": loader},
    )
    monkeypatch.setattr(
        adapter,
        "_safe_to_device",
        lambda model, device: {"model": model, "device": device},
    )

    loaded = adapter.load_model("fake/model", device="cpu", trust_remote_code=False)

    assert loaded == {"model": {"loader": "direct-loader"}, "device": "cpu"}
    assert adapter._last_model_id == "fake/model"
    assert adapter._last_loader_strategy == "direct_submodule"
    assert adapter._last_loader_label == "direct-loader"
    assert calls == [("direct-loader", "fake/model", {"trust_remote_code": False})]


def test_hf_multimodal_load_model_falls_back_to_auto_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HF_Multimodal_Adapter()
    strategy = SimpleNamespace(
        strategy="direct_submodule",
        loader_label="direct-loader",
        loader="direct-loader",
    )
    auto_strategy = SimpleNamespace(
        strategy="auto",
        loader_label="auto-loader",
        loader="auto-loader",
    )

    monkeypatch.setattr(
        "invarlock.adapters.hf_multimodal.resolve_core_loader_strategy",
        lambda **kwargs: auto_strategy
        if kwargs.get("allow_direct_submodule") is False
        else strategy,
    )
    load_calls: list[str] = []

    def _load(loader: str, model_id: str, **kwargs: object) -> object:
        del model_id, kwargs
        load_calls.append(loader)
        if loader == "direct-loader":
            raise RuntimeError("no direct multimodal class")
        return {"loader": loader}

    monkeypatch.setattr(adapter, "_load_pretrained_model", _load)
    monkeypatch.setattr(
        adapter, "_safe_to_device", lambda model, device: (model, device)
    )

    loaded = adapter.load_model("fake/model", device="cuda")

    assert loaded == ({"loader": "auto-loader"}, "cuda")
    assert load_calls == ["direct-loader", "auto-loader"]
    assert adapter._last_loader_strategy == "auto"
    assert adapter._last_loader_label == "auto-loader"


def test_hf_multimodal_load_model_wraps_auto_strategy_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HF_Multimodal_Adapter()
    strategy = SimpleNamespace(
        strategy="auto",
        loader_label="auto-loader",
        loader="auto-loader",
    )

    monkeypatch.setattr(
        "invarlock.adapters.hf_multimodal.resolve_core_loader_strategy",
        lambda **kwargs: strategy,
    )
    monkeypatch.setattr(
        adapter,
        "_load_pretrained_model",
        lambda loader, model_id, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(ModelLoadError, match="MODEL-LOAD-FAILED: auto-loader"):
        adapter.load_model("fake/model")


def test_hf_multimodal_load_model_wraps_missing_transformers_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HF_Multimodal_Adapter()
    monkeypatch.setattr(
        "invarlock.adapters.hf_multimodal.resolve_core_loader_strategy",
        lambda **kwargs: (_ for _ in ()).throw(ModuleNotFoundError("missing")),
    )

    with pytest.raises(DependencyError, match="DEPENDENCY-MISSING: transformers"):
        adapter.load_model("fake/model")


@pytest.mark.parametrize("shape", ["language_model", "nested_language_model", "plain"])
def test_hf_multimodal_unwrap_prefers_decoder_language_model(
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
) -> None:
    adapter = HF_Multimodal_Adapter()
    seen: list[object] = []
    monkeypatch.setattr(
        "invarlock.adapters.hf_causal.HF_Causal_Adapter._unwrap",
        lambda self, model: seen.append(model) or ("model", "base", "layers"),
    )

    language_model = object()
    if shape == "language_model":
        model = SimpleNamespace(language_model=language_model)
        expected = language_model
    elif shape == "nested_language_model":
        model = SimpleNamespace(model=SimpleNamespace(language_model=language_model))
        expected = language_model
    else:
        model = SimpleNamespace()
        expected = model

    assert adapter._unwrap(model) == ("model", "base", "layers")
    assert seen == [expected]


def test_hf_multimodal_require_processor_raises_before_load_model() -> None:
    adapter = HF_Multimodal_Adapter()

    with pytest.raises(RuntimeError, match="Processor unavailable before load_model"):
        adapter._require_processor()


def test_hf_multimodal_require_processor_imports_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HF_Multimodal_Adapter()
    adapter._last_model_id = "fake/model"
    processor = _FakeProcessor()

    class _AutoProcessor:
        calls: list[str] = []

        @classmethod
        def from_pretrained(cls, model_id: str):  # noqa: ANN102
            cls.calls.append(model_id)
            return processor

    fake_transformers = SimpleNamespace(AutoProcessor=_AutoProcessor)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    loaded = adapter._require_processor()

    assert loaded is processor
    assert _AutoProcessor.calls == ["fake/model"]
    assert adapter.processor_digest == adapter._compute_processor_digest(processor)
    assert adapter._require_processor() is processor
    assert _AutoProcessor.calls == ["fake/model"]


def test_hf_multimodal_require_processor_wraps_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HF_Multimodal_Adapter()
    adapter._last_model_id = "fake/model"
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "transformers":
            raise ModuleNotFoundError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(DependencyError, match="DEPENDENCY-MISSING: transformers"):
        adapter._require_processor()


def test_hf_multimodal_processor_digest_returns_none_when_processor_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HF_Multimodal_Adapter()
    monkeypatch.setattr(
        adapter,
        "_require_processor",
        lambda: (_ for _ in ()).throw(RuntimeError("not ready")),
    )

    assert adapter.processor_digest is None


def test_hf_multimodal_compute_digest_and_helpers_cover_fallback_paths(
    tmp_path: Path,
) -> None:
    adapter = HF_Multimodal_Adapter()
    image_path = tmp_path / "demo.ppm"
    _write_ppm(image_path)

    digest = adapter._compute_processor_digest(SimpleNamespace(name_or_path="minimal"))
    assert isinstance(digest, str)
    assert adapter._reference_answers({"answer": "  cat  "}) == ["cat"]
    assert adapter._reference_answers({"answers": [" ", ""], "answer": "dog"}) == [
        "dog"
    ]
    assert adapter._reference_answers({}) == []
    assert (
        adapter._processor_text(object(), prompt="describe", answer=None) == "describe"
    )
    assert (
        adapter._processor_text(object(), prompt="describe", answer="cat")
        == "describe\ncat"
    )

    moved = adapter._move_to_device(
        {"tensor": torch.tensor([1]), "label": "cat"}, "cpu"
    )
    assert moved["tensor"].device.type == "cpu"
    assert moved["label"] == "cat"

    opened = adapter._open_image({"image_path": str(image_path)})
    assert opened.mode == "RGB"

    with pytest.raises(ValueError, match="missing image_path"):
        adapter._open_image({})


def test_hf_multimodal_prepare_model_inputs_handles_non_tensor_labels(
    tmp_path: Path,
) -> None:
    adapter = HF_Multimodal_Adapter()
    adapter._processor = _ProcessorWithoutTemplate()
    adapter._processor_digest = "digest"
    adapter._last_model_id = "fake/model"
    image_path = tmp_path / "demo.ppm"
    _write_ppm(image_path)

    prepared = adapter.prepare_model_inputs(
        {
            "example_id": "ex-3",
            "image_path": str(image_path),
            "prompt": "describe",
            "answer": "cat",
        },
        device="cpu",
        include_labels=True,
    )

    assert prepared["_decode_prompt_length"] == 0
    assert prepared["_answer_token_count"] == 0
    assert prepared["_max_new_tokens"] == 16
    assert "labels" not in prepared


def test_hf_multimodal_processor_call_raises_when_retry_not_allowed() -> None:
    adapter = HF_Multimodal_Adapter()
    processor = Mock(side_effect=ValueError("plain processor failure"))

    with pytest.raises(ValueError, match="plain processor failure"):
        adapter._processor_call(
            processor,
            text="USER:describe\nASSISTANT:",
            image=None,
            seq_len=8,
        )


def test_hf_multimodal_prepare_model_inputs_recomputes_prompt_after_answer_retry(
    tmp_path: Path,
) -> None:
    adapter = HF_Multimodal_Adapter()
    processor = _ProcessorPromptRetry()
    adapter._processor = processor
    adapter._processor_digest = adapter._compute_processor_digest(processor)
    adapter._last_model_id = "fake/model"
    image_path = tmp_path / "demo.ppm"
    _write_ppm(image_path)

    prepared = adapter.prepare_model_inputs(
        {
            "id": "ex-prompt-retry",
            "image_path": str(image_path),
            "prompt": "what is shown?",
            "answers": ["cat"],
            "seq_len": 8,
        },
        device="cpu",
        include_labels=True,
    )

    assert prepared["_decode_prompt_length"] == 3
    assert prepared["labels"].tolist() == [[-100, -100, -100, 14, 15]]
    assert processor.calls == [
        ("USER:what is shown?\nASSISTANT:", True, 8),
        ("USER:what is shown?\nASSISTANT:cat", True, 8),
        ("USER:what is shown?\nASSISTANT:cat", False, None),
        ("USER:what is shown?\nASSISTANT:", False, None),
    ]


def test_hf_multimodal_prepare_model_inputs_skips_prompt_mask_when_prompt_is_empty(
    tmp_path: Path,
) -> None:
    adapter = HF_Multimodal_Adapter()
    processor = _ProcessorZeroPromptLength()
    adapter._processor = processor
    adapter._processor_digest = adapter._compute_processor_digest(processor)
    adapter._last_model_id = "fake/model"
    image_path = tmp_path / "demo.ppm"
    _write_ppm(image_path)

    prepared = adapter.prepare_model_inputs(
        {
            "image_path": str(image_path),
            "prompt": "describe",
            "answer": "cat",
        },
        device="cpu",
        include_labels=True,
    )

    assert prepared["_decode_prompt_length"] == 0
    assert prepared["labels"].tolist() == [[21, 22]]
    assert prepared["_answer_token_count"] == 2


def test_hf_multimodal_decode_generated_uses_tokenizer_fallback_and_tensor_unsqueeze() -> (
    None
):
    adapter = HF_Multimodal_Adapter()
    tokenizer = _TokenizerOnlyDecoder()
    adapter._processor = SimpleNamespace(tokenizer=tokenizer)
    adapter._processor_digest = "digest"
    adapter._last_model_id = "fake/model"

    decoded = adapter.decode_generated(
        torch.tensor([7, 8, 9], dtype=torch.long),
        {"_decode_prompt_length": 0},
    )

    assert isinstance(tokenizer.decoded_inputs[0], torch.Tensor)
    assert tokenizer.decoded_inputs[0].shape == (1, 3)
    assert decoded == ["decoded:7 8 9"]


def test_hf_multimodal_decode_generated_accepts_non_tensor_ids() -> None:
    adapter = HF_Multimodal_Adapter()
    tokenizer = _TokenizerOnlyDecoder()
    adapter._processor = SimpleNamespace(tokenizer=tokenizer)
    adapter._processor_digest = "digest"
    adapter._last_model_id = "fake/model"

    decoded = adapter.decode_generated([[3, 4]], {"_decode_prompt_length": 0})

    assert tokenizer.decoded_inputs == [[[3, 4]]]
    assert decoded == ["decoded:3 4"]


def test_hf_multimodal_decode_generated_raises_without_batch_decoder() -> None:
    adapter = HF_Multimodal_Adapter()
    adapter._processor = _ProcessorWithoutDecoder()
    adapter._processor_digest = "digest"
    adapter._last_model_id = "fake/model"

    with pytest.raises(RuntimeError, match="does not expose batch_decode"):
        adapter.decode_generated([[1, 2]], {"_decode_prompt_length": 0})
