from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.adapters.hf_multimodal import HF_Multimodal_Adapter


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
        self.template_kwargs: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages,  # noqa: ANN001
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: object,
    ) -> str:
        del tokenize
        self.template_kwargs.append(dict(kwargs))
        prompt = messages[0]["content"][1]["text"]
        suffix = "\nASSISTANT:" if add_generation_prompt else ""
        return f"USER:{prompt}{suffix}"


def test_hf_multimodal_load_model_consumes_chat_template_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HF_Multimodal_Adapter()
    strategy = SimpleNamespace(
        strategy="auto",
        loader_label="auto-loader",
        loader="auto-loader",
    )

    calls: list[tuple[str, str, dict[str, object]]] = []
    monkeypatch.setattr(
        "invarlock.adapters.hf_multimodal.resolve_core_loader_strategy",
        lambda **kwargs: strategy,
    )
    monkeypatch.setattr(
        adapter,
        "_load_pretrained_model",
        lambda loader, model_id, **kwargs: (
            calls.append((loader, model_id, kwargs)) or {"loader": loader}
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_safe_to_device",
        lambda model, device: {"model": model, "device": device},
    )

    adapter.load_model(
        "fake/model",
        device="cpu",
        dtype="bfloat16",
        chat_template_kwargs={"enable_thinking": False},
    )

    assert adapter._chat_template_kwargs == {"enable_thinking": False}
    assert calls == [
        (
            "auto-loader",
            "fake/model",
            {"load_device": "cpu", "dtype": "bfloat16"},
        )
    ]


def test_hf_multimodal_load_model_rejects_non_mapping_chat_template_kwargs() -> None:
    adapter = HF_Multimodal_Adapter()

    with pytest.raises(
        ValueError, match="model.chat_template_kwargs must be a mapping"
    ):
        adapter.load_model("fake/model", chat_template_kwargs=["enable_thinking"])


def test_hf_multimodal_processor_text_forwards_chat_template_kwargs() -> None:
    adapter = HF_Multimodal_Adapter()
    processor = _FakeProcessor()
    adapter._chat_template_kwargs = {"enable_thinking": False}

    rendered = adapter._processor_text(processor, prompt="what is shown?")

    assert rendered == "USER:what is shown?\nASSISTANT:"
    assert processor.template_kwargs == [
        {"chat_template_kwargs": {"enable_thinking": False}}
    ]


def test_hf_multimodal_processor_digest_includes_chat_template_kwargs() -> None:
    adapter = HF_Multimodal_Adapter()
    processor = _FakeProcessor()
    default_digest = adapter._compute_processor_digest(processor)

    adapter._chat_template_kwargs = {"enable_thinking": False}

    assert adapter._compute_processor_digest(processor) != default_digest
