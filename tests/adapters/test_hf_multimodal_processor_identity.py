from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invarlock.adapters.hf_multimodal import (
    HF_Multimodal_Adapter,
    _json_safe_processor_value,
    _strict_processor_digest,
    _strict_tokenizer_digest,
)
from tests.adapters._support_hf_multimodal import (
    _ProcessorPromptRetry,
    _ProcessorWithoutDecoder,
    _ProcessorWithoutTemplate,
    _ProcessorZeroPromptLength,
    _TokenizerOnlyDecoder,
    _write_ppm,
)


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


def test_hf_multimodal_processor_digest_computes_from_existing_processor() -> None:
    adapter = HF_Multimodal_Adapter()
    processor = SimpleNamespace(name_or_path="already-loaded")
    adapter._processor = processor
    adapter._processor_digest = None

    assert adapter.processor_digest == adapter._compute_processor_digest(processor)


def test_hf_multimodal_processor_identity_binds_tokenizer_processor_and_template() -> (
    None
):
    adapter = HF_Multimodal_Adapter()
    tokenizer = SimpleNamespace(
        name_or_path="public/model",
        vocab_size=42,
        eos_token="<eos>",
        pad_token="<pad>",
        special_tokens_map={"eos_token": "<eos>"},
        chat_template="{{ messages }}",
        get_vocab=lambda: {"<eos>": 0, "cat": 1},
        get_added_vocab=lambda: {},
        init_kwargs={"model_max_length": 2048},
    )
    processor = SimpleNamespace(
        name_or_path="public/model",
        tokenizer=tokenizer,
        image_processor=SimpleNamespace(
            size={"height": 224, "width": 224},
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.25, 0.25, 0.25],
        ),
        to_dict=lambda: {"image_size": 224, "do_normalize": True},
    )
    adapter._processor = processor
    adapter._processor_digest = adapter._compute_processor_digest(processor)

    identity = adapter.processor_identity

    assert identity is not None
    assert set(identity) == {
        "chat_template_sha256",
        "processor_sha256",
        "tokenizer_sha256",
    }
    assert all(
        value.startswith("sha256:") and len(value) == 71 for value in identity.values()
    )

    tokenizer.chat_template = "{{ messages | changed }}"
    changed_identity = adapter.processor_identity
    assert changed_identity is not None
    assert changed_identity["chat_template_sha256"] != identity["chat_template_sha256"]
    assert changed_identity["tokenizer_sha256"] == identity["tokenizer_sha256"]

    tokenizer.get_vocab = lambda: {"<eos>": 0, "dog": 1}
    vocab_changed_identity = adapter.processor_identity
    assert vocab_changed_identity is not None
    assert vocab_changed_identity["tokenizer_sha256"] != identity["tokenizer_sha256"]
    assert vocab_changed_identity["processor_sha256"] != identity["processor_sha256"]


def test_hf_multimodal_processor_identity_requires_chat_template() -> None:
    adapter = HF_Multimodal_Adapter()
    adapter._processor = SimpleNamespace(
        name_or_path="public/model",
        tokenizer=SimpleNamespace(
            name_or_path="public/model",
            vocab_size=42,
            special_tokens_map={},
            get_vocab=lambda: {"cat": 1},
            get_added_vocab=lambda: {},
            init_kwargs={},
        ),
        to_dict=lambda: {"image_size": 224},
    )
    adapter._processor_digest = adapter._compute_processor_digest(adapter._processor)

    assert adapter.processor_identity is None


def test_strict_tokenizer_digest_rejects_unstable_or_ambiguous_vocabularies() -> None:
    assert _strict_tokenizer_digest(SimpleNamespace()) is None
    assert (
        _strict_tokenizer_digest(
            SimpleNamespace(
                get_vocab=lambda: (_ for _ in ()).throw(RuntimeError("unavailable"))
            )
        )
        is None
    )
    assert _strict_tokenizer_digest(SimpleNamespace(get_vocab=lambda: [])) is None
    assert (
        _strict_tokenizer_digest(
            SimpleNamespace(get_vocab=lambda: {"cat": True}, get_added_vocab=lambda: {})
        )
        is None
    )
    assert (
        _strict_tokenizer_digest(
            SimpleNamespace(
                get_vocab=lambda: {"cat": 1},
                get_added_vocab=lambda: (_ for _ in ()).throw(ValueError("bad")),
            )
        )
        is None
    )
    assert (
        _strict_tokenizer_digest(
            SimpleNamespace(
                get_vocab=lambda: {"cat": 1}, get_added_vocab=lambda: ["bad"]
            )
        )
        is None
    )


def test_strict_processor_digest_uses_image_config_and_fails_closed() -> None:
    assert (
        _strict_processor_digest(
            SimpleNamespace(to_dict=lambda: (_ for _ in ()).throw(RuntimeError("bad"))),
            tokenizer_sha256="tokenizer",
            load_kwargs={},
        )
        is None
    )
    assert (
        _strict_processor_digest(
            SimpleNamespace(
                image_processor=SimpleNamespace(to_dict=lambda: {"size": 224})
            ),
            tokenizer_sha256="tokenizer",
            load_kwargs={"revision": "main"},
        )
        is not None
    )
    assert (
        _strict_processor_digest(
            SimpleNamespace(
                image_processor=SimpleNamespace(
                    to_dict=lambda: (_ for _ in ()).throw(ValueError("bad"))
                )
            ),
            tokenizer_sha256="tokenizer",
            load_kwargs={},
        )
        is None
    )
    assert (
        _strict_processor_digest(
            SimpleNamespace(image_processor=SimpleNamespace(to_dict=lambda: {})),
            tokenizer_sha256="tokenizer",
            load_kwargs={},
        )
        is None
    )


def test_json_safe_processor_value_normalizes_sets_deterministically() -> None:
    assert _json_safe_processor_value({"beta", "alpha"}) == ["alpha", "beta"]


def test_processor_identity_fails_closed_on_processor_and_config_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HF_Multimodal_Adapter()
    monkeypatch.setattr(
        adapter,
        "_require_processor",
        lambda: (_ for _ in ()).throw(RuntimeError("not loaded")),
    )
    assert adapter.processor_identity is None

    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"cat": 1},
        get_added_vocab=lambda: {},
        chat_template="{{ messages }}",
        init_kwargs={},
        special_tokens_map={},
    )
    adapter._processor = SimpleNamespace(tokenizer=tokenizer, to_dict=lambda: {})
    assert adapter.processor_identity is None


def test_hf_multimodal_processor_digest_normalizes_processor_metadata() -> None:
    adapter = HF_Multimodal_Adapter()

    class _SizeDict:
        def to_dict(self) -> dict[str, int]:
            return {"height": 224, "width": 224}

    processor = SimpleNamespace(
        name_or_path="qwen-like",
        tokenizer=SimpleNamespace(
            name_or_path="qwen-like",
            vocab_size=42,
            eos_token="<eos>",
            pad_token="<pad>",
        ),
        image_processor=SimpleNamespace(
            size=_SizeDict(),
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.25, 0.25, 0.25),
        ),
    )

    digest = adapter._compute_processor_digest(processor)

    assert isinstance(digest, str)
    assert len(digest) == 64


def test_hf_multimodal_processor_digest_normalizes_mapping_and_fallback_values() -> (
    None
):
    adapter = HF_Multimodal_Adapter()

    class _BrokenToDict:
        def to_dict(self) -> dict[str, int]:
            raise ValueError("metadata not serializable")

        def __str__(self) -> str:
            return "fallback-size"

    class _StringOnlyValue:
        def __str__(self) -> str:
            return "string-only-value"

    processor = SimpleNamespace(
        name_or_path="mapping-like",
        image_processor=SimpleNamespace(
            size={2: _BrokenToDict(), "height": 224},
            image_mean=[0.5, {"nested": (0.25, 0.75)}],
            image_std=_StringOnlyValue(),
        ),
    )

    digest = adapter._compute_processor_digest(processor)

    assert isinstance(digest, str)
    assert len(digest) == 64


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
