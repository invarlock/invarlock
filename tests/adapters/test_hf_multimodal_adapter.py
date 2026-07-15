from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from invarlock.adapters.hf_multimodal import HF_Multimodal_Adapter
from invarlock.core.exceptions import DependencyError, ModelLoadError
from tests.adapters._support_hf_multimodal import (
    _build_adapter,
    _FakeProcessor,
    _RetryingProcessor,
    _write_ppm,
)


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
        lambda **kwargs: (
            auto_strategy if kwargs.get("allow_direct_submodule") is False else strategy
        ),
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

    loaded = adapter.load_model("fake/model", device="cpu", trust_remote_code=False)

    assert loaded == {"model": {"loader": "direct-loader"}, "device": "cpu"}
    assert adapter._last_model_id == "fake/model"
    assert adapter._last_loader_strategy == "direct_submodule"
    assert adapter._last_loader_label == "direct-loader"
    assert calls == [
        (
            "direct-loader",
            "fake/model",
            {"load_device": "cpu", "trust_remote_code": False},
        )
    ]


def test_hf_multimodal_processor_reuses_only_compatible_model_identity_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HF_Multimodal_Adapter()
    strategy = SimpleNamespace(
        strategy="auto",
        loader_label="auto-loader",
        loader="auto-loader",
    )
    processor = _FakeProcessor()
    processor_calls: list[tuple[str, dict[str, object]]] = []

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeProcessor:
            processor_calls.append((model_id, kwargs))
            return processor

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=_AutoProcessor),
    )
    monkeypatch.setattr(
        "invarlock.adapters.hf_multimodal.resolve_core_loader_strategy",
        lambda **_kwargs: strategy,
    )
    monkeypatch.setattr(
        adapter,
        "_load_pretrained_model",
        lambda loader, model_id, **kwargs: object(),
    )
    monkeypatch.setattr(adapter, "_safe_to_device", lambda model, device: model)

    revision = "a" * 40
    adapter.load_model(
        "org/model",
        device="cpu",
        revision=revision,
        trust_remote_code=False,
        prefer_local_files_only=True,
        torch_dtype="float16",
        device_map="auto",
        collect_loading_info=True,
        chat_template_kwargs={"tokenize": False},
    )

    assert adapter._require_processor() is processor
    assert processor_calls == [
        (
            "org/model",
            {
                "revision": revision,
                "trust_remote_code": False,
                "local_files_only": True,
            },
        )
    ]


def test_hf_multimodal_processor_cache_is_scoped_to_model_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HF_Multimodal_Adapter()
    strategy = SimpleNamespace(
        strategy="auto",
        loader_label="auto-loader",
        loader="auto-loader",
    )
    processor_calls: list[tuple[str, dict[str, object]]] = []

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeProcessor:
            processor_calls.append((model_id, kwargs))
            return _FakeProcessor()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=_AutoProcessor),
    )
    monkeypatch.setattr(
        "invarlock.adapters.hf_multimodal.resolve_core_loader_strategy",
        lambda **_kwargs: strategy,
    )
    monkeypatch.setattr(
        adapter,
        "_load_pretrained_model",
        lambda loader, model_id, **kwargs: object(),
    )
    monkeypatch.setattr(adapter, "_safe_to_device", lambda model, device: model)

    revision_a = "a" * 40
    revision_b = "b" * 40
    adapter.load_model("org/model", device="cpu", revision=revision_a)
    processor_a = adapter._require_processor()
    adapter.load_model("org/model", device="cpu", revision=revision_b)
    processor_b = adapter._require_processor()

    assert processor_a is not processor_b
    assert processor_calls == [
        ("org/model", {"revision": revision_a}),
        ("org/model", {"revision": revision_b}),
    ]


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
        lambda **kwargs: (
            auto_strategy if kwargs.get("allow_direct_submodule") is False else strategy
        ),
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
