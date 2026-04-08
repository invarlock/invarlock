from __future__ import annotations

import sys
import types

import pytest

from invarlock.core.exceptions import ModelLoadError


def test_hf_bnb_uses_quantization_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    tr = types.ModuleType("transformers")

    class _Auto:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            calls["model_id"] = model_id
            calls["kwargs"] = dict(kwargs)
            return object()

    class _BitsAndBytesConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = dict(kwargs)

    tr.AutoModelForCausalLM = _Auto
    tr.BitsAndBytesConfig = _BitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "transformers", tr)

    from invarlock.plugins.hf_bnb_adapter import HF_BNB_Adapter

    adapter = HF_BNB_Adapter()
    adapter.load_model("nonexistent-model-id-for-test")

    kwargs = calls.get("kwargs")
    assert isinstance(kwargs, dict)
    assert "quantization_config" in kwargs
    assert "load_in_8bit" not in kwargs
    assert "load_in_4bit" not in kwargs


def test_hf_bnb_surfaces_checkpoint_quantization_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tr = types.ModuleType("transformers")

    class _Auto:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            raise ValueError(
                "The model is quantized with FineGrainedFP8Config but you are "
                "passing a BitsAndBytesConfig config."
            )

    class _BitsAndBytesConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = dict(kwargs)

    tr.AutoModelForCausalLM = _Auto
    tr.BitsAndBytesConfig = _BitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "transformers", tr)

    from invarlock.plugins.hf_bnb_adapter import HF_BNB_Adapter

    adapter = HF_BNB_Adapter()
    with pytest.raises(ModelLoadError) as excinfo:
        adapter.load_model(
            "org/fp8-checkpoint",
            quantization_config={"bits": 4},
        )

    exc = excinfo.value
    assert exc.code == "E201"
    assert "checkpoint quantization_config" in exc.message
    assert exc.details == {
        "model_id": "org/fp8-checkpoint",
        "checkpoint_quantization": "FineGrainedFP8Config",
        "requested_quantization": "BitsAndBytesConfig",
        "recommended_adapter": "hf_causal",
    }
