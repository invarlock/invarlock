from __future__ import annotations

import sys
import types

import pytest


def test_hf_bnb_uses_quantization_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    tr = types.ModuleType("transformers")

    class _Auto:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object):  # type: ignore[no-untyped-def]
            calls["model_id"] = model_id
            calls["kwargs"] = dict(kwargs)
            return object()

    class _BitsAndBytesConfig:
        def __init__(self, **kwargs: object):  # type: ignore[no-untyped-def]
            self.kwargs = dict(kwargs)

    tr.AutoModelForCausalLM = _Auto  # type: ignore[attr-defined]
    tr.BitsAndBytesConfig = _BitsAndBytesConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", tr)

    from invarlock.plugins.hf_bnb_adapter import HF_BNB_Adapter

    adapter = HF_BNB_Adapter()
    adapter.load_model("nonexistent-model-id-for-test")

    kwargs = calls.get("kwargs")
    assert isinstance(kwargs, dict)
    assert "quantization_config" in kwargs
    assert "load_in_8bit" not in kwargs
    assert "load_in_4bit" not in kwargs
