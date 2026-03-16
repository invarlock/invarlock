from __future__ import annotations

import sys
import types

import pytest


def _clear_remote_code_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "INVARLOCK_TRUST_REMOTE_CODE",
        "TRUST_REMOTE_CODE_BOOL",
        "ALLOW_REMOTE_CODE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.unit
def test_hf_causal_onnx_uses_opt_in_remote_code(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []

    optimum = types.ModuleType("optimum")
    onnxruntime = types.ModuleType("optimum.onnxruntime")

    class _ORTModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object):  # type: ignore[no-untyped-def]
            calls.append({"model_id": model_id, "kwargs": dict(kwargs)})
            return object()

    onnxruntime.ORTModelForCausalLM = _ORTModelForCausalLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "optimum", optimum)
    monkeypatch.setitem(sys.modules, "optimum.onnxruntime", onnxruntime)

    from invarlock.adapters.hf_causal_onnx import HF_Causal_ONNX_Adapter

    _clear_remote_code_env(monkeypatch)
    adapter = HF_Causal_ONNX_Adapter()
    adapter.load_model("demo/model")
    assert calls[-1]["kwargs"]["trust_remote_code"] is False

    monkeypatch.setenv("INVARLOCK_TRUST_REMOTE_CODE", "1")
    adapter.load_model("demo/model")
    assert calls[-1]["kwargs"]["trust_remote_code"] is True


@pytest.mark.unit
def test_hf_bnb_uses_resolved_remote_code(monkeypatch: pytest.MonkeyPatch):
    transformers = types.ModuleType("transformers")

    class _AutoModelForCausalLM:
        pass

    class _BitsAndBytesConfig:
        def __init__(self, **kwargs: object):  # type: ignore[no-untyped-def]
            self.kwargs = dict(kwargs)

    transformers.AutoModelForCausalLM = _AutoModelForCausalLM  # type: ignore[attr-defined]
    transformers.BitsAndBytesConfig = _BitsAndBytesConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    from invarlock.plugins.hf_bnb_adapter import HF_BNB_Adapter

    calls: dict[str, object] = {}
    adapter = HF_BNB_Adapter()
    monkeypatch.setattr(
        adapter,
        "_load_pretrained_model",
        lambda loader, model_id, **kwargs: calls.update(kwargs=kwargs) or object(),
    )
    monkeypatch.setattr(adapter, "_resolve_device", lambda device: device)

    _clear_remote_code_env(monkeypatch)
    adapter.load_model("demo/model")
    assert calls["kwargs"]["trust_remote_code"] is False

    adapter.load_model("demo/model", trust_remote_code=True)
    assert calls["kwargs"]["trust_remote_code"] is True


@pytest.mark.unit
def test_hf_awq_uses_resolved_remote_code(monkeypatch: pytest.MonkeyPatch):
    autoawq = types.ModuleType("autoawq")

    class _AutoAWQForCausalLM:
        @staticmethod
        def from_quantized(model_id: str, **kwargs: object):  # type: ignore[no-untyped-def]
            return {"model_id": model_id, "kwargs": dict(kwargs)}

    autoawq.AutoAWQForCausalLM = _AutoAWQForCausalLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autoawq", autoawq)

    from invarlock.plugins.hf_awq_adapter import HF_AWQ_Adapter

    adapter = HF_AWQ_Adapter()
    monkeypatch.setattr(
        adapter,
        "_safe_to_device",
        lambda model, device, capabilities=None: model,
    )

    _clear_remote_code_env(monkeypatch)
    loaded = adapter.load_model("demo/model")
    assert loaded["kwargs"]["trust_remote_code"] is False

    monkeypatch.setenv("INVARLOCK_TRUST_REMOTE_CODE", "1")
    loaded = adapter.load_model("demo/model")
    assert loaded["kwargs"]["trust_remote_code"] is True


@pytest.mark.unit
def test_hf_gptq_uses_resolved_remote_code(monkeypatch: pytest.MonkeyPatch):
    auto_gptq = types.ModuleType("auto_gptq")

    class _AutoGPTQForCausalLM:
        @staticmethod
        def from_quantized(model_id: str, **kwargs: object):  # type: ignore[no-untyped-def]
            return {"model_id": model_id, "kwargs": dict(kwargs)}

    auto_gptq.AutoGPTQForCausalLM = _AutoGPTQForCausalLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "auto_gptq", auto_gptq)

    from invarlock.plugins.hf_gptq_adapter import HF_GPTQ_Adapter

    adapter = HF_GPTQ_Adapter()
    monkeypatch.setattr(
        adapter,
        "_safe_to_device",
        lambda model, device, capabilities=None: model,
    )

    _clear_remote_code_env(monkeypatch)
    loaded = adapter.load_model("demo/model")
    assert loaded["kwargs"]["trust_remote_code"] is False

    loaded = adapter.load_model("demo/model", trust_remote_code=True)
    assert loaded["kwargs"]["trust_remote_code"] is True
