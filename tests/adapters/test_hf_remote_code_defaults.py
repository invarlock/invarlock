from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from invarlock.runtime_security import runtime_allowances_scope


def _clear_remote_code_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INVARLOCK_ALLOW_REMOTE_CODE", raising=False)


@pytest.mark.unit
def test_hf_bnb_uses_resolved_remote_code(monkeypatch: pytest.MonkeyPatch):
    transformers = types.ModuleType("transformers")

    class _AutoModelForCausalLM:
        pass

    class _BitsAndBytesConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = dict(kwargs)

    transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    transformers.BitsAndBytesConfig = _BitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    from invarlock.plugins import HF_BNB_Adapter

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

    with runtime_allowances_scope(allow_remote_code=True):
        adapter.load_model("demo/model", trust_remote_code=True)
        assert calls["kwargs"]["trust_remote_code"] is True


@pytest.mark.unit
def test_hf_awq_uses_resolved_remote_code(monkeypatch: pytest.MonkeyPatch):
    transformers = types.ModuleType("transformers")
    gptqmodel = types.ModuleType("gptqmodel")

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> dict[str, object]:
            return {"model_id": model_id, "kwargs": dict(kwargs)}

    transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "gptqmodel", gptqmodel)

    from invarlock.plugins import HF_AWQ_Adapter

    adapter = HF_AWQ_Adapter()
    monkeypatch.setattr(
        adapter,
        "_safe_to_device",
        lambda model, device, capabilities=None: model,
    )

    _clear_remote_code_env(monkeypatch)
    loaded = adapter.load_model("demo/model")
    assert loaded["kwargs"]["trust_remote_code"] is False
    assert loaded["kwargs"]["device_map"] == "auto"

    with runtime_allowances_scope(allow_remote_code=True):
        loaded = adapter.load_model("demo/model", trust_remote_code=True)
        assert loaded["kwargs"]["trust_remote_code"] is True


@pytest.mark.unit
def test_hf_gptq_uses_resolved_remote_code(monkeypatch: pytest.MonkeyPatch):
    gptqmodel = types.ModuleType("gptqmodel")

    class _GPTQModel:
        @staticmethod
        def load(model_id: str, **kwargs: object) -> dict[str, object]:
            return {"model_id": model_id, "kwargs": dict(kwargs)}

    gptqmodel.GPTQModel = _GPTQModel
    monkeypatch.setitem(sys.modules, "gptqmodel", gptqmodel)

    from invarlock.plugins import HF_GPTQ_Adapter

    adapter = HF_GPTQ_Adapter()
    monkeypatch.setattr(
        adapter,
        "_safe_to_device",
        lambda model, device, capabilities=None: model,
    )

    _clear_remote_code_env(monkeypatch)
    loaded = adapter.load_model("demo/model")
    assert loaded["kwargs"]["trust_remote_code"] is False

    with runtime_allowances_scope(allow_remote_code=True):
        loaded = adapter.load_model("demo/model", trust_remote_code=True)
        assert loaded["kwargs"]["trust_remote_code"] is True


@pytest.mark.unit
def test_gptqmodel_hub_compat_patch_bridges_transformers_512_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = types.ModuleType("transformers")
    transformers.utils = SimpleNamespace(hub=SimpleNamespace())
    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.create_repo = object()

    class _Api:
        def list_repo_tree(self) -> list[object]:
            return []

    huggingface_hub.HfApi = _Api
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    from invarlock.plugins import _patch_gptqmodel_transformers_hub_compat

    _patch_gptqmodel_transformers_hub_compat()

    assert transformers.utils.hub.create_repo is huggingface_hub.create_repo
    assert transformers.utils.hub.list_repo_tree.__self__.__class__ is _Api
