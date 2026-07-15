from __future__ import annotations

import sys
import types

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

    import invarlock.plugins as plugins
    from invarlock.plugins import HF_AWQ_Adapter

    adapter = HF_AWQ_Adapter()
    monkeypatch.setattr(
        plugins,
        "_gptqmodel_jit_toolchain_required",
        lambda device: device == "cuda",
    )
    runtime_calls: list[str] = []
    monkeypatch.setattr(
        plugins,
        "import_gptqmodel",
        lambda **kwargs: (
            runtime_calls.append(str(kwargs["require_jit_toolchain"])) or gptqmodel
        ),
    )
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
    adapter.load_model("demo/model", device="cuda")
    assert runtime_calls == ["False", "False", "True"]


@pytest.mark.unit
def test_hf_gptq_uses_resolved_remote_code(monkeypatch: pytest.MonkeyPatch):
    gptqmodel = types.ModuleType("gptqmodel")

    class _GPTQModel:
        @staticmethod
        def load(model_id: str, **kwargs: object) -> dict[str, object]:
            return {"model_id": model_id, "kwargs": dict(kwargs)}

    gptqmodel.GPTQModel = _GPTQModel
    monkeypatch.setitem(sys.modules, "gptqmodel", gptqmodel)

    import invarlock.plugins as plugins
    from invarlock.plugins import HF_GPTQ_Adapter

    adapter = HF_GPTQ_Adapter()
    monkeypatch.setattr(
        plugins,
        "_gptqmodel_jit_toolchain_required",
        lambda device: device == "cuda",
    )
    runtime_calls: list[str] = []
    validated_models: list[object] = []
    monkeypatch.setattr(
        plugins,
        "import_gptqmodel",
        lambda **kwargs: (
            runtime_calls.append(str(kwargs["require_jit_toolchain"])) or gptqmodel
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_safe_to_device",
        lambda model, device, capabilities=None: model,
    )
    monkeypatch.setattr(
        plugins,
        "validate_gptq_checkpoint_bindings",
        lambda model: validated_models.append(model),
    )

    _clear_remote_code_env(monkeypatch)
    loaded = adapter.load_model("demo/model")
    assert loaded["kwargs"]["trust_remote_code"] is False

    with runtime_allowances_scope(allow_remote_code=True):
        loaded = adapter.load_model("demo/model", trust_remote_code=True)
        assert loaded["kwargs"]["trust_remote_code"] is True
    adapter.load_model("demo/model", device="cuda")
    assert runtime_calls == ["False", "False", "True"]
    assert len(validated_models) == 3


@pytest.mark.unit
def test_gptqmodel_jit_preflight_policy_detects_explicit_and_auto_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import invarlock.plugins as plugins

    torch = types.ModuleType("torch")
    torch.cuda = type("Cuda", (), {"is_available": staticmethod(lambda: True)})()
    monkeypatch.setattr(
        plugins.importlib,
        "import_module",
        lambda name: (
            torch if name == "torch" else (_ for _ in ()).throw(AssertionError(name))
        ),
    )

    assert plugins._gptqmodel_jit_toolchain_required("cuda:0") is True
    assert plugins._gptqmodel_jit_toolchain_required("auto") is True
    assert plugins._gptqmodel_jit_toolchain_required("cpu") is False
