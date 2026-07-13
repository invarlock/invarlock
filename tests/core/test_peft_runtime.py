from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from invarlock import peft_runtime


class _DenseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace()


class _Config:
    def __init__(self) -> None:
        self.mapping: dict[type[Any], Any] | None = None

    def _register_custom_module(self, mapping: dict[type[Any], Any]) -> None:
        self.mapping = dict(mapping)


def _install_fake_peft_imports(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dispatch_default: Any,
) -> None:
    original_import = importlib.import_module
    layer = SimpleNamespace(dispatch_default=dispatch_default)

    def import_module(name: str, package: str | None = None) -> Any:
        if name == "peft.tuners.lora.layer":
            return layer
        return original_import(name, package)

    monkeypatch.setattr(peft_runtime.importlib, "import_module", import_module)


def test_dense_construction_uses_config_local_dispatch_before_class_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[type[Any]] = []
    optional_backend_calls = 0

    def dispatch_default(target: Any, *_args: Any, **_kwargs: Any) -> object:
        calls.append(type(target))
        return object()

    _install_fake_peft_imports(monkeypatch, dispatch_default=dispatch_default)
    config = _Config()

    def incompatible_awq_probe() -> None:
        nonlocal optional_backend_calls
        optional_backend_calls += 1
        raise ImportError("AwqGEMMQuantLinear was renamed")

    def construct(model: _DenseModel, selected: _Config) -> object:
        assert selected.mapping is not None
        # PEFT's per-config custom dispatcher is first. An installed optional
        # AWQ dispatcher with a renamed class is therefore never consulted.
        factory = selected.mapping.get(type(model.projection))
        if factory is None:
            incompatible_awq_probe()
            raise AssertionError("optional dispatcher unexpectedly returned")
        return factory(model.projection, "default", config=selected)

    result = peft_runtime.get_dense_peft_model(
        _DenseModel(), config, get_peft_model=construct
    )
    assert result is not None
    assert calls == [torch.nn.Linear]
    assert optional_backend_calls == 0


def test_real_peft_custom_dispatch_precedes_installed_optional_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peft = pytest.importorskip("peft")
    lora_model = importlib.import_module("peft.tuners.lora.model")

    def incompatible_awq_probe(*_args: Any, **_kwargs: Any) -> None:
        raise ImportError("AwqGEMMQuantLinear was renamed")

    monkeypatch.setattr(lora_model, "dispatch_awq", incompatible_awq_probe)
    config = peft.LoraConfig(r=2, lora_alpha=4, target_modules=["projection"])
    result = peft_runtime.get_dense_peft_model(
        _DenseModel(),
        config,
        get_peft_model=peft.get_peft_model,
    )
    assert type(result.base_model.model.projection).__module__.startswith(
        "peft.tuners.lora"
    )
    assert lora_model.dispatch_awq is incompatible_awq_probe


def test_dense_reload_uses_serialized_config_dispatch_before_optional_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optional_backend_calls = 0

    def dispatch_default(target: Any, *_args: Any, **_kwargs: Any) -> object:
        return {"target": target}

    _install_fake_peft_imports(monkeypatch, dispatch_default=dispatch_default)
    config = _Config()

    def incompatible_awq_probe() -> None:
        nonlocal optional_backend_calls
        optional_backend_calls += 1
        raise AttributeError("transformers hub API drift")

    def reload_adapter(
        model: _DenseModel,
        adapter_path: Path,
        *,
        config: _Config,
        is_trainable: bool,
        local_files_only: bool,
    ) -> object:
        assert adapter_path == Path("serialized-adapter")
        assert is_trainable is False
        assert local_files_only is True
        assert config.mapping is not None
        factory = config.mapping.get(type(model.projection))
        if factory is None:
            incompatible_awq_probe()
            raise AssertionError("optional dispatcher unexpectedly returned")
        return factory(model.projection, "default", config=config)

    result = peft_runtime.load_dense_peft_model(
        _DenseModel(),
        config,
        Path("serialized-adapter"),
        from_pretrained=reload_adapter,
        is_trainable=False,
        local_files_only=True,
    )

    assert result is not None
    assert optional_backend_calls == 0


def test_real_peft_dense_reload_skips_installed_optional_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peft = pytest.importorskip("peft")
    initial_base = _DenseModel()
    initial_base.config = {}
    initial_config = peft.LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["projection"],
    )
    wrapped = peft_runtime.get_dense_peft_model(
        initial_base,
        initial_config,
        get_peft_model=peft.get_peft_model,
    )
    wrapped.save_pretrained(
        tmp_path,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    serialized_config = peft.LoraConfig.from_pretrained(
        tmp_path,
        local_files_only=True,
    )
    lora_model = importlib.import_module("peft.tuners.lora.model")

    def incompatible_awq_probe(*_args: Any, **_kwargs: Any) -> None:
        raise AttributeError("transformers hub API drift")

    monkeypatch.setattr(lora_model, "dispatch_awq", incompatible_awq_probe)
    reload_base = _DenseModel()
    reload_base.config = {}
    reloaded = peft_runtime.load_dense_peft_model(
        reload_base,
        serialized_config,
        tmp_path,
        from_pretrained=peft.PeftModel.from_pretrained,
        is_trainable=False,
        local_files_only=True,
    )

    assert type(reloaded.base_model.model.projection).__module__.startswith(
        "peft.tuners.lora"
    )
    assert lora_model.dispatch_awq is incompatible_awq_probe


def test_dense_dispatch_is_per_config_and_safe_for_concurrent_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_peft_imports(
        monkeypatch,
        dispatch_default=lambda *_args, **_kwargs: object(),
    )

    def construct(index: int) -> _Config:
        config = _Config()
        result = peft_runtime.get_dense_peft_model(
            _DenseModel(),
            config,
            get_peft_model=lambda _model, _config: index,
        )
        assert result == index
        return config

    with ThreadPoolExecutor(max_workers=4) as executor:
        configs = list(executor.map(construct, range(12)))
    assert all(config.mapping for config in configs)
    assert len({id(config.mapping) for config in configs}) == len(configs)


@pytest.mark.parametrize(
    ("mutate", "diagnostic"),
    [
        (lambda model: setattr(model, "is_quantized", True), "model.is_quantized"),
        (
            lambda model: setattr(model.config, "quantization_config", {}),
            "config.quantization_config",
        ),
        (lambda model: setattr(model, "hf_quantizer", object()), "model.hf_quantizer"),
        (
            lambda model: setattr(model, "quantization_method", "awq"),
            "model.quantization_method",
        ),
    ],
)
def test_dense_boundary_rejects_quantized_models_before_peft_dispatch(
    mutate: Any,
    diagnostic: str,
) -> None:
    model = _DenseModel()
    mutate(model)
    with pytest.raises(peft_runtime.PeftRuntimeError, match=diagnostic):
        peft_runtime.get_dense_peft_model(
            model,
            _Config(),
            get_peft_model=lambda *_args: pytest.fail("must not construct"),
        )


@pytest.mark.parametrize(
    "packed_attribute",
    ("qweight", "qzeros", "scales", "packed_weight", "packed_weights"),
)
def test_dense_boundaries_reject_packed_module_attributes(
    packed_attribute: str,
) -> None:
    class Packed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            setattr(self, packed_attribute, object())

    model = _DenseModel()
    model.projection = Packed()
    with pytest.raises(peft_runtime.PeftRuntimeError, match="packed-module"):
        peft_runtime.get_dense_peft_model(
            model, _Config(), get_peft_model=lambda *_args: None
        )
    with pytest.raises(peft_runtime.PeftRuntimeError, match="packed-module"):
        peft_runtime.load_dense_peft_model(
            model,
            _Config(),
            Path("adapter"),
            from_pretrained=lambda *_args, **_kwargs: pytest.fail("must not reload"),
        )


def test_dense_boundaries_reject_quantized_tensor_weights() -> None:
    class QuantizedProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "weight",
                torch.quantize_per_tensor(
                    torch.randn(4, 4),
                    scale=0.1,
                    zero_point=0,
                    dtype=torch.qint8,
                ),
            )

    model = _DenseModel()
    model.projection = QuantizedProjection()
    with pytest.raises(peft_runtime.PeftRuntimeError, match="packed-module"):
        peft_runtime.get_dense_peft_model(
            model,
            _Config(),
            get_peft_model=lambda *_args: pytest.fail("must not construct"),
        )
    with pytest.raises(peft_runtime.PeftRuntimeError, match="packed-module"):
        peft_runtime.load_dense_peft_model(
            model,
            _Config(),
            Path("adapter"),
            from_pretrained=lambda *_args, **_kwargs: pytest.fail("must not reload"),
        )


def test_dense_boundaries_reject_unreadable_packed_metadata() -> None:
    class UnreadableProjection(torch.nn.Module):
        @property
        def weight(self) -> object:
            raise RuntimeError("unreadable packed storage")

    model = _DenseModel()
    model.projection = UnreadableProjection()
    with pytest.raises(peft_runtime.PeftRuntimeError, match="inspection failed"):
        peft_runtime.get_dense_peft_model(
            model,
            _Config(),
            get_peft_model=lambda *_args: pytest.fail("must not construct"),
        )
    with pytest.raises(peft_runtime.PeftRuntimeError, match="inspection failed"):
        peft_runtime.load_dense_peft_model(
            model,
            _Config(),
            Path("adapter"),
            from_pretrained=lambda *_args, **_kwargs: pytest.fail("must not reload"),
        )


def test_dense_boundary_fails_closed_on_incompatible_peft_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_peft_imports(monkeypatch, dispatch_default=None)
    with pytest.raises(peft_runtime.PeftRuntimeError, match="API is incompatible"):
        peft_runtime.get_dense_peft_model(
            _DenseModel(), _Config(), get_peft_model=lambda *_args: None
        )

    _install_fake_peft_imports(
        monkeypatch,
        dispatch_default=lambda *_args, **_kwargs: None,
    )
    config = _Config()

    def construct(model: _DenseModel, selected: _Config) -> object:
        assert selected.mapping is not None
        return selected.mapping[torch.nn.Linear](
            model.projection, "default", config=selected
        )

    with pytest.raises(peft_runtime.PeftRuntimeError, match="rejected"):
        peft_runtime.get_dense_peft_model(
            _DenseModel(), config, get_peft_model=construct
        )


def test_dense_boundary_wraps_construction_failure_without_global_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_peft_imports(
        monkeypatch,
        dispatch_default=lambda *_args, **_kwargs: object(),
    )

    def fail(_model: Any, _config: Any) -> None:
        raise ValueError("bad target")

    with pytest.raises(
        peft_runtime.PeftRuntimeError,
        match="dense LoRA construction failed: ValueError",
    ):
        peft_runtime.get_dense_peft_model(_DenseModel(), _Config(), get_peft_model=fail)
